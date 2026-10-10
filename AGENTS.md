# AGENTS.md

## Start here

**Read [`docs/status.md`](docs/status.md) first.** It is the handoff document: where the project
actually stands, which PRs are open, the ablation results and the decisions taken from them, the
GPU training plan, and the ordered next steps. This file covers how the codebase behaves;
`status.md` covers what to do next.

Whetstone is a from-scratch PyTorch LLM training/inference codebase (no web server, no
long-running service) targeting bilingual zh/en **verifiable tasks** (arithmetic, code). The core
workflow is a set of CLI scripts:

- `python3 -m datatools.fetch_evals` → `python3 -m datatools.prepare <spec>` → `train_tokenizer.py`
 — the data pipeline, which runs *before* any training. See `docs/corpus-plan.md`.
- `train_tokenizer.py --data <prepared jsonl> --out <dir> --vocab_size N` — train a BPE tokenizer
 on prepared corpus files. It no longer reads a hardcoded path. **Digits are split one per token**
 (a `Digits` pre-tokenizer before `ByteLevel`), as in Llama and Qwen: byte-level BPE alone made
 `1987` one token, `2024` `20|24` and `12345` `12|345`, which leaves arithmetic no consistent unit.
 Keep it.
- `pretrain.py` → `sft.py` → `distill.py` → `dpo.py` → `grpo.py` — the five-stage training pipeline
 (Pretrain → SFT → real Knowledge Distillation → DPO → GRPO/RLVR). `distill.py` does real KD (frozen
 teacher, CE + temperature-scaled KL on assistant tokens via `losses.kd_loss`), not the old fake
 special-token-weighted version. `dpo.py` runs standard Bradley-Terry DPO with a frozen reference
 model (`losses.dpo_loss` + `sequence_logprobs`). `grpo.py` runs on-policy GRPO against a
 rule-based environment in `envs/` (no reward model, no TRL dependency).
- `eval_ppl.py` — perplexity evaluation. `chat.py` — interactive inference REPL.

**README.md is the source of truth** for CLI commands/flags for every stage (including the CPU
smoke-test walkthrough using `tests/fixtures/`) — consult it and each script's `argparse` block
rather than duplicating commands here.

### Tests

- `tests/` holds CPU-only unit tests (`pytest>=8.0`, already in `requirements.txt`) plus small
 JSONL fixtures under `tests/fixtures/` (`pretrain_tiny.jsonl`, `sft_tiny.jsonl`,
 `preference_tiny.jsonl`) used both by the tests and by the README's CPU smoke-test commands for
 the pretrain/SFT/KD/DPO stages. The GRPO stage needs no fixture: `envs/` generates its own
 prompts and data (`python3 -m envs.generate_data --split {sft,preference,eval}`).
- Run the whole suite with `python -m pytest tests/ -q`. No GPU, network, or external data is
  required.
- CI (`.github/workflows/ci.yml`) installs CPU-wheel `torch` + `requirements.txt` and runs the same
  `pytest tests/ -q` on every push/PR.

### Environment / running caveats (non-obvious)

- **Python 3.12+ and a venv are required** (the system Python on macOS is 3.9, which
 `transformers` 5.x will not run on). `uv venv --python 3.12 && uv pip install -r requirements.txt`.
- **The HF `datasets` library is required by the data pipeline** (`prepare`, `fetch_evals`) and is
 in `requirements.txt`, imported lazily so tests stay offline. If it is missing, `import datasets`
 silently resolves to the repo's git-ignored `datasets/` data directory as a namespace package,
 and fails with `cannot import name 'load_dataset' from 'datasets' (unknown location)`.
- **No HF token is configured by default**, and `bigcode/starcoderdata` (the `code` source) and
 `big_math` are gated: accept the terms on the dataset page, then `hf auth login`. Everything
 else in `configs/mixture_v1.json` and `fetch_evals --decontamination_only` is open.
- **Always set `HF_HUB_OFFLINE=1`.** The tokenizer is committed, but `transformers` still phones
  home to check for updates on every load: the suite takes 47s without it and 16s with it, of
  which only ~4s is CPU.
- **`--device` auto-selects `cuda` > `mps` > `cpu`** via `train_utils.resolve_device()`, in all
  five training scripts plus `eval_ppl.py` and `chat.py`. Do not reintroduce a bare
  `"cuda" if torch.cuda.is_available() else "cpu"` default — that silently costs ~6x on Apple
  Silicon (measured: 1.4k token/s on cpu vs 9.1k on mps+bf16 for a 100M model).
- **Only `--dtype float16` enables GradScaler.** `bfloat16` needs no loss scaling and will not
  create one (see `build_autocast_scaler`) — that is intended, not a missing feature.
- **Use bf16, not fp16, on MPS.** Measured on an M5 Pro, fp16 autocast moved a 100M model's loss
  from -0.3278 to +0.0018 while bf16 held at -0.3276.
- **At real scale, pretraining reads a pre-tokenized `.bin`, not JSONL.**
 `python3 -m datatools.tokenize_corpus <jsonl> --tokenizer <dir> --out <prefix>` writes
 `<prefix>.bin` (documents packed as `bos + text + eos`, `uint16` up to a 65536 vocab),
 `<prefix>.idx` (`uint64` document offsets) and `<prefix>.meta.json`. `pretrain.py` goes through
 `dataset.build_pretrain_dataset`, which picks `MemmapPretrainDataset` for a `.bin` path and the
 old `PretrainDataset` (whole file in a list, tokenized per item) otherwise, for train and val.
 **`PretrainDataset` keeps only the first `max_seq_len` tokens of each document**: on the smoke
 corpus at 512 it trains on 28% of the tokens and 1% of the book tokens, which rewrites the
 mixture (books 6.4% → 0.2%). Mixture ablations are only meaningful on `.bin` data.
 - The memmap dataset **raises** on a tokenizer fingerprint mismatch and on a `.bin` whose size
 disagrees with its meta (a truncated copy). Both are intended: ids from another vocabulary
 train silently on garbage. Retraining the tokenizer means re-running `tokenize_corpus`.
 - Windows are `max_seq_len` tokens at stride `max_seq_len - 1`, so every token is a target
 exactly once per epoch and `X`/`Y` stay `max_seq_len - 1` long, like the JSONL reader. A test
 asserts the stored documents equal what `PretrainDataset` feeds the model; keep them equal.
 - `SFTDataset` / `PreferenceDataset` still load whole files, which is fine at SFT scale.
- **Attention goes through `F.scaled_dot_product_attention`, except on MPS.**
 - Only a full forward with no cache and no padding mask (`is_causal=True`) and single-token
 decode (no mask) pass `attn_mask=None`; that is what lets CUDA pick flash attention. Do not
 pass `is_causal=True` with a cache: SDPA aligns it top-left, so a multi-token continuation
 would see the wrong keys. Cached continuation and padding build an explicit boolean mask.
 - `Attention.use_sdpa = False` selects the explicit-score reference; tests compare the two
 paths on every masking case, plus training gradients.
 - MPS always takes the explicit path on purpose. Measured on an M5 Pro, torch 2.14, bf16:
 SDPA was 8% slower (29M proxy, seq 512) and saved only 15% memory at seq 2048, so it
 still materializes the scores there. On CPU it halves activation memory at seq 2048.
- **No datasets or checkpoints are committed.** Training scripts expect JSONL under `datasets/`,
  which is git-ignored along with `results*/` and `*.pth`. Build real data with
  `datatools.prepare`, or generate synthetic task data with `envs.generate_data`; the committed
  `tests/fixtures/*.jsonl` are enough for a CPU smoke of every stage.
- **`tokenizer/zh_6400/` is a legacy tokenizer**, committed so the CPU tests and smoke runs work
  (vocab 6400, bos `<s>`, eos `</s>`, pad `<unk>`). It was trained on Chinese only and compresses
  code at 2.23 chars/token against 4.00 for English prose. **The formal tokenizer is
  `tokenizer/v1_32k/`** (vocab 32768, same specials and chat template, digits split): the ablation's
  v32k, kept as evaluated rather than retrained on the formal corpus. The formal mixture is
  `configs/mixture_v2.json` (zh 20%, 10B tokens counted in v1_32k); `docs/status.md` §4 step 4 has
  the ablation behind both. **Checkpoints do not survive a tokenizer change** —
  `resolve_model_config` raises on a `vocab_size` mismatch, which is intended.
- **`chat.py` is an interactive REPL** (`input()`), so pipe input for non-interactive runs, e.g.
 `printf 'question\nquit\n' | python3 chat.py --save_dir results --model_mode 1 --device cpu`.
 `--model_mode` selects the checkpoint: 0=`pretrain*.pth`, 1=`sft*.pth`, 2=`distill*.pth`,
 3=`dpo*.pth`, 4=`grpo*.pth`, and it falls back to `*_final.pth` filenames. `--save_dir` already
 defaults to `results`, matching the other stages' `--save_dir results`.
- **`--use_wandb True` requires `swanlab`** (imported lazily, not installed by default). Leave
 wandb off unless you install it. All five training scripts (`pretrain.py`/`sft.py`/`distill.py`/
 `dpo.py`/`grpo.py`) support `--use_wandb`/`--wandb_project` via `train_utils.init_wandb_if_needed`.
 `grpo.py` additionally appends every step's metrics to `{save_dir}/grpo_metrics.jsonl`, so RL
 curves can be plotted with no tracker installed.
- Installed with a recent major `transformers` (5.x) and `torch` 2.x CPU; the model code (custom
  `PreTrainedModel`/`PretrainedConfig` subclasses) is compatible with these.
- **Model architecture is CLI-configurable, and checkpoints carry their own architecture.**
 `train_utils.add_model_args(parser)` adds `--tokenizer_path` plus `--dim`/`--n_layers`/
 `--n_heads`/`--n_kv_heads`/`--hidden_dim`/`--multiple_of`/`--norm_eps`/`--rope_theta`/`--dropout`
 to all five training scripts and to `eval_ppl.py`/`chat.py`. Every arch flag defaults to `None`
 so `resolve_model_config(args, vocab_size, checkpoint_path=...)` can apply the precedence
 **explicit CLI > checkpoint > `LLMConfig` default**. Never hardcode `LLMConfig(...)` in a script
 again.
  - `n_heads` is **not** recoverable from tensor shapes (`head_dim = dim // n_heads`, so `wq` is
    always `dim x dim`; only the kv/q ratio is visible). `save_final_weights` therefore writes a
    `*.config.json` sidecar next to each bare `*_final.pth`; resolving from shapes alone warns
    that `n_heads` was assumed.
  - A checkpoint/tokenizer `vocab_size` mismatch raises. Retraining the tokenizer invalidates old
    weights — expect this when swapping corpora.
  - `load_weights` now warns on missing/unexpected keys: `strict=False` raises on shape mismatch
    but silently tolerates *absent* keys, which would leave whole layers randomly initialized.
  - `distill.py` resolves teacher and student architectures independently from their own
    checkpoints, so cross-size KD works; they only need a shared vocab.
- **The data pipeline lives in `datatools/`.** `python3 -m datatools.prepare <spec>` runs
 pull → filter → exact-dedup → decontaminate → split → manifest from a mixture spec
 (`configs/mixture_v1.json`). Individual stages are also CLIs: `stats`, `filters` (library only),
 `dedup`, `decontaminate`, `split`, `tokenizer_stats`.
  - `datatools/records.py` is the shared schema layer: it detects `text` / `conversations` /
    `prompt+chosen+rejected` / `question+answer` automatically, so **no tool takes a `--schema`
    flag**. `record_text` joins a record for stats/dedup/split; `record_parts` keeps the pieces
    separate for decontamination (the joined form inserts `=>` and role prefixes that never occur
    in natural text and would block n-gram matches); `prompt_text` isolates the input side.
 - **The whole pipeline is streaming.** Mixture weights are in tokens while corpora are published
 in documents and bytes, so `prepare` tokenizes as it pulls and stops when a source's share is
 met. At 10B tokens the corpus is ~30GB; don't add a stage that materializes it.
 - **Run `prepare --probe 3` before a real `prepare`.** Sources are pulled serially, so a broken
 source (gated repo, wrong config name, renamed field) otherwise only fails after every source
 before it has finished. The probe reports all of them at once and exits 1 if any is not `ok`.
 - **`prepare` resumes per source and locks `--out_dir`.** A source is pulled into
 `<name>.jsonl.partial`, renamed when complete, then marked by `<name>.jsonl.done.json` (its
 report, the file size, and the source settings, budget and tokenizer fingerprint it was pulled
 with). Rerunning the same command reuses every source whose marker still matches and pulls the
 rest from their start; there is no checkpoint inside a source. A second run on the same
 directory exits instead of starting: two runs truncate each other's files, which is how the
 first formal build lost 3.6 hours of zh_web while the hub client was riding out an outage.
 `--only NAME...` pulls just those sources and stops before decontamination; markers do not
 depend on the directory, so a source pulled elsewhere can be moved into `sources/` with its
 marker. **Never send the main HF token to a mirror.** The formal build pulls the five open
 sources through `hf-mirror.com` with `HF_TOKEN_PATH=/nonexistent`. The gated starcoderdata needs
 a token, and the only one that may reach a mirror is a fine-grained token made for the build
 that can read nothing but public gated repos (no per-namespace permissions), deleted afterwards.
 - **`prepare`'s `__main__` runs the atexit handlers and then calls `os._exit`, on purpose.**
 pyarrow 25 deadlocks in a static thread pool's destructor if a parquet read is still in flight
 when the process exits; a probe always is in that state (it abandons each stream after a few
 rows), and Ctrl-C mid-source can be. Don't turn it back into a plain `main()` call.
 - `to_record` **projects** each row onto its schema's fields and drops every upstream column;
 `prepare_source` then adds a `source` tag (needed to split merged val data per language). Do
 not go back to passing rows through: FineWeb2-HQ ships a 768-float embedding per document,
 which made the Chinese slice 9.3x the size of its text.
 - `to_record` also applies the source's `cleaners` (`datatools/cleaners.py`, named in the spec,
 unknown names raise) to pretrain text, so filters, exact dedup and token counts see the
 cleaned document. `starcoder_metadata` removes the `<reponame>…<filename>…<gh_stars>…` first
 line that 49% of starcoderdata files carry, and only when that whole line is metadata: a
 `'-f <filename>'` inside the code is content.
 - **`max_repetition` is off for the code source on purpose.** Character 10-gram repetition
 grows with file length (indentation, boilerplate): on the smoke corpus it rejected 2% of code
 files under 2k chars and 57% of those over 20k, nearly all ordinary source. Don't re-enable it
 for consistency with the prose sources.
 - `duplicate_line_ratio` counts only lines with a letter or digit: a lone `"""` or `)` is
 structure, and counting it rejected docstring-heavy code. Generated code (Django migrations,
 protobuf) is deliberately **not** filtered: marker-based rules also hit Colab/nbdev exports,
 which are hand-written.
 - `hf.columns` (parquet sources only; Gutenberg is `jsonl.gz`) limits which columns are
 downloaded. Parquet streams whole row groups, so without it zh_web also downloads every
 embedding before the first row comes out. Projection alone is not enough: fsspec reads 5MiB
 past every read, straight into the dropped embedding column, so `hf_load_options` sets a 64KB
 read-ahead for projected sources (per 1000-row group: 25.6MB unprojected, 8.7MB projected
 with the default read-ahead, 3.8MB now). `SourceSpec` rejects a `columns` list that omits
 `text_field` or a `where` root, since that would reject every row. `where` matches upstream
 metadata by dotted path (books keep `metadata.language == "en"`; ~5% of Gutenberg is not).
  - `datatools/minhash.py` is a self-contained MinHash+LSH implementation on numpy (no
    `datasketch`); its permutation coefficients are bounded so uint64 arithmetic never wraps —
    don't "simplify" that away. LSH proposes candidates and every candidate is verified against
    the full signature, so banding only trades recall for speed. It holds ~1KB per document, so
    near-dedup is bounded to 1–2M docs and is deliberately **not** part of `prepare`'s pass.
  - `datatools/decontaminate.py` is the real contamination check (13-gram + optional LCS 0.6,
 following SmolLM2); `dedup --against` is only exact prompt equality. CJK is split per
 character and other letter/digit runs per word; **punctuation and symbols are not units**.
 Questions/solutions shorter than `n` are indexed at their own length down to `MIN_GRAM` (8);
 **answers shorter than `n` only match exactly**. Both rules came from auditing the first real
 smoke corpus: with punctuation units and own-length answers, 3.9% of documents were removed and
 nearly all were false positives (MMLU's `1,2,3`, MATH's `\frac{3}{2}`, a run of dashes matching
 every markdown table). `finalize` reports the eval items that removed the most documents
 (`manifest.json` → `split.top_matches`, also printed): one item removing many documents is
 the signature of a generic phrase, not a leak. Check it before loosening either rule.
 The LCS gate is computed **at the shared n-grams** (the shorter text aligned against the
 longer, ±`LCS_CONTEXT` units), not over the whole page: a whole-page LCS lets a short item
 collect its words by chance, and the old 2000-unit truncation never compared a leak past
 that point in a book. On the smoke corpus both remaining 13-gram hits (a formula sheet
 sharing a determinant with a MATH-500 problem; the Declaration of Independence quoted by an
 MMLU question) score 0.51 / 0.59 and are kept.
  - `prepare`'s report is meant to be trustworthy: `fill < 100%` plus `ran out of data` means a
    source was silently down-weighted, and `kept%` excludes records pulled into the tokenization
    batch but never emitted. Don't regress either.
  - `datatools/fetch_evals.py` pulls benchmarks into the repo's `{"question","answer"}` schema, so
    one file serves both `prepare`'s `decontaminate.against` and `grpo.py --eval_path`. Converters
 are pure functions tested offline against recorded rows — update the recorded row when a
 field name changes upstream rather than loosening the converter. Code sets (HumanEval, MBPP)
 carry their tests and put the reference code in `answer`, not `solution`: short answers only
 match exactly, whereas a `solution` is indexed at its own length, and a 10-unit idiom like
 `return [x for x in strings if substring in x]` would then flag every file that uses it. **A mixture spec with an
    empty `decontaminate.against` silently checks nothing**, so run `fetch_evals
    --decontamination_only --update_spec <spec>` before `prepare`.
  - **Multiple-choice items (TAL-SCQ5K, MMLU) put their options in `question`** (`stem\nA. …`), and
    `answer` is the correct option's text. A bare stem such as `下列说法正确的是．` is indexed at its
    own length and matched every page using the phrase: 136 of the ablation pool's 252 removals came
    from two such items. Items whose stem has no letter or digit (an image upstream) are skipped,
    since their option letters alone reach `MIN_GRAM`. Removals on the pool fell to 57.
- **Data ablations run through `run_ablation.py <spec> {plan,run,pool,tokenizers,data,train,probe,report}`**
 (`configs/ablation_v1.json`; the data side is `datatools/ablation.py`, the evaluations `probes.py`).
 Every stage skips finished outputs, so a killed run resumes by repeating the command.
 - Every arm is cut from **one pool** that `prepare` pulls once from a derived spec, taking each
 source's **first** documents in pool order: the 15% arm's Chinese pages are a prefix of the 30%
 arm's. Don't sample arms independently; they would then also differ in which pages they drew.
 - Quotas and `tokens_per_arm` are counted **in the arm's own tokenizer, bos/eos included**, so arms
 train on equal tokens and a share is what the model sees. `prepare` counts the pool in zh_6400,
 which compresses worse, hence `pool_margin`; pool targets are also divided by the train fraction,
 since arms and the tokenizer sample read only `train.jsonl`. A pool too small for a quota makes
 `cut_corpus` raise and leave no files: that is the intended signal to raise the margin.
 - Anything that moves the pool targets (arms, `tokens_per_arm`, margin, the mixture) makes
 `check_pool` refuse the existing pool. Delete `data_dir/pool` and pull again.
 - Compare arms on **per-source bits per byte** (`probes.bits_per_byte`), not on loss: a larger vocab
 predicts more information per token, so val loss is comparable only within one vocabulary.
 - `tests/test_run_ablation.py` runs every stage on tiny local data. It takes ~30s, nearly all of it
 subprocess startup (`prepare` and two `pretrain.py` runs each import torch and transformers).
- **Every stage records its run to `{save_dir}/runs/{run_id}/`** via `runlog.RunRecorder`
 (`meta.json` / `metrics.jsonl` / `summary.json`). Review with `analyze_runs.py`
 (`list` / `show` / `compare` / `plot`). No tracker service is involved; `swanlab` stays optional
 and orthogonal.
  - `meta.json` deliberately records the git commit, dirty flag and input-file fingerprints. A
    curve with no record of which data and commit produced it cannot be reviewed — don't drop
    them to "simplify".
  - Pass stage-specific metadata through `extra=`, which is **namespaced under `meta["extra"]`**.
    It used to be merged at the top level, and `grpo.py` passing `extra={"env": ...}` silently
    overwrote the recorded environment with a string.
  - `metrics.jsonl` is flushed per row so a killed run keeps its points, and `read_run` tolerates
    a truncated final line. Log train and eval as **separate rows**; folding an evaluation into
    the same row as the step's training metrics files reward/kl/entropy under the held-out split.
- **Held-out metrics come from `evaluate.py`**, wired through `--val_data_path` / `--val_every` /
 `--val_batches`. `evaluate_lm` is token-weighted, not batch-averaged, so the number does not
 drift with batch composition. `build_val_loader` reads the set in **one fixed permutation**, not
 file order: `--val_batches` evaluates only the first batches, and `prepare` writes splits grouped
 by source, so file order measured val loss on zh_web alone. Every evaluation still sees the same
 batches. For DPO watch `accuracy`, not loss: DPO loss keeps falling while
 the model merely sharpens an ordering it already had.
- **Common training CLI flags come from `train_utils.add_common_train_args(parser, **overrides)`**
 (`--save_dir`, `--epochs`, `--batch_size`, `--learning_rate`, `--device`, `--use_wandb`,
 `--wandb_project`, `--dtype`, `--num_workers`, `--accumulation_steps`, `--grad_clip`, `--log_step`,
 `--save_step`, `--max_seq_len`, `--data_path`, `--resume_from`, `--seed`, `--weight_decay`,
 `--adam_beta1`, `--adam_beta2`, `--max_steps`). Each script calls it
 first with its own default overrides, then adds its stage-specific extras (e.g. `dpo.py` adds
 `--policy_path`/`--beta`). Don't hand-roll these flags in a script — add/change them in
 `add_common_train_args` so all five scripts stay in sync. A stage that genuinely has no use for a
 shared flag passes `skip=(...)` rather than defining its own (`grpo.py` skips `--epochs`,
 `--accumulation_steps`, `--num_workers`, `--max_steps` because it is driven by `--rl_steps` over
 env-sampled prompts with no DataLoader). `--device` defaults to `resolve_device()` (cuda > mps > cpu)
 everywhere (train scripts, `eval_ppl.py`, `chat.py`).
- **Every optimizer comes from `train_utils.build_optimizer`**: AdamW that decays only parameters
  with `ndim >= 2` (decaying RMSNorm gains pulls them toward zero), betas `(0.9, 0.95)`, `fused=True`
  on CUDA, frozen parameters left out. GRPO uses it too. Don't construct `optim.AdamW` in a script.
- **pretrain / SFT / distill / DPO share one loop, `trainer.train`.** A stage supplies a step
  function, `batch -> (loss, {name: scalar tensor}, target tokens)` (`trainer.lm_step` for
  pretrain and SFT), plus an optional `evaluate()` closure. Don't reintroduce a per-script
  `train_epoch`.
  - Logging, evaluation and checkpoints run only **right after an optimizer update**, so
    `--log_step` / `--val_every` / `--save_step` count updates (`--log_step` used to count
    micro-batches). Checked per micro-batch, an evaluation due at update 1000 ran once per
    micro-batch of that accumulation window. An epoch's trailing partial window is an update too.
  - **`--resume_from` resumes exactly**: an epoch's order is the permutation seeded by
    `seed + epoch` (`EpochSampler`), a checkpoint's `step` is **batches of that epoch consumed**,
    the resumed sampler starts after them without loading them, and the checkpoint carries the
    RNG state. The DataLoader gets its own seeded `generator` on purpose: each new iterator draws
    a worker seed, from the global RNG otherwise, which shifts a resumed run's dropout stream by
    one draw. `tests/test_trainer.py` asserts bitwise-equal weights after a resume, with dropout,
    and fails if any of the three is removed. Resuming needs the same `--batch_size` and data.
  - Losses and metrics are summed **on the device** and read only when logged; target-token counts
    come from the mask before it moves. A `float(loss)` per micro-batch is a host sync that costs
    real throughput on CUDA; `optimizer_step` syncs once per update for the grad norm.
  - `save_checkpoint` / `save_final_weights` write through `atomic_torch_save` (temp file, fsync,
    `os.replace`): a rented instance can be reclaimed mid-write.
  - `pretrain.py --compile` compiles only the training forward; evaluation and saving use the raw
    module. RoPE's complex multiply is not fused by Inductor (it falls back to eager).
- **`bench_train.py`** times the real pretraining step on random ids over `--batch_sizes` ×
  `--compile` and reports tok/s, peak memory, TFLOPS and MFU (`--peak_tflops`), plus whether the
  training forward reaches flash attention on CUDA. `probes.py` is also a CLI that runs the three
  probes on any checkpoint against a `prepare` holdout.
- **`scripts/autodl/` launches the formal run on an AutoDL instance**: upload from a Mac, verify,
  CPU preflight, then bench, trial and the formal run (order in `docs/status.md` §4 step 6).
  - The instance-side scripts source `env.sh`, which sets `OMP_NUM_THREADS` from
    `/sys/fs/cgroup/cpu.max`. The instance reports the host's 176 cores, and at the no-GPU 0.5
    CPU torch's 91 threads made one update of a 2M model take 188 s instead of 1.4 s.
  - The no-GPU instance has 2GB, so `preflight.sh` trains on `val.bin` and probes shortened
    documents. Pointing it at `train.bin` gets it killed: no window length fits, since fp32 logits
    over the 32k vocabulary and the sampler's permutation trade off against each other.
- **`eval_ppl.py` wraps text pretrain-style** (`bos_token + text + eos_token`, matching
  `dataset.PretrainDataset`) before tokenizing, so PPL is computed on the same input distribution
  the model was trained on — don't strip that wrapping when touching `calculate_ppl`.
- **GRPO/RLVR specifics (`envs/`, `rollout.py`, `grpo.py`)**:
  - A rollout group is *one prompt repeated `--group_size` times*, never a batch of different
    prompts. The model applies RoPE from `start_pos` and has no left-padding offset, so mixed-length
    prompts cannot share a `generate` call. Multiple prompts per step are generated sequentially.
  - Token ids returned by `generate` are produced under `inference_mode` and **must be cloned**
    before they feed a training forward pass, or embedding backward raises on the saved index
    tensor. `rollout.generate_group` already does this.
  - Old/reference log-probs are **recomputed** with a no-grad forward, not captured during
    sampling: temperature/top-p reshape the sampling distribution but the importance ratio needs
    the policy's own distribution. With `--top_p < 1` the data is therefore slightly off-policy.
  - When micro-batching (`--micro_batch_size`), every chunk must divide by the *whole batch's*
    normalizer — that is what `grpo_policy_loss(..., normalizer=...)` is for. There are tests
    asserting chunked loss == full-batch loss for all three aggregations; don't "simplify" it away.
  - **The on-policy `seq_mean` loss value is ≈0 by construction** (ratio ≡ 1 and group advantages
    sum to zero). This is not a bug and not a sign the run is dead — look at `grad_norm`.
  - GRPO needs a policy that already emits the env's output format, otherwise every rollout scores
    0, every group has zero reward variance, and `grad_norm` is exactly 0. Cold start with SFT on
    `envs.generate_data --split sft` output first.
- **`model.generate`/`_stream_generate` supports batch>1 with per-row EOS**: each row tracks its
  own `finished` flag; once a row hits `eos_token_id` it emits that real EOS token on the hit step
  and `pad_token_id` on every step after, while other rows keep generating until they finish or
  `max_new_tokens` is reached. Callers must truncate at each row's own EOS position themselves —
  the returned tensor is not automatically trimmed per row.
