"""Figures for docs/blog/ from the committed records:  python3 docs/figures/make_blog_figures.py"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
REC = ROOT / "records"
OUT = Path(__file__).parent
RUNAWAY = {"grpo_s3", "dr_grpo_s1", "dr_grpo_s3", "dapo_lite_s2"}


def train(run: Path):
    rows = [json.loads(x) for x in (run / "grpo_metrics.jsonl").read_text().splitlines() if x]
    return [r for r in rows if "lr" in r]


def smooth(v, k=10):
    return [sum(v[max(0, i - k + 1): i + 1]) / len(v[max(0, i - k + 1): i + 1]) for i in range(len(v))]


def eos_figure() -> None:
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.6), sharey=True)
    for run in sorted((REC / "posttrain_v4").glob("*_s[123]")):
        t = train(run)
        bad = run.name in RUNAWAY
        ax[0].plot([r["step"] for r in t], smooth([r["completion_len"] for r in t]),
                   color="#c0392b" if bad else "#95a5a6", lw=1.6 if bad else 1.0, label=run.name if bad else None)
    for run in sorted((REC / "posttrain_v5").glob("*_s[123]")):
        if run.name in RUNAWAY:
            t = train(run)
            ax[1].plot([r["step"] for r in t], smooth([r["completion_len"] for r in t]), color="#27ae60", lw=1.6, label=run.name)
    ax[0].set_title("9 runs, no fix: 4 stop writing EOS")
    ax[1].set_title("the same 4 cells with --truncated_reward 0")
    for a in ax:
        a.axhline(120, color="k", ls=":", lw=0.8)
        a.set_xlabel("GRPO step")
        a.legend(fontsize=7, loc="center right")
    ax[0].set_ylabel("completion length (tokens, 10-step mean)\n120 = max_new_tokens")
    fig.tight_layout()
    fig.savefig(OUT / "blog_eos_runaway.png", dpi=150)


def grid_figure() -> None:
    def tests(root: Path, name: str, seeds):
        out = []
        for s in seeds:
            for base in ([REC / "posttrain_v5", REC / "posttrain_v4"] if "600" in root.name and not name.endswith("_cur") else [root]):
                p = base / f"{name}_s{s}" / "test.json"
                if p.exists():
                    out.append(json.loads(p.read_text())["accuracy"])
                    break
        return out

    names = [("grpo", "GRPO"), ("dr_grpo", "Dr. GRPO"), ("dapo_lite", "DAPO-lite"), ("grpo_cur", "GRPO\n+curriculum"),
             ("dapo_lite_cur", "DAPO-lite\n+curriculum")]
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.6), sharey=False)
    for a, (root, seeds, title) in zip(ax, ((REC / "curriculum_v1", range(1, 6), "300 steps, 5 seeds"),
                                          (REC / "curriculum_v1_600", range(1, 4), "600 steps, 3 seeds"))):
        for i, (n, label) in enumerate(names):
            src = root if n.endswith("_cur") else root
            v = tests(src, n, seeds)
            a.scatter([i] * len(v), v, color="#2980b9" if not n.endswith("_cur") else "#e67e22", zorder=3)
            if v:
                a.hlines(sum(v) / len(v), i - 0.3, i + 0.3, color="k", lw=1.5)
        a.set_xticks(range(len(names)))
        a.set_xticklabels([l for _, l in names], fontsize=8)
        a.set_title(title + " (test accuracy, 3-digit +/-)")
        a.grid(axis="y", alpha=0.3)
    ax[0].set_ylabel("test accuracy (one dot per seed)")
    fig.tight_layout()
    fig.savefig(OUT / "blog_grid.png", dpi=150)


if __name__ == "__main__":
    eos_figure()
    grid_figure()
