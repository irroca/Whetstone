# ablation-v1

300M tokens per arm. bpb: holdout bits per UTF-8 byte (lower is better), comparable across vocabularies. val loss: per token, comparable only within one vocabulary.

| arm | vocab | zh share | val loss | bpb zh_web | bpb en_web | bpb code | bpb math | bpb books | bpb synthetic_textbook | bpb mix | zh→en | en→zh | add_1digit | add_2digit |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| zh00_v32k | v32k | 0.0% | 4.656 | 2.914 | 1.235 | 0.864 | 1.167 | 1.501 | 0.912 | 1.658 | 39% | 0% | 4% | 2% |
| zh15_v32k | v32k | 15.0% | 3.500 | 1.665 | 1.251 | 0.881 | 1.188 | 1.522 | 0.933 | 1.296 | 1% | 0% | 5% | 1% |
| zh30_v32k | v32k | 30.0% | 3.455 | 1.584 | 1.271 | 0.901 | 1.212 | 1.546 | 0.957 | 1.287 | 3% | 0% | 5% | 1% |
| zh45_v32k | v32k | 45.0% | 3.452 | 1.534 | 1.295 | 0.925 | 1.241 | 1.569 | 0.988 | 1.290 | 2% | 0% | 9% | 3% |
| zh30_v16k | v16k | 30.0% | 3.237 | 1.616 | 1.293 | 0.917 | 1.238 | 1.568 | 0.998 | 1.312 | 1% | 0% | 6% | 0% |
| zh30_v48k | v48k | 30.0% | 3.508 | 1.579 | 1.256 | 0.895 | 1.203 | 1.530 | 0.938 | 1.277 | 2% | 0% | 8% | 2% |
