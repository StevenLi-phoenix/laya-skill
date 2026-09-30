# Jev vs Laya vs fine-tuned Laya

## Decision guide

| situation | pick | why |
|---|---|---|
| Data must stay on-prem / offline / in the browser | Laya | open weights, Apache-2.0, runs on CPU / MPS / CUDA / WebGPU |
| One choice question with >20 options (e.g. 77 intents) and no time to tune | Jev | takes up to 255 options; Laya's shared option budget gives each option ~3–4 tokens past ~50 (Banking77 0.870 Jev vs 0.425 Laya) |
| Latency-critical loop (agent step, guardrail, router) | Laya | ~30–80 ms local vs 236–276 ms p50 measured for Jev by third parties |
| Hard domain decisions, you have ≥ a few hundred labels | fine-tuned Laya | base checkpoints near chance zero-shot on multi-workflow decisions, 0.766 after fine-tuning |
| Quick prototype, English, no labels, no infra | Jev | nothing to host; $0.042 / M input tokens |
| Non-English | Laya `multilingual` (Router does this) or Jev | Jev says CJK works "not equally well"; Laya-multilingual clears 3x random in 48/51 languages |
| Long documents (> 1k tokens) | Laya `multilingual` with `max_len=8192`, or Jev (32k state) | English checkpoint is 512 tokens |

Because the wire format is shared, you rarely have to commit: write against `/v1/systemone`, run
`compare.py` with both backends on your labels, pick per use case.

## Published numbers and why to distrust them

From the Laya README (0.3.22). Laya rows are measured by the Laya author; **Jev rows are
third-party/published, never measured by the Laya project**, and are not on TypeSafe's docs.

| | Jev 1.13.0 | Laya (routed) |
|---|---|---|
| typed-decisions (2,000 decisions) | 0.727 | 0.766 (the `typed-decisions` fine-tune) |
| AG News (4 labels) | 0.910 | 0.950 |
| DAIR Emotion (6 labels) | 0.480 | 0.595 |
| Banking77 | 0.870 (72 labels) | 0.425 (77 labels) |
| ECE | 0.246 (0.144 in another table) | 0.081 after temperature fit |
| p50 latency, 1 question | 236–276 ms (third party) | 32.8 ms (T4) |

typed-decisions on all three Laya checkpoints: `laya-typed-decisions` 0.766, `laya` 0.362,
`laya-multilingual` 0.352. The random baseline is 0.318 and the majority-class baseline is 0.461, so **the base
checkpoints are below majority class** on this benchmark.

Caveats:
- Vendor benchmarks are self-reported, on different samples, prompts and label counts (72 vs 77).
  Nothing here is an apples-to-apples comparison. Measure on your data.
- The 0.766 checkpoint was fine-tuned on that benchmark's own training split, so it is an in-domain
  number, not a general zero-shot one.
- Laya's ECE figure is after fitting temperatures on held-out data. Raw ECE is worse than Jev's
  (0.213 vs 0.144 on typed-decisions).
- Jev has better soft accuracy (0.580 vs 0.471): its distributions match the teacher more closely.
- **Probabilities are not interchangeable** between Jev and Laya, or between two Laya checkpoints.
  Observed here: the same English ticket gave `churn_risk` noul 0.879 on `english` (Python Router)
  and 0.045 on `multilingual` (browser sample). After any switch, re-pick every threshold on held-out data.

## A 30-minute evaluation recipe

1. Label 50–200 real examples in the `laya-evals` JSONL format (`assets/examples/tickets.jsonl` is a template).
2. `uv run scripts/compare.py data.jsonl --backend laya:english --backend laya:multilingual [--backend jev] --json runs/cmp.json`
3. Look at accuracy per primitive, ECE, p50 latency and pairwise agreement. Low agreement between
   backends points to the questions that need better criteria wording or fine-tuning.
4. If Laya is not good enough, run `finetune_laya.py` (see `finetune.md`) and add
   `--backend laya:runs/ft/checkpoint` to the same comparison.

Measured on this repo's 40-ticket example set, Mac mini M4 (see README for the full table).
