# Fine-tuning Laya

`scripts/finetune_laya.py` is a single-process port of upstream's RLCD loop
(`notebooks/laya_finetune_typed_decisions_mps.py`, the local sibling of the Kaggle notebook
`laya_finetune_typed_decisions_2xT4_kaggle.ipynb`). It takes your JSONL instead of the
`LocalLLaMA/typed-decisions` dataset and adds a held-out base-vs-tuned evaluation.

## 1. Data

One JSON object per line (the `laya-evals` format), optionally with soft targets:

```json
{"state": "Billed twice, refund or we leave", "questions": {...},
 "expected": {"department": "billing", "urgency": 2, "churn_risk": true},
 "gold": {"department": {"probabilities": {"billing": 0.9, "account": 0.1}}}}
```

- `expected`: label for `choice`, bool for `noul`, level index (0..n-1, may be fractional) for `score`.
  It becomes a one-hot target smoothed by `--label-smoothing` (0.05). A fractional score splits its mass
  between the two neighbouring levels.
- `gold` (optional, per question) wins over `expected`. Use it when a teacher (an LLM, Jev, or several
  annotators) gives probabilities, because RLCD learns best from distributions. Keys follow the option order:
  choice labels, `"false"/"true"` for noul, `"0".."n-1"` for score.
- Rows whose label is not an option, or whose options overflow the head budget, are skipped and counted in the log.
- Questions are validated with the same rules as the predict scripts.
- How much data: the upstream benchmark used 1,200 cases × 5 questions. A few hundred labelled
  decisions per question type is a sensible start. The bundled 40-row set is for wiring checks only.

## 2. Train

```bash
uv run scripts/finetune_laya.py --smoke                                      # ~15 s, head-only, 8 updates
uv run scripts/finetune_laya.py --data train.jsonl --base english --out runs/ft            # full fine-tune
uv run scripts/finetune_laya.py --data train.jsonl --eval-data test.jsonl --freeze-encoder  # head only
uv run scripts/finetune_laya.py --data d.jsonl --base multilingual --max-len 1024 --epochs 4
```

The recipe (upstream defaults kept):
- Loss = GRPO-style policy gradient over 4 noisy logit samples, with exploration σ annealed 0.4 → 0.1
  and rewards from proper scoring rules (spherical 0.75 + ranked-probability 1.0), plus a full-weight
  soft cross-entropy to the target distribution.
- AdamW, encoder lr 2.5e-5, head lr 1e-4, cosine schedule, grad-norm clip 1.0, weight decay 0.01.
- Gradient checkpointing on encoder + head (off with `--no-checkpointing`).
- `--micro-batch 2 --grad-accum 16` → effective 32 sequences (upstream Kaggle: 64 across 2 GPUs).
- `--freeze-encoder` trains only the 2-layer decision head: several times faster, a fraction of the
  memory, and good when labels are few. Full fine-tuning moves accuracy most when data is plentiful.
- `--base` accepts `english | multilingual | typed-decisions`, a local dir (continue from your own
  fine-tune) or `org/repo[:subfolder]`.

Measured on a Mac mini M4 16 GB (MPS), 40-row example set (32 train rows = 87 decisions):

| run | base | mode | updates | train time | peak memory footprint |
|---|---|---|---|---|---|
| `--smoke` | multilingual | head only, max_len 256 | 8 | 0.9 s (14 s end to end incl. eval) | not measured |
| full | english | encoder + head, max_len 512 | 33 (3 epochs) | 70 s | 9.8 GB (`/usr/bin/time -l`, 1-epoch run) |

Upstream reference: ~4–5 h on Kaggle 2×T4 for 4 epochs over ~30k questions. The browser-agent
worked example (`docs/finetune_browser_agent.md`) used one 16 GB GPU.

## 3. Calibrate (automatic)

Before training, up to 10% of the decisions (max `--calib-max` 400, and only when there are
≥ 20 decisions) are held out as a **calibration slice**. After training, one temperature per type
(choice, score, noul) is fitted on it with `laya.calibrate.fit_temperature_map` (LBFGS on
log-temperature). A type with fewer than 10 calibration decisions keeps the base checkpoint's
temperature. Any inherited `temperature_by_options` is removed from the config: those bucket values
take precedence at inference and would silently mask the new fit (upstream docs/finetune.md).

Temperature scaling never changes accuracy, only confidence. For a proper per-bucket fit later,
use `records_from_labeled` + `agent.fit_temperatures(..., compute_ece=True)` on a separate labelled
set (see `laya-api.md`).

## 4. Evaluate (automatic)

The held-out split (`--eval-frac 0.2`, or `--eval-data`) is answered by the base and the tuned
checkpoint; `runs/.../report.json` has accuracy per primitive, score MAE, ECE and latency for both.
Treat small eval sets as noise: 8 rows × 3 questions moves in steps of 4 %. For CI gating, use
`laya-evals run test.jsonl --model runs/ft/checkpoint --min-accuracy ... --max-ece ...`.

## 5. Output and reuse

`<out>/checkpoint/` holds `model.safetensors` (fp16), `encoder/`, `tokenizer/`,
`rl_agent_config.json` (with `temperature`, `fine_tuned: true`) and `finetune_meta.json` (base,
data hash, args, temperatures, timings). It loads like any checkpoint:

```bash
uv run scripts/laya_predict.py --model runs/ft/checkpoint --state "..." --questions q.json
uv run scripts/compare.py test.jsonl --backend laya:english --backend laya:runs/ft/checkpoint
uv run scripts/export_onnx.py --model runs/ft/checkpoint --target browser --out-dir webgpu/model
LAYA_MODELS=... laya-serve   # or python: laya.load("runs/ft/checkpoint")
```

Push: `HF_TOKEN=<write token> uv run scripts/finetune_laya.py ... --push-to-hub you/laya-mydomain`
(private repo unless `--public`). The token is read from the environment only.

## What to watch

- The loop is only as good as the targets. With hard labels you teach argmax; with `gold`
  distributions you also teach confidence.
- Keep the slices you care about (language, workflow) in the held-out data.
- A copied `rl_agent_config.json` that still has `temperature_by_options` silently undoes the calibration.
- Known limits carry over: >20 options, negated choices (#377) and first-level bias on multilingual
  score questions (#131). Fine-tuning helps, but check each one on your own data.
