---
name: system-one
description: Typed-decision ("System One") models — Laya (open, Apache-2.0, runs locally/in the browser) and TypeSafe Jev (hosted API). Use when the user wants to classify/route/score/flag text or JSON with choice / score / noul questions and calibrated probabilities instead of LLM text generation; mentions Laya, laya-serve, laya-ts, Jev, TypeSafe, api.typesafe.ai, /v1/systemone, "System 1" or typed decisions; wants to swap between Jev and Laya, fine-tune or calibrate Laya on their own labels, export Laya to ONNX, or run it with WebGPU in a browser.
---

# System One: Laya + Jev typed decisions

One forward pass, no text generation: a **state** (text / JSON) plus typed **questions** in, calibrated
probabilities out. Three primitives: `choice` (pick one label), `score` (ordinal level 0..n-1,
expected value), `noul` (P(true)). Jev and Laya share the request/response format
(`POST /v1/systemone`), so code written for one runs on the other.

Scripts live in `scripts/` next to this file (run with `uv run`, JSON in / JSON out, logs on stderr).
Set `S=<this skill dir>/scripts`.

## Pick a backend

1. **Can the data leave the machine, and is no local GPU/CPU budget wanted?** → Jev
   (`$TYPESAFE_API_KEY`, $0.042 / M input tokens, output free). Also prefer Jev for **>20 options
   in one choice** question (it takes up to 255; Laya degrades past ~20).
2. **Otherwise start with Laya zero-shot** (`--model auto` routes English → `laya`, other
   languages → `laya-multilingual`) and **measure it on 50–200 labelled examples** with `compare.py`.
3. **If zero-shot accuracy is not enough → fine-tune Laya** on your labels. Base checkpoints are
   near chance on hard multi-workflow decisions (0.36 vs 0.318 random); the fine-tuned one reaches
   0.766. Treat Laya as a fast base to specialise.
4. **Browser / no server** → export to ONNX and run with laya-ts on WebGPU (`assets/webgpu/`).

Probabilities are not interchangeable between models: after any switch, re-pick thresholds on
held-out data. Details and benchmark caveats: `references/model-selection.md`.

## Commands

```bash
# Laya, local (first call downloads 0.6–0.85 GB from Hugging Face)
uv run $S/laya_predict.py --state "Billed twice, refund or we cancel" --questions $S/../assets/examples/questions.json
uv run $S/laya_predict.py --model multilingual --max-len 8192 --state-file long.txt --questions q.json

# Jev (hosted) — or any Jev-compatible server via --base-url
TYPESAFE_API_KEY=... uv run $S/jev_predict.py --state "..." --questions q.json
$S/serve.sh &   # laya-serve on 127.0.0.1:8000, then:
uv run $S/jev_predict.py --base-url http://127.0.0.1:8000 --state "..." --questions q.json

# Measure: accuracy / ECE / latency / agreement on a labelled JSONL
uv run $S/compare.py data.jsonl --backend laya:english --backend laya:multilingual [--backend jev]

# Fine-tune → calibrate → evaluate base vs tuned (→ optionally push to the Hub)
uv run $S/finetune_laya.py --smoke                       # 15 s wiring check on bundled examples
uv run $S/finetune_laya.py --data train.jsonl --base english --out runs/ft
uv run $S/laya_predict.py --model runs/ft/checkpoint --state "..." --questions q.json

# ONNX: browser (split, for laya-ts/WebGPU) or CPU (flat, optional INT8)
uv run $S/export_onnx.py --model runs/ft/checkpoint --target browser --out-dir ./webgpu/model
```

Data format for `compare.py` / `finetune_laya.py` (the `laya-evals` format), one line each:
`{"state": "...", "questions": {...}, "expected": {"qid": "label" | true | 2}}`.
Example: `assets/examples/tickets.jsonl` (40 tickets, EN/中文/ES).

## Rules of thumb

- Question sets are validated before any call (`scripts/systemone.py`); warnings flag Laya weak
  spots: >20 options, boolean-word choice labels (`yes`/`no`), noul without `criteria`.
- Always give `noul` a `criteria: {"true": ..., "false": ...}`; zero-shot noul varies by checkpoint.
- For English `score` questions pin `--model english` (multilingual has a position bias on score).
- Never hard-code or print API keys: Jev key from `$TYPESAFE_API_KEY`, Hub token from `$HF_TOKEN`.
- Jev responses are saved raw under `runs/jev/` before parsing; identical requests hit that cache.
- Model weights and ONNX exports never go into git.

## References (read the one you need)

- `references/laya-api.md` — Python SDK, CLI, request/response fields, token budgets, long docs, batching
- `references/jev-api.md` — TypeSafe Jev endpoint, auth, schema, pricing, SDKs; verified vs unverified facts
- `references/model-selection.md` — Jev vs Laya vs fine-tuned Laya, benchmarks and their caveats
- `references/finetune.md` — data prep, the RLCD recipe, calibration, evaluation, memory, pushing
- `references/serving.md` — laya-serve (Jev-compatible HTTP), swapping clients, MCP, ONNX on CPU
- `references/webgpu.md` — the browser sample, export, verification, measured MBP vs mini numbers
