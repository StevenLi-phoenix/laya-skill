# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.0] - 2026-09-29

### Added
- `system-one` Claude Code skill (`skills/system-one/SKILL.md`), installable as a plugin through
  `.claude-plugin/plugin.json` + `marketplace.json`.
- `laya_predict.py`: run Laya locally (Router auto-routing, pinned checkpoint, local/Hub fine-tune),
  single, file or JSONL batch input, `--max-len` / `--head-max-len` / `--calibration`, warm-latency repeat.
- `jev_predict.py`: stdlib client for TypeSafe Jev `POST /v1/systemone` and any Jev-compatible server.
  Key from `$TYPESAFE_API_KEY` only. Raw responses are saved before parsing and reused as a cache.
  Exponential backoff on 429/529.
- `compare.py`: score any mix of Laya checkpoints, Jev and HTTP backends on a labelled JSONL
  (accuracy per primitive, score MAE, ECE, latency, pairwise agreement).
- `finetune_laya.py`: RLCD fine-tuning on your own `laya-evals` JSONL (hard labels or soft `gold`),
  optional head-only mode, per-type temperature calibration on a held-out slice, base-vs-tuned
  evaluation, optional Hub push, and a `--smoke` wiring check.
- `export_onnx.py`: browser split export (laya-ts / WebGPU) and CPU flat export with optional INT8,
  wrapping the vendored upstream exporters.
- `serve.sh`: `laya-serve` (Jev-compatible HTTP) through uv, bound to 127.0.0.1 by default.
- `systemone.py`: shared question-schema validation (the rules both Jev and Laya accept, plus
  warnings for known Laya weak spots) and backend-agnostic metrics.
- References: Laya API, Jev API (verified vs unverified facts), model selection, fine-tuning,
  serving, WebGPU.
- WebGPU browser sample (`assets/webgpu/`) with a headless-Chrome verifier; Mac mini M4 numbers
  recorded next to the MacBook Pro M4 Pro numbers.
- Example dataset `assets/examples/tickets.jsonl` (40 labelled tickets, EN / 中文 / ES) and `questions.json`.
- pytest suite: schema, metrics, fine-tune data prep, mocked Jev HTTP, manifest checks, and
  weight-loading integration tests (`LAYA_INTEGRATION=1`).

### Changed
- The WebGPU sample moved from `webgpu/` to `skills/system-one/assets/webgpu/` so it ships with the skill.
