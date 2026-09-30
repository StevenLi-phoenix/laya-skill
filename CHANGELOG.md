# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.2.0] - 2026-09-29

### Added
- 4-bit browser export: `export_onnx.py --target browser --quantize` and `assets/webgpu/export_model.sh <ckpt> --q4`
  quantize encoder + head with `MatMulNBits` (block 32, symmetric) through the unit-tested
  `assets/webgpu/tools/quantize_q4.py`. typed-decisions goes from 1.69 GB to 467 MB, 110/120 decisions equal
  to fp32, and it still runs on WebGPU (361 ms p50).
- `assets/webgpu/tools/eval_onnx.mjs` (+ `package.json`, `pnpm eval`): accuracy, agreement and probability drift of
  split exports on a labelled JSONL, in Node on the CPU.
- Model picker in the WebGPU page: typed-decisions q4 (default, hosted at
  `huggingface.co/Steven10429/laya-typed-decisions-webgpu-q4`), multilingual fp32, and local `./model/`.
- The WebGPU page is also a standalone repo: `github.com/game-design-projects/laya-webgpu`.
- Measured q4 / q8 / fp32 trade-offs (`references/webgpu.md`), including the finding that 8-bit MatMulNBits has
  no WebGPU kernel and silently runs on the CPU.
- `tests/test_quantize_q4.py`.

### Changed
- The WebGPU page collapses the model picker after loading, so the run controls fit itch's `scrolling="no"` iframe.
- `verify_iframe.py` no longer needs a local `./model` when `MODEL_URL` is set.
- `export_model.sh` takes a checkpoint name (`typed-decisions` by default) and `--q4`.

### Added
- WebGPU sample works embedded in a cross-site iframe (itch.io). `verify_iframe.py` reproduces the itch
  setup on three loopback origins (itch `allow` attribute, bare iframe, model host without CORS), and
  can also load from a remote `MODEL_URL`.
- Public weight mirror for the sample: `huggingface.co/Steven10429/laya-multilingual-webgpu` (fp32 split
  export of `convaiinnovations/laya` multilingual). Live demo at `stevenli-phoenix-work.itch.io/laya-webgpu`.
- `?model=<url>` query parameter to point the sample at any CORS-enabled export.

### Changed
- The sample picks `./model/` on localhost and the Hugging Face mirror elsewhere.
- Desktop layout fits one viewport with per-panel scrolling, so the controls stay reachable inside
  itch's `scrolling="no"` iframe. The model chip shows the repo name, and the footer names the weight source.
- Install docs default to repo-level installs: `--scope project` for the plugin commands, or
  copying the skill into `<repo>/.claude/skills/`. No global (`~/.claude`) install is suggested any more.

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
