# Serving Laya (Jev-compatible) and swapping backends

## laya-serve: the Jev wire protocol, self-hosted

`laya.serve` (extra `laya[serve]`, adds fastapi + uvicorn) exposes the Router on the same
`POST /v1/systemone` as TypeSafe's hosted Jev, plus `POST /v1/systemone/batch` (≤ 64 states,
shared forward passes, `results` + `total_usage`) and `GET /health`.

```bash
scripts/serve.sh                                         # 127.0.0.1:8000 via uv, laya[serve]==0.3.22
LAYA_PORT=8011 LAYA_MODELS=english scripts/serve.sh
LAYA_PRELOAD=1 LAYA_DEVICE=mps scripts/serve.sh
LAYA_API_KEY=... scripts/serve.sh                        # clients must then send Authorization: Bearer <key>
```

`serve.sh` defaults `LAYA_HOST` to 127.0.0.1 (upstream binds 0.0.0.0). Set it explicitly to expose
the server, and add `LAYA_API_KEY` when you do.

Environment (upstream `laya/serve.py`): `LAYA_HOST`, `LAYA_PORT`, `LAYA_DEVICE`, `LAYA_PRELOAD`,
`LAYA_MODELS` (comma list), `LAYA_THREADS` (CPU intra-op threads ≤ physical cores),
`LAYA_AUTO_TASK`, `LAYA_MAX_LOADED` (default 2 resident checkpoints), `LAYA_API_KEY`,
`LAYA_ROOT_PATH` (reverse-proxy prefix), `LAYA_MAX_TOKEN_BUDGET`.

Request body: Jev's body, plus optional `model` (a Laya name pins the checkpoint; any other value,
e.g. `jev-latest`, lets the router choose), `task`, `lang`, `lang_guess`, `min_confidence`,
`max_len`, `head_max_len`. Hooks cannot be sent over HTTP (422).

## Swapping a Jev client to Laya

Only the base URL changes:

```bash
uv run scripts/jev_predict.py --base-url http://127.0.0.1:8000 --state "..." --questions q.json
JEV_BASE_URL=http://127.0.0.1:8000 uv run scripts/jev_predict.py ...
```

With the official SDKs, point their base URL option at the server (check the SDK docs for the
option name. It is not verified here).

Verified on the Mac mini (2026-09-29): `jev_predict.py → laya-serve (english)` and in-process
`laya:english` gave identical answers on 10 example tickets (agreement 1.0), ~90 ms p50 each.

Three differences to handle when porting (upstream README):
- **Options**: Laya shares an option token budget (`head_max_len`); laya-serve rejects > 100 choice
  options with 413 (Jev allows 255). Past ~20 options accuracy drops, so shortlist or split.
- **Score levels**: every level needs a description; `null` → 422.
- **`confidence`** is a different formula. Gate on `answer_confidence`, and re-fit thresholds.

## Other surfaces

- **Batch CLI**: `laya --batch file.txt --predict --json --batch-size 8 --sort-by-length` (1.4× on
  mixed-length inputs, measured upstream).
- **MCP** (`laya[mcp]`): `laya-mcp-server` over stdio, tools `laya_predict`, `laya_predict_batch`,
  `laya_route`, `laya_route_batch`, `laya_decide`, `laya_shortlist`, `laya_preset`, `laya_status`.
  Claude Code (command shape, not run here): `claude mcp add laya -- uvx --from "laya[mcp]==0.3.22" laya-mcp-server`.
- **ONNX on CPU**: `uv run scripts/export_onnx.py --model english --target cpu --out-dir onnx --quantize`
  → `laya.ONNXAgent` or `laya-evals run data.jsonl --onnx onnx/laya.int8.onnx`. Upstream measured fp32
  ONNX identical to torch and INT8 within 0.006 ECE on its fixture.
- **TypeScript over HTTP**: upstream `sdk/typescript` (`laya-client`, not on npm, so build from source).
- **Docker / Nix**: upstream `Dockerfile`, `compose*.yaml`, `flake.nix` (`nix run .#laya-serve`).
- **Other runtimes**: [`receptron/laya`](https://github.com/receptron/laya) (Node ONNX),
  [`mizorewww/laya-mlx`](https://github.com/mizorewww/laya-mlx) (MLX),
  [`bladedevoff/stuntd`](https://github.com/bladedevoff/stuntd) (local Jev-API proxy that trains heads from your rows).
