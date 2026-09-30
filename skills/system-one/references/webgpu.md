# Laya in the browser (WebGPU)

`assets/webgpu/` is a self-contained static page: laya-ts (vendored ES modules, built from
upstream `laya-ts/` at `6d942c9`) + `onnxruntime-web@1.30.0` from jsDelivr (import map), a split ONNX export in `model/`,
presets (EN / 中文 / हिन्दी / ES / phishing), a JSON question editor, probability bars, and cold /
warm p50 / p90 timing with an execution-provider chip. Nothing leaves the browser.

## Run it

The skill directory may be read-only or shared (plugin cache), so copy the sample first:

```bash
cp -R <skill>/assets/webgpu ./laya-webgpu && cd laya-webgpu
./export_model.sh                       # standalone: clones laya @6d942c9 into ./.laya-src, exports multilingual → ./model
# or with the skill's exporter, from any checkpoint including a fine-tune:
uv run <skill>/scripts/export_onnx.py --model multilingual --target browser --out-dir ./model
uv run <skill>/scripts/export_onnx.py --model runs/ft/checkpoint --target browser --out-dir ./model
python3 -m http.server 8765    # then open http://127.0.0.1:8765/
```

`model/` holds `encoder.onnx` (+ 1.23 GB `encoder.onnx.data`, fp32), `head.onnx` (+60 MB),
`tokenizer.json` (34 MB) and `rl_agent_config.json`; ~1.3 GB for `multilingual`. Keep it out of git
(`.gitignore` already does). The exporter checks torch against ONNX to within 1e-4 at two sequence lengths
before writing.

## Verify it really uses the GPU

```bash
uv run verify_webgpu.py                 # headless Chrome, own temp profile, 20 warm decisions, JSON report + screenshot
uv run verify_webgpu.py --force-wasm    # hides navigator.gpu → WASM baseline
uv run verify_webgpu.py --headed        # watch it
```

It launches its own Chrome (`--remote-debugging-port` on a free port, temp `--user-data-dir`), so
it never touches a Chrome you already have open. Exit code 0 only if the model loads and the preset
answers come back.

Why the `--force-wasm` run matters: laya-ts runs **only the encoder** on WebGPU; the small decision
head always runs on WASM. If the WebGPU session cannot be created, it **silently falls back to
WASM**. The page's EP chip reads the fallback flag, and a ~4–5× slower WASM run is the independent proof.

## Measured

Multilingual checkpoint, fp32 split export, Chrome headless, served from localhost, preset 0 (EN
billing ticket, 3 questions, 163 input tokens), 20 warm runs:

| machine | GPU (WebGPU adapter) | load | WebGPU p50 / p90 | WASM p50 / p90 | speed-up |
|---|---|---|---|---|---|
| MacBook Pro M4 Pro | apple metal-3 | 7.3 s | **148 / 156 ms** | 808 / 846 ms | 5.5× |
| Mac mini M4 16 GB | apple metal-3 | 3.9 s | **183 / 190 ms** (re-run: 165 / 187) | 810 / 819 ms | 4.4× |

Answers were identical on both providers and both machines: `department=billing (1.00)`,
`urgency=1.79 (blocking 80.7%)`, `churn_risk noul=0.045`. Python on MPS (native torch) is
faster still: 46–120 ms warm.

Note the noul: the Python Router sends this English text to the **English** checkpoint and gets
0.879. The browser sample loads **multilingual** and gets 0.045. Checkpoints disagree on zero-shot
noul, which is one more reason to give `noul` explicit criteria and to calibrate on your own data.

## Other browser / edge options (verified to exist on 2026-09-29)

- [`@r4ai/laya-web`](https://www.npmjs.com/package/@r4ai/laya-web) 0.2.0: ORT-web, WebGPU → WASM,
  needs its own export format.
- [`receptron/laya`](https://github.com/receptron/laya): Node ONNX runtime.
- Prebuilt ONNX on the Hub: [`onnx-community/laya-ONNX`](https://huggingface.co/onnx-community/laya-ONNX)
  (fp16, ~842 MB) and [`m1rhan/laya-typed-decisions-ONNX`](https://huggingface.co/m1rhan/laya-typed-decisions-ONNX)
  (q4, 448 MB). These are community exports; their graph layout is not necessarily the laya-ts split layout.
- Apple native: [`mizorewww/laya-mlx`](https://github.com/mizorewww/laya-mlx),
  [`mizorewww/laya-coreml`](https://github.com/mizorewww/laya-coreml),
  [`tc3oliver/laya-apple`](https://github.com/tc3oliver/laya-apple).
- `laya-ts` and `laya-client` are **not on npm** (404 on 2026-09-29): build from the upstream repo
  (`cd laya-ts && pnpm install && pnpm add -D @types/node && pnpm run build`; without `@types/node`
  the build fails on `node:fs/promises`).

## Troubleshooting

- `EP wasm` chip on a machine with a GPU: check `chrome://gpu`; headless Chrome needs a real GPU
  process (no `--disable-gpu`). The sample logs the adapter (`apple metal-3`) to the console.
- Buffer limits: the adapter's `maxBuffer` is logged (4 GB on the Mac mini M4), and the fp32
  multilingual encoder fits. Adapters with smaller limits are untested here.
- Load time is dominated by reading the 1.3 GB model. The "清空缓存" button deletes the page's
  Cache Storage entries (`caches.delete`).
