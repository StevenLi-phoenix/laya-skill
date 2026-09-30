# Laya in the browser (WebGPU)

`assets/webgpu/` is a self-contained static page: laya-ts (vendored ES modules, built from
upstream `laya-ts/` at `6d942c9`) + `onnxruntime-web@1.30.0` from jsDelivr (import map), a split ONNX export in `model/`,
presets (EN / 中文 / हिन्दी / ES / phishing), a JSON question editor, probability bars, and cold /
warm p50 / p90 timing with an execution-provider chip. Nothing leaves the browser.

## Run it

The skill directory may be read-only or shared (plugin cache), so copy the sample first:

```bash
cp -R <skill>/assets/webgpu ./laya-webgpu && cd laya-webgpu
./export_model.sh typed-decisions --q4  # standalone: clones laya @6d942c9 into ./.laya-src, exports + 4-bit → ./model (467 MB)
./export_model.sh multilingual          # fp32, 100+ languages (1.3 GB)
# or with the skill's exporter, from any checkpoint including a fine-tune:
uv run <skill>/scripts/export_onnx.py --model typed-decisions --target browser --out-dir ./model --quantize
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

## Smaller downloads: 4-bit (q4)

`export_onnx.py --target browser --quantize` (or `export_model.sh <ckpt> --q4`) runs `tools/quantize_q4.py`:
ONNX Runtime's `MatMulNBitsQuantizer`, 4 bit, block 32, symmetric, over every weight MatMul of the encoder
(112 for ModernBERT-large) and the head (8). The token embedding is a Gather and stays fp32. The result is
deterministic: a fresh export is byte-identical to the published
[Steven10429/laya-typed-decisions-webgpu-q4](https://huggingface.co/Steven10429/laya-typed-decisions-webgpu-q4).

Measured on typed-decisions. Fidelity comes from `tools/eval_onnx.mjs` (Node, CPU) on the 40 example tickets × 3
questions. Speed is Chrome on an M4 Pro, 3 questions / 163 tokens:

| variant | size | WebGPU p50 | WASM p50 | agrees with fp32 | max drift | dept / urgency / churn acc |
|---|---|---|---|---|---|---|
| fp32 | 1.69 GB | 265 ms | 2443 ms | — | — | 0.875 / 0.425 / 0.700 |
| **q4 enc + head** | **467 MB** | **361 ms** | ~2.8 s | 110 / 120 | 0.22 | 0.875 / 0.450 / 0.625 |
| q4 enc only | 534 MB | 350 ms | 2777 ms | 110 / 120 | 0.22 | same as above |
| q4 asym b32 / sym b16 / asym b16 | 539–588 MB | — | — | 110–114 / 120 | 0.14–0.25 | within ±2 answers |
| q8 | 705 MB | **2800 ms** | 2792 ms | 119 / 120 | 0.013 | 0.900 / 0.425 / 0.700 |

What this means:

- **q8 is a trap in the browser.** ORT Web has no 8-bit MatMulNBits WebGPU kernel, so those nodes run on the CPU.
  The session still reports WebGPU because the rest of the graph is on the GPU. Only the timing gives it away,
  which is why the `--force-wasm` baseline matters. Use q8 on CPU / Node only.
- q4 is slower than fp32 on the GPU (dequantization), but the first download is 3.6× smaller: 18–20 s from Hugging Face
  (measured), vs 33 s for the 1.3 GB multilingual export. For a web page, the first load usually decides.
- q4 changes about 8 % of decisions relative to fp32, and 40 tickets is a small set. Re-pick thresholds on your own
  held-out data, and compare with `pnpm install && pnpm eval` (in `assets/webgpu/`) before shipping.
- Quantizing the multilingual checkpoint helps much less. Its 256k-token embedding is ~786 MB of fp32 Gather
  that MatMulNBits does not touch.
- `typed-decisions` is English-only, and it is bigger (ModernBERT-large) than multilingual (mmBERT-base).
  On the English billing preset it gets churn right (0.765), where multilingual says 0.045.

## Hosted demo and embedding (iframe / itch.io)

Live demo: **https://stevenli-phoenix-work.itch.io/laya-webgpu** (itch.io HTML embed, 1100×820); source:
**https://github.com/game-design-projects/laya-webgpu**. The page has a model picker. The default is
[typed-decisions q4](https://huggingface.co/Steven10429/laya-typed-decisions-webgpu-q4) (467 MB); the alternative is
[multilingual fp32](https://huggingface.co/Steven10429/laya-multilingual-webgpu) (1.3 GB). Both are public and
Apache-2.0 with attribution. On localhost a third `./model/` option appears.

The page chooses its model URL in this order: `?model=<url>`, then `./model/` when served from
`localhost` / `127.0.0.1`, then the q4 Hugging Face mirror everywhere else. After loading, the picker collapses so
the run controls fit the iframe. To point it at your own
export (for example a fine-tune pushed to the Hub), pass `?model=https://huggingface.co/<you>/<repo>/resolve/main/`.

Things that matter when embedding:

- **The weights cannot live on itch.** itch caps HTML uploads at 200 MB per file and 500 MB in total,
  and `encoder.onnx.data` alone is 1.23 GB. The code bundle is 16 files / 0.2 MiB. The model has to come
  from a CORS-enabled host. Hugging Face returns `access-control-allow-origin` for the itch origin, both
  on the `resolve/` redirect and on its CDN (`*`).
- **CORS is mandatory.** Without it, loading fails with a misleading
  `Incompatible model: ... does not contain 'rl_agent_config.json'`.
- **WebGPU needs no permission.** It works in a cross-site iframe both with itch's exact
  `allow="autoplay; fullscreen *; ...; cross-origin-isolated; web-share"` attribute and with no `allow` at all.
  CacheStorage also works (partitioned per top-level site).
- **Fit one viewport.** itch embeds with `scrolling="no"`, so on desktop the page is laid out to exactly
  `100vh` and each panel scrolls on its own. Otherwise the Decide button ends up below the fold where
  it cannot be reached.
- Don't autostart. The first load downloads ~1.3 GB (about 33 s over a fast connection), so let
  the user click "加载模型".

`verify_iframe.py` reproduces the itch setup on three loopback origins: parent on `127.0.0.1`, game
on `localhost` (cross-site, so an out-of-process iframe), and model on a third port with or without CORS:

```bash
uv run verify_iframe.py                       # scenarios: itch (itch allow attr + CORS), noallow, nocors (expected to fail)
MODEL_URL=https://huggingface.co/Steven10429/laya-multilingual-webgpu/resolve/main/ uv run verify_iframe.py --scenario itch
```

| scenario (MBP M4 Pro, Chrome headless) | WebGPU | load | p50 / p90 |
|---|---|---|---|
| itch iframe, model from local CORS origin | ✓ | 5.4 s | 149 / 182 ms |
| bare iframe (no `allow`) | ✓ | 6.1 s | 149 / 169 ms |
| model origin without CORS | ✓ | fails (as expected) | — |
| itch iframe, model from Hugging Face | ✓ | 34 s (cold download) | 147 / 165 ms |
| **live itch page**, logged-out fresh profile | ✓ | 32.7 s | 146 / 171 ms |

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
