#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10,<3.14"
# dependencies = ["laya==0.3.22", "onnx", "onnxruntime", "onnxscript"]
# ///
"""Export a Laya checkpoint (bundled, Hub or a local fine-tune) to ONNX.

  --target browser   split encoder.onnx + head.onnx + tokenizer.json for laya-ts / the WebGPU
                     sample (upstream laya-ts/scripts/export_onnx.py; verifies torch vs ONNX to 1e-4)
  --target cpu       one flat graph for laya.ONNXAgent / `laya-evals run --onnx`
                     (upstream scripts/export_onnx.py); --quantize adds an INT8 copy (CPU only)

Examples:
  uv run export_onnx.py --model multilingual --target browser --out-dir ./webgpu/model
  uv run export_onnx.py --model runs/finetune/checkpoint --target browser --out-dir ./webgpu/model
  uv run export_onnx.py --model english --target cpu --out-dir ./onnx --quantize
fp32 output is large (~1.3 GB multilingual, ~1.7 GB english); keep it out of git.
"""
from __future__ import annotations

import argparse
import logging
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from finetune_laya import resolve_base
from systemone import setup_logging

log = logging.getLogger("export_onnx")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="multilingual", help="english | multilingual | typed-decisions | <dir> | org/repo[:subfolder]")
    ap.add_argument("--target", choices=["browser", "cpu"], default="browser")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--quantize", action="store_true", help="cpu target only: also write an INT8 copy")
    ap.add_argument("--no-verify", action="store_true", help="browser target: skip the torch-vs-ONNX check")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args(argv)
    setup_logging(args.verbose)
    if args.quantize and args.target != "cpu":
        ap.error("--quantize applies to --target cpu (ONNX Runtime web has no INT8 MatMul path here)")

    ckpt, ref = resolve_base(args.model)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    if args.target == "browser":
        cmd = [sys.executable, str(HERE / "vendor/export_onnx_split.py"), "--model-dir", ckpt, "--out-dir", str(out)]
        if args.no_verify:
            cmd.append("--no-verify")
    else:
        cmd = [sys.executable, str(HERE / "vendor/export_onnx_flat.py"), "--model", ckpt, "--output", str(out / "laya.onnx")]
        if args.quantize:
            cmd.append("--quantize")
    log.info("exporting %s -> %s (%s)", ref, out, args.target)
    t0 = time.perf_counter()
    subprocess.run(cmd, check=True)
    size = sum(p.stat().st_size for p in out.rglob("*") if p.is_file())
    log.info("done in %.0fs, %.2f GB in %s", time.perf_counter() - t0, size / 1e9, out)
    for p in sorted(out.iterdir()):
        log.info("  %-28s %8.1f MB", p.name, p.stat().st_size / 1e6)
    return 0


if __name__ == "__main__":
    sys.exit(main())
