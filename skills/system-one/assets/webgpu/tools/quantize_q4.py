#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = ["onnx", "onnx-ir", "onnxruntime>=1.20"]
# ///
"""Weight-only 4-bit quantization of a laya-ts split export (encoder.onnx [+ head.onnx]) for the browser.

MatMulNBits (4 bit, block 32, symmetric) has WebGPU and WASM kernels in ONNX Runtime Web. The token
embedding (a Gather) stays fp32. Measured on laya typed-decisions: 1.69 GB -> 467 MB with --head,
110/120 decisions identical to fp32, WebGPU p50 361 ms vs 265 ms fp32. --bits 8 is kept for CPU/Node
experiments only: ORT Web has no 8-bit MatMulNBits WebGPU kernel, so it silently runs on the CPU.

Usage: uv run quantize_q4.py <fp32_dir> <out_dir> [--head] [--block-size 32] [--asym] [--bits 4|8]
"""
from __future__ import annotations

import argparse
import logging
import shutil
import time
from pathlib import Path

import onnx
from onnxruntime.quantization.matmul_nbits_quantizer import DefaultWeightOnlyQuantConfig, MatMulNBitsQuantizer

log = logging.getLogger("quantize_q4")


def quantize_encoder(src: Path, dst: Path, block_size: int = 32, symmetric: bool = True, bits: int = 4,
                     name: str = "encoder") -> None:
    """Quantize every weight MatMul of `<src>/<name>.onnx` into `<dst>/<name>.onnx` (+ .onnx.data)."""
    model = onnx.load(str(src / f"{name}.onnx"))  # pulls <name>.onnx.data in
    n_matmul = sum(1 for n in model.graph.node if n.op_type == "MatMul")
    log.info("%s: %d nodes, %d MatMul; bits=%d block_size=%d symmetric=%s",
             name, len(model.graph.node), n_matmul, bits, block_size, symmetric)
    cfg = DefaultWeightOnlyQuantConfig(block_size=block_size, is_symmetric=symmetric, bits=bits)
    q = MatMulNBitsQuantizer(model, algo_config=cfg)
    q.process()
    qm = q.model.model
    n_nbits = sum(1 for n in qm.graph.node if n.op_type == "MatMulNBits")
    # MatMuls left over multiply two activations (attention scores), there is no weight to quantize.
    log.info("%s: %d MatMul -> MatMulNBits, %d activation MatMul left", name, n_nbits,
             sum(1 for n in qm.graph.node if n.op_type == "MatMul"))
    onnx.save_model(qm, str(dst / f"{name}.onnx"), save_as_external_data=True, all_tensors_to_one_file=True,
                    location=f"{name}.onnx.data", size_threshold=1024)


def copy_rest(src: Path, dst: Path, head_quantized: bool) -> None:
    """Copy what quantization did not rewrite, so `dst` is a complete laya-ts model dir."""
    names = ["tokenizer.json", "rl_agent_config.json"]
    if not head_quantized:
        names += ["head.onnx", "head.onnx.data"]
    for name in names:
        if (src / name).exists():
            shutil.copy2(src / name, dst / name)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("src", type=Path)
    ap.add_argument("dst", type=Path)
    ap.add_argument("--head", action="store_true", help="also quantize head.onnx (costs nothing measurable)")
    ap.add_argument("--block-size", type=int, default=32)
    ap.add_argument("--asym", action="store_true", help="asymmetric (zero-point) quantization")
    ap.add_argument("--bits", type=int, default=4, choices=[4, 8])
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    if args.bits == 8:
        log.warning("8-bit MatMulNBits has no WebGPU kernel in ORT Web: fine for CPU/Node, runs on CPU in browsers")
    args.dst.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    quantize_encoder(args.src, args.dst, args.block_size, not args.asym, args.bits)
    if args.head:
        quantize_encoder(args.src, args.dst, args.block_size, not args.asym, args.bits, name="head")
    copy_rest(args.src, args.dst, head_quantized=args.head)
    for p in sorted(args.dst.iterdir()):
        log.info("  %-24s %8.1f MB", p.name, p.stat().st_size / 1e6)
    total = sum(p.stat().st_size for p in args.dst.iterdir())
    log.info("done in %.0fs, total %.1f MB", time.perf_counter() - t0, total / 1e6)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
