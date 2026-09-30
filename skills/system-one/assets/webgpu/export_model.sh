#!/usr/bin/env bash
# Export a Laya checkpoint to split ONNX (encoder.onnx + head.onnx) for this page, optionally 4-bit.
# Weights come from Hugging Face (convaiinnovations/laya) and land in ./model (gitignored).
#
# Usage: ./export_model.sh [typed-decisions|multilingual|english] [--q4]
#   ./export_model.sh typed-decisions --q4   # what the hosted default is: 467 MB, English
#   ./export_model.sh multilingual           # fp32, 1.3 GB, 100+ languages
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
CKPT="${1:-typed-decisions}"
Q4="${2:-}"
LAYA_REV="${LAYA_REV:-6d942c92081fbc139e736bbd9ac0023223c29b7f}"
WORK="${LAYA_WORK:-$HERE/.laya-src}"
log() { printf '[export_model] %s\n' "$*" >&2; }

case "$CKPT" in
  typed-decisions|multilingual) SUB="$CKPT" ;;
  english) SUB="" ;;
  *) log "unknown checkpoint '$CKPT' (typed-decisions | multilingual | english)"; exit 2 ;;
esac

if [ ! -d "$WORK/.git" ]; then
  log "cloning NandhaKishorM/laya -> $WORK"
  git clone -q https://github.com/NandhaKishorM/laya.git "$WORK"
fi
git -C "$WORK" fetch -q origin "$LAYA_REV" 2>/dev/null || true
git -C "$WORK" checkout -q "$LAYA_REV"

if [ ! -x "$WORK/.venv/bin/python" ]; then
  log "creating venv (uv, python 3.12)"
  uv venv -q --python 3.12 "$WORK/.venv"
fi
VPY="$WORK/.venv/bin/python"
uv pip install -q --python "$VPY" -e "$WORK" onnx onnxruntime onnxscript

FP32="$HERE/model"
[ "$Q4" = "--q4" ] && FP32="$HERE/.model-fp32"
ARGS=(--repo convaiinnovations/laya --out-dir "$FP32")
[ -n "$SUB" ] && ARGS+=(--subfolder "$SUB")
log "exporting $CKPT (fp32, verified vs torch) -> $FP32"
rm -rf "$FP32"
"$VPY" "$WORK/laya-ts/scripts/export_onnx.py" "${ARGS[@]}"

if [ "$Q4" = "--q4" ]; then
  log "quantizing encoder + head to 4 bit (MatMulNBits, block 32, symmetric) -> $HERE/model"
  rm -rf "$HERE/model"
  "$VPY" "$HERE/tools/quantize_q4.py" "$FP32" "$HERE/model" --head
  rm -rf "$FP32"
fi
ls -la "$HERE/model"
log "done. serve: python3 -m http.server 8765 -d \"$HERE\"  then open http://127.0.0.1:8765/"
