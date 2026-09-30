#!/usr/bin/env bash
# Export a Laya checkpoint to split ONNX (encoder.onnx + head.onnx) for the browser sample.
# Weights are downloaded from Hugging Face and written to ./model (gitignored, ~1.3 GB fp32).
# Usage: ./export_model.sh [subfolder]   (default: multilingual; "" = English laya)
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
SUB="${1-multilingual}"
LAYA_REV="${LAYA_REV:-6d942c92081fbc139e736bbd9ac0023223c29b7f}"
WORK="${LAYA_WORK:-$HERE/.laya-src}"
log() { printf '[export_model] %s\n' "$*" >&2; }

if [ ! -d "$WORK/.git" ]; then
  log "cloning NandhaKishorM/laya@$LAYA_REV -> $WORK"
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

ARGS=(--repo convaiinnovations/laya --out-dir "$HERE/model")
[ -n "$SUB" ] && ARGS+=(--subfolder "$SUB")
log "exporting ${SUB:-english} -> $HERE/model"
rm -rf "$HERE/model"
"$VPY" "$WORK/laya-ts/scripts/export_onnx.py" "${ARGS[@]}"
ls -la "$HERE/model"
log "done. serve with: python3 -m http.server 8765 -d \"$HERE\"  then open http://127.0.0.1:8765/"
