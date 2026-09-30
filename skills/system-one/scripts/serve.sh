#!/usr/bin/env bash
# Run laya-serve: a Jev-compatible POST /v1/systemone (+ /v1/systemone/batch, /health) server.
# Point any Jev client at http://$LAYA_HOST:$LAYA_PORT instead of https://api.typesafe.ai.
#
# Usage: serve.sh                      # 127.0.0.1:8000, english + multilingual, lazy load
#        LAYA_PRELOAD=1 LAYA_MODELS=english,multilingual serve.sh
#        LAYA_API_KEY=... serve.sh     # then clients must send Authorization: Bearer <key>
# Binds to 127.0.0.1 by default (upstream default is 0.0.0.0); set LAYA_HOST to expose it.
set -euo pipefail
export LAYA_HOST="${LAYA_HOST:-127.0.0.1}"
export LAYA_PORT="${LAYA_PORT:-8000}"
LAYA_VERSION="${LAYA_VERSION:-0.3.22}"
log() { printf '[serve] %s\n' "$*" >&2; }
command -v uv >/dev/null || { log "uv not found: https://docs.astral.sh/uv/"; exit 1; }
log "laya[serve]==$LAYA_VERSION on http://$LAYA_HOST:$LAYA_PORT (models: ${LAYA_MODELS:-auto}, device: ${LAYA_DEVICE:-auto})"
[ -n "${LAYA_API_KEY:-}" ] && log "bearer auth enabled (LAYA_API_KEY is set)"
exec uv run --quiet --no-project --python 3.12 --with "laya[serve]==$LAYA_VERSION" laya-serve "$@"
