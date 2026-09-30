#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = []
# ///
"""Call TypeSafe Jev (or any Jev-compatible server such as `laya-serve`) over POST /v1/systemone.

The API key is read from an environment variable only (default TYPESAFE_API_KEY) and is never
logged. Every raw response is written to --raw-dir *before* it is parsed; a later identical
request is answered from that file instead of calling (and paying for) the API again.

Examples:
  uv run jev_predict.py --state "billed twice, refund" --questions questions.json
  uv run jev_predict.py --request request.json --no-cache
  uv run jev_predict.py --base-url http://127.0.0.1:8000 --batch rows.jsonl --questions q.json   # laya-serve
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
from systemone import SchemaError, add_input_args, request_key, requests_from_args, setup_logging

log = logging.getLogger("jev_predict")

JEV_BASE_URL = "https://api.typesafe.ai"  # docs.typesafe.ai/api
DEFAULT_MODEL = "jev-latest"  # alias of jev-1.13.0 as of 2026-09 (docs.typesafe.ai/models)
DEFAULT_KEY_ENV = "TYPESAFE_API_KEY"
RETRY_STATUSES = {429, 529}  # rate limit / overloaded: docs say retry with exponential backoff


class JevError(RuntimeError):
    def __init__(self, status: int, message: str, raw_path: Path | None) -> None:
        super().__init__(f"HTTP {status}: {message} (raw response: {raw_path})")
        self.status = status
        self.raw_path = raw_path


def _post(url: str, body: dict[str, Any], api_key: str | None, timeout: float) -> tuple[int, str]:
    headers = {"content-type": "application/json", "accept": "application/json"}
    if api_key:
        headers["authorization"] = f"Bearer {api_key}"
    req = urllib.request.Request(url, data=json.dumps(body).encode(), headers=headers, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.status, resp.read().decode("utf-8", "replace")
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode("utf-8", "replace")


def call_systemone(
    body: dict[str, Any],
    *,
    base_url: str = JEV_BASE_URL,
    api_key: str | None = None,
    raw_dir: Path = Path("runs/jev"),
    use_cache: bool = True,
    timeout: float = 60.0,
    retries: int = 4,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """POST one request. Returns (parsed response, meta). Raw text is on disk before parsing."""
    url = base_url.rstrip("/") + "/v1/systemone"
    raw_dir.mkdir(parents=True, exist_ok=True)
    raw_path = raw_dir / f"{request_key(url, body)}.json"
    if use_cache and raw_path.exists():
        saved = json.loads(raw_path.read_text(encoding="utf-8"))
        if saved.get("status") == 200:
            log.info("cache hit %s", raw_path)
            return json.loads(saved["body"]), {"cached": True, "raw_path": str(raw_path), "latency_ms": saved.get("latency_ms")}
    for attempt in range(retries + 1):
        t0 = time.perf_counter()
        status, text = _post(url, body, api_key, timeout)
        latency_ms = round((time.perf_counter() - t0) * 1000, 1)
        # Save first, parse later: a parse bug must never cost a second paid call.
        raw_path.write_text(
            json.dumps({"url": url, "status": status, "latency_ms": latency_ms, "request": body, "body": text},
                       ensure_ascii=False, indent=1),
            encoding="utf-8",
        )
        log.info("POST %s -> %d in %.0f ms (raw: %s)", url, status, latency_ms, raw_path)
        if status in RETRY_STATUSES and attempt < retries:
            delay = min(2 ** attempt, 30)
            log.warning("HTTP %d, retrying in %ds (%d/%d)", status, delay, attempt + 1, retries)
            time.sleep(delay)
            continue
        break
    if status != 200:
        raise JevError(status, text[:300], raw_path)
    return json.loads(text), {"cached": False, "raw_path": str(raw_path), "latency_ms": latency_ms}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_input_args(ap)
    ap.add_argument("--base-url", default=os.environ.get("JEV_BASE_URL", JEV_BASE_URL),
                    help="API root (env JEV_BASE_URL); point at laya-serve to swap backends")
    ap.add_argument("--model", default=DEFAULT_MODEL, help="Jev model id (laya-serve auto-routes unknown ids)")
    ap.add_argument("--api-key-env", default=DEFAULT_KEY_ENV, help="name of the env var holding the key")
    ap.add_argument("--raw-dir", default="runs/jev", help="where raw responses are saved")
    ap.add_argument("--no-cache", action="store_true", help="always call the API, even if a raw response exists")
    ap.add_argument("--timeout", type=float, default=60.0)
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args(argv)
    setup_logging(args.verbose)

    api_key = os.environ.get(args.api_key_env) or None
    if not api_key and "typesafe.ai" in args.base_url:
        log.error("%s is not set; get a key at https://console.typesafe.ai/keys", args.api_key_env)
        return 2
    try:
        bodies = requests_from_args(args)
    except SchemaError as e:
        log.error("%s", e)
        return 2
    rc = 0
    for body in bodies:
        body = {"model": args.model, **body} if "model" not in body else body
        try:
            resp, meta = call_systemone(body, base_url=args.base_url, api_key=api_key, raw_dir=Path(args.raw_dir),
                                        use_cache=not args.no_cache, timeout=args.timeout)
            print(json.dumps({**resp, "_meta": meta}, ensure_ascii=False))
        except JevError as e:
            log.error("%s", e)
            print(json.dumps({"error": str(e), "status": e.status}, ensure_ascii=False))
            rc = 1
    return rc


if __name__ == "__main__":
    sys.exit(main())
