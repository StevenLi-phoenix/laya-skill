#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10,<3.14"
# dependencies = ["laya==0.3.22"]
# ///
"""Score two or more backends on the same labelled JSONL: accuracy, ECE, latency, agreement.

Backends (repeat --backend):
  laya:auto | laya:english | laya:multilingual | laya:typed-decisions   local Laya via Router
  laya:<dir-or-org/repo>                                                a single checkpoint (e.g. a fine-tune)
  jev                                                                   TypeSafe Jev (key in $TYPESAFE_API_KEY)
  http:<base-url>                                                       any Jev-compatible server (laya-serve)

Dataset: one JSON object per line, the `laya-evals` format:
  {"state": ..., "questions": {...}, "expected": {"qid": "label" | true/false | level}}

Example:
  uv run compare.py data.jsonl --backend laya:english --backend laya:multilingual --json report.json
"""
from __future__ import annotations

import argparse
import itertools
import json
import logging
import os
import sys
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
from systemone import agreement, read_jsonl, score_results, setup_logging, validate_questions

log = logging.getLogger("compare")
Predict = Callable[[Any, dict[str, Any]], dict[str, Any]]


def make_backend(spec: str, args: argparse.Namespace) -> Predict:
    """Turn a backend spec into predict(state, questions) -> Jev-shaped response with _meta."""
    if spec.startswith("laya:"):
        from laya_predict import LayaRunner

        runner = LayaRunner(spec[5:], device=args.device)

        def predict(state: Any, questions: dict[str, Any]) -> dict[str, Any]:
            t0 = time.perf_counter()
            r = runner.predict(state, questions, max_len=args.max_len, head_max_len=args.head_max_len)
            return {**r, "_meta": {"latency_ms": round((time.perf_counter() - t0) * 1000, 1)}}

        return predict
    if spec == "jev" or spec.startswith("http:"):
        from jev_predict import JEV_BASE_URL, call_systemone

        base = JEV_BASE_URL if spec == "jev" else spec[5:]
        key = os.environ.get(args.api_key_env) or None
        if spec == "jev" and not key:
            raise SystemExit(f"backend 'jev' needs ${args.api_key_env}")

        def predict(state: Any, questions: dict[str, Any]) -> dict[str, Any]:
            body = {"state": state, "model": args.jev_model, "questions": questions}
            resp, meta = call_systemone(body, base_url=base, api_key=key, raw_dir=Path(args.raw_dir))
            return {**resp, "_meta": meta}

        return predict
    raise SystemExit(f"unknown backend {spec!r}")


def run_backend(predict: Predict, rows: list[dict[str, Any]], name: str, warmup: bool = True) -> list[dict[str, Any] | None]:
    if warmup and rows:  # first call loads weights / opens connections; keep it out of latency
        predict(rows[0]["state"], rows[0]["questions"])
    out: list[dict[str, Any] | None] = []
    for i, row in enumerate(rows):
        try:
            out.append(predict(row["state"], row["questions"]))
        except Exception as e:  # noqa: BLE001 - keep scoring the rest; failures are counted in n_failed
            log.warning("%s row %d failed: %s", name, i, e)
            out.append(None)
    return out


def render_markdown(report: dict[str, Any]) -> str:
    cols = ["accuracy", "choice_accuracy", "noul_accuracy", "score_accuracy", "score_mae", "ece", "latency_p50_ms", "n_failed"]
    lines = ["| backend | " + " | ".join(cols) + " |", "|---" * (len(cols) + 1) + "|"]
    for name, m in report["backends"].items():
        lines.append(f"| {name} | " + " | ".join("—" if m.get(c) is None else str(m.get(c)) for c in cols) + " |")
    if report["agreement"]:
        lines += ["", "| pair | agreement |", "|---|---|"]
        lines += [f"| {k} | {v} |" for k, v in report["agreement"].items()]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dataset", help="labelled JSONL (laya-evals format)")
    ap.add_argument("--backend", action="append", required=True, help="repeatable; see above")
    ap.add_argument("--limit", type=int, default=None, help="only the first N rows")
    ap.add_argument("--device", default=None)
    ap.add_argument("--max-len", type=int, default=None)
    ap.add_argument("--head-max-len", type=int, default=None)
    ap.add_argument("--jev-model", default="jev-latest")
    ap.add_argument("--api-key-env", default="TYPESAFE_API_KEY")
    ap.add_argument("--raw-dir", default="runs/jev")
    ap.add_argument("--json", help="write the full report (metrics + per-row answers) here")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args(argv)
    setup_logging(args.verbose)

    rows = read_jsonl(args.dataset)[: args.limit]
    for i, row in enumerate(rows):
        validate_questions(row.get("questions"))
        if "expected" not in row:
            raise SystemExit(f"row {i} has no 'expected'")
    log.info("%d rows from %s", len(rows), args.dataset)

    results: dict[str, list[dict[str, Any] | None]] = {}
    for spec in args.backend:
        t0 = time.perf_counter()
        results[spec] = run_backend(make_backend(spec, args), rows, spec)
        log.info("%s done in %.1fs", spec, time.perf_counter() - t0)
    report = {
        "dataset": args.dataset,
        "backends": {k: score_results(rows, v) for k, v in results.items()},
        "agreement": {f"{a} vs {b}": agreement(results[a], results[b]) for a, b in itertools.combinations(results, 2)},
    }
    print(render_markdown(report))
    if args.json:
        Path(args.json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json).write_text(json.dumps({**report, "results": results}, ensure_ascii=False, indent=1, default=str))
        log.info("report -> %s", args.json)
    return 0


if __name__ == "__main__":
    sys.exit(main())
