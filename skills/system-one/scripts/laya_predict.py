#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10,<3.14"
# dependencies = ["laya==0.3.22"]
# ///
"""Run Laya locally: state + typed questions in, calibrated answers out (Jev-shaped JSON).

--model picks the checkpoint:
  auto              Router picks english / multilingual per request (default)
  english | multilingual | typed-decisions   pin a bundled checkpoint (aliases like ml work)
  <dir> | <org/repo>                         a local or Hub checkpoint, e.g. a fine-tune

Examples:
  uv run laya_predict.py --state "billed twice, refund" --questions questions.json
  uv run laya_predict.py --model multilingual --max-len 8192 --state-file long.txt --questions q.json
  uv run laya_predict.py --model ./runs/ft/checkpoint --batch rows.jsonl --questions q.json
First call downloads the checkpoint from Hugging Face (0.6-0.85 GB each).
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
from systemone import SchemaError, add_input_args, requests_from_args, setup_logging

log = logging.getLogger("laya_predict")
ROUTER_NAMES = {"english", "en", "laya", "multilingual", "ml", "typed-decisions", "typed", "typed_decisions"}


class LayaRunner:
    """One interface over a Router (named checkpoints) or a single Agent (path / Hub id)."""

    def __init__(self, model: str = "auto", device: str | None = None, calibration: str | None = None) -> None:
        import laya

        self.model = model
        t0 = time.perf_counter()
        if model == "auto" or model.lower() in ROUTER_NAMES:
            if calibration:
                raise SystemExit("--calibration needs a single checkpoint (--model <dir or repo>), not a Router name")
            self.router, self.agent = laya.Router(device=device), None
        else:
            self.router, self.agent = None, laya.load(model, device=device, calibration=calibration)
            log.info("loaded %s in %.1fs", model, time.perf_counter() - t0)

    def predict(self, state: Any, questions: dict[str, Any], **kw: Any) -> dict[str, Any]:
        kw = {k: v for k, v in kw.items() if v is not None}
        if self.agent is not None:
            return self.agent.predict(state, questions, **kw)
        if self.model != "auto":
            kw["model"] = self.model
        return self.router.predict(state, questions, **kw)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_input_args(ap)
    ap.add_argument("--model", default="auto", help="auto | english | multilingual | typed-decisions | <dir> | <org/repo>")
    ap.add_argument("--device", default=None, help="cpu | mps | cuda (default: best available)")
    ap.add_argument("--max-len", type=int, default=None, help="token budget; multilingual reads up to 8192")
    ap.add_argument("--head-max-len", type=int, default=None, help="option budget; raise for >20 choice options")
    ap.add_argument("--min-confidence", type=float, default=None, help="flag answers below this answer_confidence")
    ap.add_argument("--calibration", default=None, help="calibration JSON from Agent.save_calibration")
    ap.add_argument("--repeat", type=int, default=1, help="run each request N times and report warm p50 latency")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args(argv)
    setup_logging(args.verbose)
    try:
        bodies = requests_from_args(args)
    except SchemaError as e:
        log.error("%s", e)
        return 2

    runner = LayaRunner(args.model, device=args.device, calibration=args.calibration)
    for body in bodies:
        # A Jev body may carry "model": "jev-latest"; only an explicit --model pins a checkpoint.
        times: list[float] = []
        result: dict[str, Any] = {}
        for _ in range(max(1, args.repeat)):
            t0 = time.perf_counter()
            result = runner.predict(body["state"], body["questions"], max_len=args.max_len,
                                    head_max_len=args.head_max_len, min_confidence=args.min_confidence)
            times.append((time.perf_counter() - t0) * 1000)
        meta = {"latency_ms": round(times[0], 1), "model_arg": args.model}
        if len(times) > 1:
            warm = sorted(times[1:])
            meta["warm_p50_ms"] = round(warm[len(warm) // 2], 1)
        log.info("answered in %.0f ms (%s)", times[0], (result.get("routing") or {}).get("model", args.model))
        print(json.dumps({**result, "_meta": meta}, ensure_ascii=False, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
