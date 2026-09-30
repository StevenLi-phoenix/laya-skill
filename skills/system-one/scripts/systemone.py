"""Shared helpers for the system-one scripts: question-schema checks, JSON/JSONL IO, logging,
and answer normalisation that works the same for Jev and Laya responses.

Stdlib only, so it can be imported by every script (including the Jev client, which has no
third-party dependencies) and tested without model weights.
"""
from __future__ import annotations

import hashlib
import json
import logging
import sys
from collections.abc import Iterable
from pathlib import Path
from typing import Any

log = logging.getLogger("systemone")

QTYPES = ("choice", "score", "noul")
JEV_MAX_CHOICE_OPTIONS = 255  # docs.typesafe.ai/api
LAYA_SERVE_MAX_CHOICE_OPTIONS = 100  # laya/serve.py MAX_CHOICE_OPTIONS (413 above this)
LAYA_SOFT_OPTION_LIMIT = 20  # upstream "Honest limits": accuracy drops past ~20 options
SCORE_LEVELS = (2, 10)  # Jev accepts 2-10 levels
BOOLEAN_WORDS = {"true", "false", "yes", "no"}


class SchemaError(ValueError):
    """A question set that neither Jev nor Laya would accept."""


def setup_logging(verbose: bool = False) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        stream=sys.stderr,
    )


def load_json_arg(value: str) -> Any:
    """Parse `value` as JSON, or read it as a JSON file when it names an existing path."""
    try:
        is_file = Path(value).is_file()
    except OSError:  # inline JSON longer than a file name
        is_file = False
    if is_file:
        return json.loads(Path(value).read_text(encoding="utf-8"))
    return json.loads(value)


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    """Read JSONL; blank lines and `#` comments are skipped (same rules as `laya-evals`)."""
    rows: list[dict[str, Any]] = []
    for n, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        try:
            row = json.loads(s)
        except json.JSONDecodeError as e:
            raise SchemaError(f"{path}:{n}: invalid JSON ({e})") from e
        if not isinstance(row, dict):
            raise SchemaError(f"{path}:{n}: each line must be a JSON object")
        rows.append(row)
    return rows


def write_jsonl(path: str | Path, rows: Iterable[dict[str, Any]]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.writelines(json.dumps(row, ensure_ascii=False) + "\n" for row in rows)


def validate_questions(questions: Any) -> list[str]:
    """Raise SchemaError on anything Jev or Laya rejects; return warnings for known weak spots.

    Rules are the intersection of docs.typesafe.ai/api and laya 0.3.22, so a question set that
    passes can be sent to either backend unchanged.
    """
    if not isinstance(questions, dict) or not questions:
        raise SchemaError("questions must be a non-empty object keyed by question id")
    warnings: list[str] = []
    for qid, q in questions.items():
        where = f"question {qid!r}"
        if not isinstance(q, dict):
            raise SchemaError(f"{where} must be an object")
        qtype = q.get("type")
        if qtype not in QTYPES:
            raise SchemaError(f"{where}: type must be one of {QTYPES}, got {qtype!r}")
        ins = q.get("instructions")
        if ins in (None, "", [], {}):
            raise SchemaError(f"{where}: instructions is required")
        crit = q.get("criteria")
        if qtype == "choice":
            if not isinstance(crit, dict) or len(crit) < 2:
                raise SchemaError(f"{where}: choice needs criteria as an object with >= 2 options")
            k = len(crit)
            if k > JEV_MAX_CHOICE_OPTIONS:
                raise SchemaError(f"{where}: {k} options; Jev caps choice at {JEV_MAX_CHOICE_OPTIONS}")
            if k > LAYA_SERVE_MAX_CHOICE_OPTIONS:
                warnings.append(f"{where}: {k} options; laya-serve rejects > {LAYA_SERVE_MAX_CHOICE_OPTIONS} (413)")
            if k > LAYA_SOFT_OPTION_LIMIT:
                warnings.append(
                    f"{where}: {k} options; Laya accuracy drops past ~{LAYA_SOFT_OPTION_LIMIT} "
                    "(raise head_max_len or use laya.predict_shortlist)"
                )
            bool_labels = sorted(str(label) for label in crit if str(label).lower() in BOOLEAN_WORDS)
            if bool_labels:
                warnings.append(
                    f"{where}: boolean-word choice labels {bool_labels}; Laya can follow the label "
                    "instead of the description - use semantic or A/B labels"
                )
            if any(v is None for v in crit.values()):
                warnings.append(f"{where}: null option descriptions are allowed by Jev; Laya reads only the label then")
        elif qtype == "score":
            if not isinstance(crit, list):
                raise SchemaError(f"{where}: score criteria must be an ordered list of level descriptions")
            lo, hi = SCORE_LEVELS
            if not lo <= len(crit) <= hi:
                raise SchemaError(f"{where}: score needs {lo}-{hi} levels, got {len(crit)}")
            if any(v in (None, "") for v in crit):
                raise SchemaError(f"{where}: every score level needs a description (Laya rejects null levels)")
        else:  # noul
            if crit is None:
                warnings.append(
                    f"{where}: noul without criteria; zero-shot Laya noul is unreliable here "
                    "(upstream #156, checkpoints disagree) - give {'true': ..., 'false': ...}"
                )
            elif not isinstance(crit, dict) or set(crit) != {"true", "false"}:
                raise SchemaError(f"{where}: noul criteria must have exactly the keys 'true' and 'false'")
    return warnings


def answer_confidence(answer: dict[str, Any]) -> float | None:
    """Probability of the reported answer, comparable across Jev and Laya.

    Laya reports it as `answer_confidence`. Jev's `confidence` is a different formula
    ((n*p_max-1)/(n-1)) and nouls carry none, so derive it from the distribution instead.
    """
    if isinstance(answer.get("answer_confidence"), (int, float)):
        return float(answer["answer_confidence"])
    t = answer.get("type")
    probs = answer.get("probabilities") or {}
    if t == "choice" and answer.get("choice") in probs:
        return float(probs[answer["choice"]])
    if t == "score" and probs:
        return float(max(probs.values()))
    if t == "noul" and isinstance(answer.get("noul"), (int, float)):
        p = float(answer["noul"])
        return max(p, 1.0 - p)
    return None


def answer_value(answer: dict[str, Any]) -> Any:
    """The decision itself: a label for choice, a bool for noul, the expected level for score."""
    t = answer.get("type")
    if t == "choice":
        return answer.get("choice")
    if t == "noul":
        return float(answer.get("noul", 0.0)) >= 0.5
    if t == "score":
        return float(answer.get("score", 0.0))
    return None


def add_input_args(parser: Any) -> None:
    """The input flags every predict script shares (argparse parser)."""
    g = parser.add_argument_group("input (one of --request, --state, --state-file, --batch)")
    g.add_argument("--request", help="full request body JSON (inline or file): {state, questions[, model]}")
    g.add_argument("--state", help="state as text (or JSON text for a structured state)")
    g.add_argument("--state-file", help="read the state from a file (.json is parsed, anything else is text)")
    g.add_argument("--batch", help="JSONL of rows with 'state' (and optionally 'questions'); one output line per row")
    parser.add_argument("--questions", help="questions JSON (inline or file), keyed by question id")


def _parse_state(text: str) -> Any:
    s = text.strip()
    if s[:1] in "{[":
        try:
            return json.loads(s)
        except json.JSONDecodeError:
            pass
    return text


def requests_from_args(args: Any) -> list[dict[str, Any]]:
    """Build one or more {state, questions} bodies from CLI args, validating every question set."""
    shared_q = load_json_arg(args.questions) if args.questions else None
    bodies: list[dict[str, Any]]
    if args.request:
        body = load_json_arg(args.request)
        if shared_q is not None:
            body["questions"] = shared_q
        bodies = [body]
    elif args.batch:
        bodies = [{"state": r["state"], "questions": r.get("questions", shared_q)} for r in read_jsonl(args.batch)]
    elif args.state is not None:
        bodies = [{"state": _parse_state(args.state), "questions": shared_q}]
    elif args.state_file:
        p = Path(args.state_file)
        text = p.read_text(encoding="utf-8")
        bodies = [{"state": json.loads(text) if p.suffix == ".json" else text, "questions": shared_q}]
    else:
        raise SchemaError("give one of --request, --state, --state-file or --batch")
    for i, body in enumerate(bodies):
        if body.get("state") in (None, ""):
            raise SchemaError(f"request {i}: state is empty")
        for w in validate_questions(body.get("questions")):
            log.warning("request %d: %s", i, w)
    return bodies


def request_key(url: str, body: dict[str, Any]) -> str:
    """Stable cache key for a request (URL + canonical JSON body)."""
    canon = json.dumps(body, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(f"{url}\n{canon}".encode()).hexdigest()[:24]


# ---------------------------------------------------------------- evaluation (pure python)

def _decision_correct(answer: dict[str, Any], expected: Any) -> bool | None:
    """Exact match for choice/noul; for score, the rounded expected level must match."""
    v = answer_value(answer)
    t = answer.get("type")
    if t == "choice" and isinstance(expected, str):
        return v == expected
    if t == "noul" and isinstance(expected, bool):
        return v == expected
    if t == "score" and isinstance(expected, (int, float)) and not isinstance(expected, bool):
        return round(float(v)) == round(float(expected))
    return None


def ece(confidences: list[float], corrects: list[bool], bins: int = 15) -> float | None:
    """Expected calibration error with equal-width bins (same definition as laya.evals)."""
    if not confidences:
        return None
    total, n = 0.0, len(confidences)
    for b in range(bins):
        lo, hi = b / bins, (b + 1) / bins
        idx = [i for i, c in enumerate(confidences) if (lo < c <= hi) or (b == 0 and c == 0.0)]
        if idx:
            acc = sum(corrects[i] for i in idx) / len(idx)
            conf = sum(confidences[i] for i in idx) / len(idx)
            total += len(idx) / n * abs(acc - conf)
    return round(total, 4)


def score_results(rows: list[dict[str, Any]], results: list[dict[str, Any] | None]) -> dict[str, Any]:
    """Accuracy per primitive, overall accuracy, score MAE, ECE and latency for one backend.

    `rows` carry `expected` ({qid: label | bool | level}); `results` are Jev/Laya responses
    aligned with rows (None = request failed). Pure python, so it runs without weights.
    """
    per: dict[str, list[bool]] = {"choice": [], "noul": [], "score": []}
    abs_err: list[float] = []
    confs: list[float] = []
    corrects: list[bool] = []
    lat = [r["_meta"]["latency_ms"] for r in results
           if r and isinstance((r.get("_meta") or {}).get("latency_ms"), (int, float))]
    failed = sum(r is None for r in results)
    for row, res in zip(rows, results):
        if res is None:
            continue
        for qid, exp in (row.get("expected") or {}).items():
            ans = (res.get("answers") or {}).get(qid)
            if not ans:
                continue
            ok = _decision_correct(ans, exp)
            if ok is None:
                continue
            per[ans["type"]].append(ok)
            if ans["type"] == "score":
                abs_err.append(abs(float(ans.get("score", 0.0)) - float(exp)))
            c = answer_confidence(ans)
            if c is not None:
                confs.append(c)
                corrects.append(ok)
    allc = [x for v in per.values() for x in v]
    out: dict[str, Any] = {
        "n_rows": len(rows), "n_failed": failed, "n_decisions": len(allc),
        "accuracy": round(sum(allc) / len(allc), 4) if allc else None,
        "ece": ece(confs, corrects),
        "score_mae": round(sum(abs_err) / len(abs_err), 4) if abs_err else None,
    }
    for t, v in per.items():
        out[f"{t}_accuracy"] = round(sum(v) / len(v), 4) if v else None
        out[f"n_{t}"] = len(v)
    if lat:
        s = sorted(lat)
        out["latency_p50_ms"] = s[len(s) // 2]
        out["latency_p90_ms"] = s[min(len(s) - 1, int(len(s) * 0.9))]
    return out


def agreement(a: list[dict[str, Any] | None], b: list[dict[str, Any] | None]) -> float | None:
    """Fraction of decisions on which two backends give the same answer (score: same rounded level)."""
    same = total = 0
    for ra, rb in zip(a, b):
        if not ra or not rb:
            continue
        for qid, xa in (ra.get("answers") or {}).items():
            xb = (rb.get("answers") or {}).get(qid)
            if not xb or xa.get("type") != xb.get("type"):
                continue
            va, vb = answer_value(xa), answer_value(xb)
            if xa["type"] == "score":
                va, vb = round(va), round(vb)
            total += 1
            same += va == vb
    return round(same / total, 4) if total else None
