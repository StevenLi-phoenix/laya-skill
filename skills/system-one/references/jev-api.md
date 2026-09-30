# TypeSafe Jev API

Checked against the official docs on **2026-09-29**. Every fact below is marked **[verified]**
(read on docs.typesafe.ai / typesafe.ai / PyPI / npm that day) or **[unverified]** (only seen in
third-party material). No Jev key was used: the client in `scripts/jev_predict.py` is tested against
a local mock (`tests/test_jev_client.py`) and against `laya-serve`, not against the live API.

Sources: <https://docs.typesafe.ai/api>, <https://docs.typesafe.ai/introduction/quickstart>,
<https://docs.typesafe.ai/models>, <https://docs.typesafe.ai/sdk/python>,
<https://docs.typesafe.ai/sdk/javascript>, <https://docs.typesafe.ai/llms.txt>. Keys:
<https://console.typesafe.ai/keys>.

## Endpoint and auth [verified]

```
POST https://api.typesafe.ai/v1/systemone
Authorization: Bearer $TYPESAFE_API_KEY
Content-Type: application/json
```

Both official SDKs read `TYPESAFE_API_KEY` from the environment.

## Request [verified]

```json
{
  "state": "text | object | array",
  "model": "jev-latest",
  "questions": {
    "department": {"type": "choice", "instructions": "Which team?",
                   "criteria": {"billing": "refunds, invoices", "technical": "bugs", "other": null}},
    "severity":   {"type": "score", "instructions": "How severe is the issue?",
                   "criteria": ["Cosmetic", "Degraded with workaround", "Blocking"]},
    "urgent":     {"type": "noul", "instructions": "Does this message express urgency?",
                   "criteria": {"true": "...", "false": "..."}}
  }
}
```

- `questions` is a **map keyed by question id**.
- `choice.criteria`: required object, option → description; up to **255 options**; a description may
  be `null` or a structured object (`what`, `not_for`, `examples`).
- `score.criteria`: required **ordered array of 2–10 levels** (strings, or objects with `what`, `examples`).
- `noul.criteria`: optional `{"true": ..., "false": ...}`.
- `instructions` is required; for `choice` it may also be an object or array.

## Response [verified]

```json
{
  "model": "jev-1.13.0",
  "answers": {
    "department": {"type": "choice", "choice": "billing", "confidence": 0.91,
                   "probabilities": {"billing": 0.94, "technical": 0.04, "other": 0.02}},
    "severity":   {"type": "score", "score": 1.43, "confidence": 0.35,
                   "legend": {"0": "Cosmetic", "1": "Degraded with workaround", "2": "Blocking"},
                   "probabilities": {"0": 0.0, "1": 0.57, "2": 0.43}},
    "urgent":     {"type": "noul", "noul": 0.99}
  },
  "usage": {"input_tokens": 112, "output_tokens": 0}
}
```

- `score` = Σ level × probability on the 0..max_level scale.
- `noul` has **no** confidence field. `choice`/`score` `confidence` is Jev's own normalisation of
  `p_max` (docs give `(3·p_max − 1)/2` for 3 options; Laya's README states the general
  `(n·p_max − 1)/(n − 1)`). It is **not** Laya's `confidence` (1 − normalised entropy). Compare the two
  backends on the probability of the chosen answer instead (`systemone.answer_confidence`).

## Errors [verified]

| status | meaning | client behaviour (`jev_predict.py`) |
|---|---|---|
| 401 | invalid / missing key | fail, no retry |
| 422 | validation error | fail, no retry |
| 429 | rate limited | exponential backoff, up to 4 retries |
| 529 | overloaded | exponential backoff, up to 4 retries |

## Models, limits, pricing [verified]

- Models: `jev-1.13.0` (current), `jev-latest` and `jev-preview` (aliases, both 1.13.0 at the time of checking).
- Context: 64k tokens per request; 32k for `state` + the longest question.
- Rate limits: 100k tokens/s and 40 requests/s ("can change without notice").
- Text only. English is best; CJK and other languages work "not equally well".
- Price: **$0.042 per million input tokens** ($42 per billion); output tokens are free.

## SDKs [verified]

- Python: `uv add typesafe-sdk` (import `typesafe_sdk`; `TypeSafeClient`, `AsyncTypeSafeClient`,
  helpers `Noul`, `Choice`, `Score`; `client.system_one(state=..., questions={...})`).
- JS/TS: `pnpm add @typesafe-ai/sdk` (0.6.0 on npm; `TypeSafeClient`, `client.systemOne({state, questions})`).
- `scripts/jev_predict.py` uses stdlib `urllib` instead, so it can save the raw body before
  parsing and target any Jev-compatible base URL.

## Unverified / third-party

- Launch as early access on 2026-09-15: search snippets only.
- Accuracy on the typed-decisions benchmark 0.727, ECE 0.144, soft accuracy 0.580: these come from
  the Laya README / benchmark tables ("Jev 1.13.0 (published)"). TypeSafe's docs have no benchmark
  page (`/models/benchmarks` returned 404). The Laya README's own head-to-head table gives Jev an ECE
  of 0.246 elsewhere, so the numbers are not even consistent with each other.
- p50 latency 236–276 ms: third-party measurements ([AbdelStark/jev-benchmarks](https://github.com/AbdelStark/jev-benchmarks),
  [nibzard/decision-model-benchmark](https://github.com/nibzard/decision-model-benchmark)).
- Homepage marketing ("193.6x faster, 444.6x cheaper", 0.114 s): vendor claims.

## Using it from this skill

```bash
export TYPESAFE_API_KEY=...        # never paste the key into a command line or a file in git
uv run scripts/jev_predict.py --state "Stripe sync failing for 3 days, losing sales" --questions q.json
uv run scripts/jev_predict.py --batch rows.jsonl --questions q.json --raw-dir runs/jev
uv run scripts/compare.py data.jsonl --backend jev --backend laya:auto --json runs/jev-vs-laya.json
```

Raw bodies land in `runs/jev/<hash>.json` (`{url, status, latency_ms, request, body}`) before parsing.
A later identical request is served from that file; pass `--no-cache` to call again.
