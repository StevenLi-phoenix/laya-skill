# Laya API (laya 0.3.22)

Source of truth: <https://github.com/NandhaKishorM/laya> at `6d942c9` (PyPI `laya==0.3.22`,
Apache-2.0, Convai Innovations). Python ≥ 3.10; use `uv venv --python 3.12`.

## Checkpoints

| name (Router) | Hub | encoder | params | context | use |
|---|---|---|---|---|---|
| `english` | `convaiinnovations/laya` | ModernBERT-large | 421M | 512 | English |
| `multilingual` | `convaiinnovations/laya` subfolder `multilingual` (also `convaiinnovations/laya-multilingual`) | mmBERT-base | 322M | 1024, up to 8192 with `max_len=8192` | 100+ languages, ~2x faster |
| `typed-decisions` | subfolder `typed-decisions` (also `convaiinnovations/laya-typed-decisions`) | ModernBERT-large | 421M | 1024 | fine-tuned on the typed-decisions workflows |

Aliases resolve the same everywhere (`en`, `ml`, `typed` …). Weights download on first use
into the Hugging Face cache (~0.6–0.85 GB each).

## Python

```python
from laya import Router
import laya

router = Router()                       # Router(preload=True) loads all three up front
r = router.predict(state, questions)    # auto-route by script / language
r = router.predict(state, questions, model="multilingual", max_len=8192)
r["answers"]["department"]["choice"]; r["routing"]["model"]

agent = laya.load("convaiinnovations/laya", subfolder="multilingual")   # or a local dir
agent.predict(state, questions, max_len=None, head_max_len=None, min_confidence=None)
agent.predict_batch(states, questions, batch_size=64, sort_by_length=True)
agent.predict_long(state, questions, window=256)   # windowed scan of long documents
laya.predict_shortlist(agent, state, questions, embed_fn=laya.embed_fn_from_agent(agent), k=20)
```

`state` may be a string, dict or list. An empty question dict returns `answers: {}` with no forward pass.

## Response (Jev-shaped, plus Laya extras)

Measured on this machine (English checkpoint):

```json
{"model": "laya-rl-agent",
 "answers": {
   "d": {"type": "choice", "choice": "billing", "probabilities": {"billing": 0.8109, "tech": 0.1891},
         "confidence": 0.3004, "answer_confidence": 0.8109, "action": {"act_probability": 1.0}},
   "u": {"type": "score", "score": 0.9249, "legend": {"0": "low", "1": "mid", "2": "high"},
         "probabilities": {"0": 0.2388, "1": 0.5975, "2": 0.1637}, "confidence": 0.139, "answer_confidence": 0.5975},
   "n": {"type": "noul", "noul": 0.6404, "confidence": 0.6404, "answer_confidence": 0.6404}},
 "usage": {"input_tokens": 95, "output_tokens": 0, "state_tokens": 8, "state_tokens_dropped": 0,
           "truncated": false, "truncated_questions": []},
 "routing": {"model": "english", "repo": "convaiinnovations/laya", "reason": "English Latin text", "...": "..."}}
```

- `confidence` on choice/score = 1 − normalised entropy (**not** Jev's formula).
- `answer_confidence` = calibrated probability of the reported answer, on every type. Gate on this.
- `action.act_probability` carries no usable signal yet (upstream #185). Ignore it.
- `usage.truncated` / `state_tokens_dropped` say when the state did not fit. Check them on long inputs.

## Question details that matter

- `noul`: always give `criteria: {"true": ..., "false": ...}` (any other keys are rejected). Optional
  `labels: {"true": "A", "false": "B"}` changes only the model-facing words (upstream #156).
- `choice`: avoid boolean-word labels (`yes`/`no`/`true`/`false`), since the model can follow the label.
  Negation is fragile (upstream #377), so validate cancel/no-cancel style questions on your data.
- `score`: every level needs a description. `laya-multilingual` rarely picks the first level (#131);
  route English score questions to `english`.
- `option_order`: a permutation that changes presentation only. Average the k rotations in one
  forward pass to cancel position bias (README "Option order").
- **Token budget:** the sequence is split between options (`head_max_len`: 192 on `english`, 256 on the
  others) and the state (`max_len − actual_head_len − 1`). Past ~`head_max_len/4` options, each option
  gets ~3–4 tokens and accuracy collapses (Banking77: 0.425). Fixes: `head_max_len=512, max_len=1024`
  per request, `predict_shortlist`, or a coarse→fine question split.

## CLI (installed with the package)

```bash
laya "I was charged twice" --predict                  # route + answer
laya "Mein Konto wurde zweimal belastet" --lang de
laya "..." --questions q.json --max-len 1024 --head-max-len 384
laya --batch tickets.txt --predict --json --batch-size 8 --sort-by-length
laya-evals run data.jsonl --model english --min-accuracy 0.8 --max-ece 0.05 --slice language
```

## Calibration

Shipped checkpoints are over-confident; `laya-multilingual` ships with no fitted temperatures.
Per-type (and per option-count bucket) temperatures, fitted on held-out labelled data:

```python
from laya.calibrate import records_from_labeled
recs = records_from_labeled(agent, [(state, questions, {"qid": [p0, p1, ...]}), ...])
agent.fit_temperatures(recs, compute_ece=True)   # per-bucket needs >= 2000 records, per-type >= 10
agent.save_calibration("calib.json")
laya.load(..., calibration="calib.json")         # or laya_predict.py --calibration calib.json
```

Temperatures are clamped to [0.5, 5.0] at load time; argmax (accuracy) never changes, only confidence.

## Performance measured here

Mac mini M4 16 GB, MPS, `laya_predict`/quickstart: English first call 19.1 s (download done,
includes load), warm 78 ms for 3 questions; multilingual 100–120 ms warm (ZH / ES). MacBook Pro M4 Pro:
first call 63 s (incl. download), warm 46 ms. Upstream T4 numbers: 39.5 ms (`laya`) / 32.8 ms
(`laya-multilingual`) for one question.
