from systemone import agreement, answer_confidence, answer_value, ece, score_results


def ch(label, p):
    return {"type": "choice", "choice": label, "probabilities": {label: p, "x": 1 - p}}


def test_answer_confidence_jev_shapes():
    # Jev noul has no confidence field; Jev choice confidence is a different formula -> use probs.
    assert answer_confidence({"type": "noul", "noul": 0.2}) == 0.8
    assert answer_confidence({**ch("a", 0.7), "confidence": 0.1}) == 0.7
    assert answer_confidence({"type": "score", "score": 1.4, "probabilities": {"0": 0.1, "1": 0.4, "2": 0.5}}) == 0.5
    assert answer_confidence({"type": "noul", "noul": 0.9, "answer_confidence": 0.77}) == 0.77


def test_answer_value():
    assert answer_value({"type": "noul", "noul": 0.5}) is True
    assert answer_value(ch("a", 0.9)) == "a"


def test_ece_perfect_and_bad():
    assert ece([1.0, 1.0], [True, True]) == 0.0
    assert ece([0.9, 0.9], [False, False]) == 0.9
    assert ece([], []) is None


def test_score_results_and_agreement():
    rows = [{"expected": {"d": "a", "n": True, "s": 2}}, {"expected": {"d": "b", "n": False, "s": 0}}]
    r1 = {"answers": {"d": ch("a", 0.9), "n": {"type": "noul", "noul": 0.8},
                      "s": {"type": "score", "score": 1.7, "probabilities": {"0": 0.1, "1": 0.1, "2": 0.8}}},
          "_meta": {"latency_ms": 10}}
    r2 = {"answers": {"d": ch("a", 0.6), "n": {"type": "noul", "noul": 0.3},
                      "s": {"type": "score", "score": 0.2, "probabilities": {"0": 0.8, "1": 0.2, "2": 0.0}}},
          "_meta": {"latency_ms": 30}}
    m = score_results(rows, [r1, r2])
    assert m["n_decisions"] == 6
    assert m["choice_accuracy"] == 0.5 and m["noul_accuracy"] == 1.0 and m["score_accuracy"] == 1.0
    assert abs(m["score_mae"] - 0.25) < 1e-9
    assert m["latency_p50_ms"] == 30
    m2 = score_results(rows, [r1, None])
    assert m2["n_failed"] == 1 and m2["n_decisions"] == 3
    assert agreement([r1, r2], [r1, r2]) == 1.0
    assert agreement([r1], [r2]) == 1 / 3 or round(agreement([r1], [r2]), 4) == 0.3333
