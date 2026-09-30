import pytest
from finetune_laya import data_fingerprint, split_rows, target_for

C = {"type": "choice", "instructions": "x", "criteria": {"a": "1", "b": "2", "c": "3"}}
S = {"type": "score", "instructions": "x", "criteria": ["l", "m", "h"]}
N = {"type": "noul", "instructions": "x"}


def test_choice_hard_label_smoothed():
    t = target_for(C, "b", smoothing=0.0)
    assert t == [0.0, 1.0, 0.0]
    t = target_for(C, "b", smoothing=0.3)
    assert abs(sum(t) - 1) < 1e-9 and t[1] == pytest.approx(0.8)


def test_noul_order_is_false_true():
    assert target_for(N, True, smoothing=0.0) == [0.0, 1.0]
    assert target_for(N, False, smoothing=0.0) == [1.0, 0.0]


def test_score_fractional_level_splits_mass():
    assert target_for(S, 1.25, smoothing=0.0) == [0.0, 0.75, 0.25]
    assert target_for(S, 2, smoothing=0.0) == [0.0, 0.0, 1.0]


def test_gold_overrides_expected_and_bad_labels_skip():
    t = target_for(C, "a", gold={"probabilities": {"a": 1, "c": 3}})
    assert t == [0.25, 0.0, 0.75]
    assert target_for(C, "zzz") is None
    assert target_for(N, "yes") is None
    assert target_for(S, 5) is None


def test_split_deterministic_and_disjoint():
    rows = [{"state": str(i)} for i in range(20)]
    a, b = split_rows(rows, 0.25, seed=1)
    a2, b2 = split_rows(rows, 0.25, seed=1)
    assert (a, b) == (a2, b2) and len(b) == 5 and len(a) == 15
    assert not {r["state"] for r in a} & {r["state"] for r in b}
    assert data_fingerprint(rows) == data_fingerprint(list(rows))
