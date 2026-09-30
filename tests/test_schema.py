import json

import pytest
from conftest import SKILL
from systemone import SchemaError, validate_questions

CHOICE = {"type": "choice", "instructions": "team?", "criteria": {"billing": "refunds", "tech": "bugs"}}
SCORE = {"type": "score", "instructions": "urgency?", "criteria": ["low", "mid", "high"]}
NOUL = {"type": "noul", "instructions": "cancel?", "criteria": {"true": "yes", "false": "no"}}


def test_valid_set_has_no_warnings():
    assert validate_questions({"d": CHOICE, "u": SCORE, "n": NOUL}) == []


def test_bundled_example_questions_are_valid():
    q = json.loads((SKILL / "assets/examples/questions.json").read_text())
    assert validate_questions(q) == []


@pytest.mark.parametrize("bad", [
    {},
    {"q": {"type": "maybe", "instructions": "x"}},
    {"q": {"type": "choice", "criteria": {"a": 1, "b": 2}}},
    {"q": {**CHOICE, "criteria": {"only": "one"}}},
    {"q": {**CHOICE, "criteria": {f"o{i}": "x" for i in range(256)}}},
    {"q": {**SCORE, "criteria": {"low": 1}}},
    {"q": {**SCORE, "criteria": ["one"]}},
    {"q": {**SCORE, "criteria": [str(i) for i in range(11)]}},
    {"q": {**SCORE, "criteria": ["low", None]}},
    {"q": {**NOUL, "criteria": {"yes": "a", "no": "b"}}},
])
def test_rejects_invalid(bad):
    with pytest.raises(SchemaError):
        validate_questions(bad)


def test_warnings_for_known_weak_spots():
    many = {**CHOICE, "criteria": {f"o{i}": "x" for i in range(120)}}
    w = validate_questions({"m": many, "b": {**CHOICE, "criteria": {"yes": "a", "no": "b"}},
                            "n": {"type": "noul", "instructions": "x"}})
    text = "\n".join(w)
    assert "laya-serve rejects" in text and "accuracy drops" in text
    assert "boolean-word" in text and "noul without criteria" in text
