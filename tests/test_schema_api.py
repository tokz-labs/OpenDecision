"""Public decision schema: specs, coercion and validation."""

from __future__ import annotations

import pytest

from opendecision import (
    Bool,
    Candidate,
    Choice,
    Decision,
    DecisionKind,
    MultiLabel,
    SchemaError,
    Score,
)
from opendecision.schema import as_decision


def test_bool_uses_yes_no():
    assert [c.id for c in Bool().options] == ["yes", "no"]
    assert [c.id for c in Bool("approve", "reject").options] == ["approve", "reject"]


def test_multilabel_threshold():
    assert MultiLabel(["a", "b"], threshold=0.3).kind is DecisionKind.MULTI_LABEL
    with pytest.raises(SchemaError):
        MultiLabel(["a", "b"], threshold=1.0)


def test_coercion_and_validation():
    assert as_decision("pick", ["x", "y"]).kind is DecisionKind.CHOICE
    assert as_decision("ok", Bool()).name == "ok"
    assert [c.id for c in Score(1, 3).options] == ["1", "2", "3"]
    assert Candidate("a", "alpha", "first letter").text() == "alpha :: first letter"
    with pytest.raises(SchemaError):
        Choice(["x", "x"])
    with pytest.raises(SchemaError):
        Decision("", Choice(["x"]))
