"""Decision schemas: what you ask the model.

A :class:`Decision` is a named question with an optional description and a spec:
:class:`Choice` (exactly one option), :class:`Bool` (yes/no), :class:`MultiLabel` (any
subset) or :class:`Score` (a point on an integer scale). Options are plain strings or
:class:`Candidate` objects with an optional description.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from enum import Enum

from .encoding import option_text

MAX_TEXT_LEN = 4096
MAX_OPTIONS = 4096


class SchemaError(ValueError):
    """Raised when a decision schema is malformed."""


class DecisionKind(str, Enum):
    CHOICE = "choice"
    BOOL = "bool"
    MULTI_LABEL = "multi_label"
    SCORE = "score"


def _check_text(value: str, what: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise SchemaError(f"{what} must be a non-empty string")
    if len(value) > MAX_TEXT_LEN:
        raise SchemaError(f"{what} exceeds {MAX_TEXT_LEN} characters")


@dataclass(frozen=True)
class Candidate:
    """One option. ``id`` keys the results; ``name`` (and ``description``) is what the model reads."""

    id: str
    name: str
    description: str | None = None

    def __post_init__(self) -> None:
        _check_text(self.id, "candidate id")
        _check_text(self.name, "candidate name")
        if self.description is not None:
            _check_text(self.description, "candidate description")

    def text(self) -> str:
        return option_text(self.name, self.description)


def _options(values: Iterable[Candidate | str]) -> tuple[Candidate, ...]:
    out = []
    for v in values:
        if isinstance(v, str):
            v = Candidate(id=v, name=v)
        elif not isinstance(v, Candidate):
            raise SchemaError(f"options must be str or Candidate, got {type(v).__name__}")
        out.append(v)
    if not out:
        raise SchemaError("a decision needs at least one option")
    if len(out) > MAX_OPTIONS:
        raise SchemaError(f"at most {MAX_OPTIONS} options per decision")
    if len({c.id for c in out}) != len(out) or len({c.name for c in out}) != len(out):
        raise SchemaError("option ids and names must be unique")
    return tuple(out)


@dataclass(frozen=True, init=False)
class Choice:
    """Pick exactly one option."""

    options: tuple[Candidate, ...]
    kind: DecisionKind = field(default=DecisionKind.CHOICE)

    def __init__(self, options: Iterable[Candidate | str]):
        object.__setattr__(self, "options", _options(options))
        object.__setattr__(self, "kind", DecisionKind.CHOICE)


@dataclass(frozen=True, init=False)
class MultiLabel:
    """Pick every option whose candidate-local probability is at least ``threshold``
    (the policy argmax if none passes)."""

    options: tuple[Candidate, ...]
    threshold: float = 0.5
    kind: DecisionKind = field(default=DecisionKind.MULTI_LABEL)

    def __init__(self, options: Iterable[Candidate | str], threshold: float = 0.5):
        if not 0.0 < threshold < 1.0:
            raise SchemaError("threshold must be in (0, 1)")
        object.__setattr__(self, "options", _options(options))
        object.__setattr__(self, "threshold", float(threshold))
        object.__setattr__(self, "kind", DecisionKind.MULTI_LABEL)


@dataclass(frozen=True)
class Bool:
    """Yes or no. The released models were trained with the labels ``yes``/``no``."""

    true_label: str = "yes"
    false_label: str = "no"
    kind: DecisionKind = field(default=DecisionKind.BOOL, init=False)

    def __post_init__(self) -> None:
        if self.true_label == self.false_label:
            raise SchemaError("bool labels must differ")

    @property
    def options(self) -> tuple[Candidate, ...]:
        return _options([self.true_label, self.false_label])


@dataclass(frozen=True)
class Score:
    """A point on the integer scale ``low..high`` (step ``step``)."""

    low: int = 0
    high: int = 5
    step: int = 1
    kind: DecisionKind = field(default=DecisionKind.SCORE, init=False)

    def __post_init__(self) -> None:
        if self.step <= 0 or self.high < self.low:
            raise SchemaError("score needs step > 0 and high >= low")

    @property
    def values(self) -> tuple[int, ...]:
        return tuple(range(self.low, self.high + 1, self.step))

    @property
    def options(self) -> tuple[Candidate, ...]:
        return _options(str(v) for v in self.values)


Spec = Choice | Bool | MultiLabel | Score


@dataclass(frozen=True)
class Decision:
    """A named question. The name (and description) is shown to the model with every option."""

    name: str
    spec: Spec
    description: str | None = None

    def __post_init__(self) -> None:
        _check_text(self.name, "decision name")
        if self.description is not None:
            _check_text(self.description, "decision description")
        if not isinstance(self.spec, (Choice, Bool, MultiLabel, Score)):
            raise SchemaError(
                f"spec must be Choice, Bool, MultiLabel or Score, got {type(self.spec).__name__}"
            )

    @property
    def kind(self) -> DecisionKind:
        return self.spec.kind

    @property
    def options(self) -> tuple[Candidate, ...]:
        return self.spec.options


def as_decision(name: str, value: Decision | Spec | Sequence[str]) -> Decision:
    """Accept a Decision, a bare spec, or a list of option strings (a Choice)."""
    if isinstance(value, Decision):
        return value
    if isinstance(value, (Choice, Bool, MultiLabel, Score)):
        return Decision(name=name, spec=value)
    if isinstance(value, Sequence) and not isinstance(value, str):
        return Decision(name=name, spec=Choice(value))
    raise SchemaError(f"decision {name!r}: expected Decision, spec or list of options")
