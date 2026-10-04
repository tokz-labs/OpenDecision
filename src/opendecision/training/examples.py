"""Training examples and their JSONL formats.

One :class:`Example` is one decision over one input: the options, a soft policy target over
them (sums to 1), a per-option outcome target in [0, 1], the gold option, and provenance.
Outcome targets are used only when ``provenance["value"]`` names a real source.

Two on-disk formats are accepted:

* full: the fields of :class:`Example` (what the generators write);
* simple: ``{"text", "name", "question"?, "labels", "gold", "multi_label"?}`` where ``gold``
  is a label or a list of labels.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

# Provenance values that mark outcome targets as placeholders, not supervision.
PLACEHOLDER_PROVENANCE = frozenset({"placeholder", "", None})


@dataclass
class Example:
    example_id: str
    state: str
    decision: dict  # {"name", "description", "kind"}
    candidates: tuple[dict, ...]  # each {"id", "name", "description"?}
    policy: dict[str, float]
    values: dict[str, float]
    gold: str
    provenance: dict[str, str] = field(default_factory=dict)
    meta: dict = field(default_factory=dict)

    @property
    def has_value_targets(self) -> bool:
        return self.provenance.get("value") not in PLACEHOLDER_PROVENANCE

    def to_json(self) -> str:
        return json.dumps({**asdict(self), "candidates": list(self.candidates)}, ensure_ascii=False)

    @classmethod
    def from_dict(cls, d: dict) -> Example:
        return cls(
            example_id=str(d["example_id"]),
            state=d["state"],
            decision=d["decision"],
            candidates=tuple(d["candidates"]),
            policy=d["policy"],
            values=d["values"],
            gold=d["gold"],
            provenance=d.get("provenance", {}),
            meta=d.get("meta", {}),
        )


def from_labels(
    example_id: str,
    text: str,
    name: str,
    labels: list[str],
    gold: str | list[str],
    *,
    question: str | None = None,
    provenance: str = "human_label",
    meta: dict | None = None,
) -> Example:
    """Example with a uniform policy over the gold set and 0/1 outcome targets."""
    gold_set = list(gold) if isinstance(gold, list) else [gold]
    missing = [g for g in gold_set if g not in labels]
    if missing:
        raise ValueError(f"{example_id}: gold {missing} not among labels")
    return Example(
        example_id=example_id,
        state=text,
        decision={"name": name, "description": question, "kind": "choice"},
        candidates=tuple({"id": lab, "name": lab, "description": None} for lab in labels),
        policy={lab: (1.0 / len(gold_set) if lab in gold_set else 0.0) for lab in labels},
        values={lab: (1.0 if lab in gold_set else 0.0) for lab in labels},
        gold=gold_set[0],
        provenance={"policy": provenance, "value": provenance, "gold": provenance},
        meta={"multi_label": len(gold_set) > 1, "gold_set": gold_set, **(meta or {})},
    )


def read_examples(path: str | Path) -> list[Example]:
    """Read a JSONL file in either the full or the simple format."""
    out = []
    with Path(path).open(encoding="utf-8") as fh:
        for i, line in enumerate(fh):
            if not line.strip():
                continue
            r = json.loads(line)
            if "state" in r and "candidates" in r:
                out.append(Example.from_dict(r))
            else:
                out.append(
                    from_labels(
                        str(r.get("id", f"line-{i}")),
                        r["text"],
                        r.get("name", "decision"),
                        list(r["labels"]),
                        r["gold"],
                        question=r.get("question"),
                    )
                )
    return out


def write_examples(examples: list[Example], path: str | Path) -> None:
    with Path(path).open("w", encoding="utf-8") as fh:
        for ex in examples:
            fh.write(ex.to_json() + "\n")
