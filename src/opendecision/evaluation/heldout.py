"""Held-out-schema tasks: label sets that no OpenDecision training data targeted.

TREC question types, Emotion, Subjectivity and GoEmotions (multi-label, 28 labels), plus TREC and
Emotion with every label renamed to measure sensitivity to label wording. Complete test splits.
"""

from __future__ import annotations

import json
from pathlib import Path

TREC = ["abbreviation", "entity", "description", "person", "location", "number"]
TREC_ALT = ["acronym", "thing", "explanation", "human", "place", "numeric value"]
EMOTION_ALT = {
    "sadness": "sad",
    "joy": "happy",
    "love": "affectionate",
    "anger": "angry",
    "fear": "afraid",
    "surprise": "surprised",
}


def _row(eid, text, task, question, labels, gold, multi=False) -> dict:
    return {
        "example_id": eid,
        "state": text,
        "task": task,
        "decision": {"name": task, "description": question, "kind": "choice"},
        "candidates": [{"id": lab, "name": lab, "description": None} for lab in labels],
        "gold": gold,
        "multi_label": multi,
    }


def _rename(rows: list[dict], mapping: dict[str, str]) -> list[dict]:
    labels = [mapping[c["name"]] for c in rows[0]["candidates"]]
    return [
        dict(
            r,
            gold=mapping[r["gold"]],
            candidates=[{"id": x, "name": x, "description": None} for x in labels],
        )
        for r in rows
    ]


def _write(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"{path.stem}: {len(rows)} examples")


def build(out_dir: str | Path) -> None:
    from datasets import load_dataset

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    trec = load_dataset("CogComp/trec", split="test", revision="refs/convert/parquet")
    rows = [
        _row(
            f"trec-{i:05d}",
            r["text"],
            "question_type",
            "What kind of answer does this question ask for?",
            TREC,
            TREC[r["coarse_label"]],
        )
        for i, r in enumerate(trec)
    ]
    _write(out / "trec.jsonl", rows)
    _write(out / "trec_alt.jsonl", _rename(rows, dict(zip(TREC, TREC_ALT))))

    emo = load_dataset("dair-ai/emotion", "split", split="test")
    names = emo.features["label"].names
    rows = [
        _row(
            f"emotion-{i:05d}",
            r["text"],
            "emotion",
            "Which emotion does the text express?",
            names,
            names[r["label"]],
        )
        for i, r in enumerate(emo)
    ]
    _write(out / "emotion.jsonl", rows)
    _write(out / "emotion_alt.jsonl", _rename(rows, EMOTION_ALT))

    subj = load_dataset("SetFit/subj", split="test")
    _write(
        out / "subj.jsonl",
        [
            _row(
                f"subj-{i:05d}",
                r["text"],
                "subjectivity",
                "Is this sentence subjective or objective?",
                ["subjective", "objective"],
                r["label_text"],
            )
            for i, r in enumerate(subj)
        ],
    )

    go = load_dataset("google-research-datasets/go_emotions", "simplified", split="test")
    names = go.features["labels"].feature.names
    _write(
        out / "go_emotions.jsonl",
        [
            _row(
                f"goemo-{i:05d}",
                r["text"],
                "emotions",
                "Which emotions does this comment express?",
                names,
                sorted(names[j] for j in r["labels"]),
                multi=True,
            )
            for i, r in enumerate(go)
        ],
    )
