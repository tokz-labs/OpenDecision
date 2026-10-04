"""The 5-task classification suite.

The classification datasets of the GLiNER2.5 zero-shot benchmark: AG News, CLINC150 (plus,
with out-of-scope), IMDb, Rotten Tomatoes and XNLI (English). ``build`` writes ``n``
seeded-shuffle test examples per dataset with one fixed question and the dataset's own
label names, so every system answers identical inputs. Gold labels come from the datasets.
"""

from __future__ import annotations

import json
import random
from collections.abc import Iterator
from pathlib import Path

MAX_STATE_CHARS = 4000

SPECS = {
    "ag_news": {
        "repo": "fancyzhx/ag_news",
        "labels": ["World", "Sports", "Business", "Sci/Tech"],
        "instruction": "Which topic does this news article belong to?",
        "text_field": "text",
    },
    "imdb": {
        "repo": "stanfordnlp/imdb",
        # HF ClassLabel order is [neg, pos] — index 0 = negative. Keep in sync or golds invert.
        "labels": ["negative", "positive"],
        "instruction": "Is this movie review positive or negative?",
        "text_field": "text",
    },
    "rotten_tomatoes": {
        "repo": "cornell-movie-review-data/rotten_tomatoes",
        "labels": ["negative", "positive"],  # HF order: [neg, pos]
        "instruction": "Is this movie review positive or negative?",
        "text_field": "text",
    },
    "xnli": {
        "repo": "facebook/xnli",
        "config": "en",
        "labels": ["entailment", "neutral", "contradiction"],  # HF label ids 0,1,2
        "instruction": "Does the premise entail the hypothesis, is it neutral, or a contradiction?",
        "pair": True,
    },
    "clinc_oos": {
        "repo": "clinc/clinc_oos",
        "config": "plus",
        "labels": None,  # from dataset features at load time
        "instruction": "Which user intent does this utterance express?",
        "text_field": "text",
        "label_field": "intent",
    },
}


def _load(name: str, seed: int = 0):
    from datasets import load_dataset

    spec = SPECS[name]
    if name in ("xnli", "clinc_oos"):
        ds = load_dataset(spec["repo"], spec["config"], split="test")
    else:
        ds = load_dataset(spec["repo"], split="test")
    # Seeded random sample: several test sets are label-ordered (IMDb is 100% one class
    # in its first 1000 rows), which would make any "first n" subset degenerate.
    ds = ds.shuffle(seed=seed)
    labels = spec["labels"]
    if labels is None:
        labels = list(ds.features[spec["label_field"]].names)
        if "oos" not in [lab.lower() for lab in labels]:
            labels = [*labels, "oos"]
    else:
        # Guard against silent gold inversion: our label order must match the dataset's
        # ClassLabel order (imdb/rotten_tomatoes are [neg, pos], which once bit us).
        feature = ds.features.get("label") or ds.features.get(spec.get("label_field", "label"))
        names = getattr(feature, "names", None)
        if names:
            for mine, theirs in zip(labels, names):
                if mine[:3].lower() != theirs[:3].lower():
                    raise ValueError(f"{name}: label order mismatch: ours={labels} dataset={names}")
    return ds, labels


def _rows(name: str, ds, labels) -> Iterator[dict]:
    spec = SPECS[name]
    for row in ds:
        if name == "xnli":
            state = f"Premise: {row['premise']}\nHypothesis: {row['hypothesis']}"
        else:
            state = str(row[spec["text_field"]])
        yield {
            "state": state[:MAX_STATE_CHARS],
            "gold": labels[int(row[spec.get("label_field", "label")])],
        }


def build(out_dir: str | Path, n: int | None = 1000, seed: int = 0) -> dict:
    """Write one JSONL per benchmark (first ``n`` seeded-shuffle test examples; ``None`` = all)."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    manifest: dict = {
        "n": n,
        "seed": seed,
        "note": "gold labels from public datasets; no LLM teacher",
    }
    for name, spec in SPECS.items():
        ds, labels = _load(name, seed)
        uniform = 1.0 / len(labels)
        rows = []
        for i, ex in enumerate(_rows(name, ds, labels)):
            if n is not None and i >= n:
                break
            rows.append(
                {
                    "example_id": f"{name}-{i:06d}",
                    "state": ex["state"],
                    "decision": {
                        "name": name,
                        "description": spec["instruction"],
                        "kind": "choice",
                    },
                    "candidates": [
                        {"id": label, "name": label, "description": None, "metadata": {}}
                        for label in labels
                    ],
                    # Uniform placeholders: these files are for zero-shot *evaluation*;
                    # every system reads only state/candidates/gold.
                    "policy": dict.fromkeys(labels, uniform),
                    "values": dict.fromkeys(labels, 0.5),
                    "gold": ex["gold"],
                    "provenance": {
                        "policy": "placeholder",
                        "value": "placeholder",
                        "gold": "human_label",
                    },
                    "meta": {"benchmark": name, "n_labels": len(labels)},
                }
            )
        random.Random(seed).shuffle(rows)
        with (out / f"{name}.jsonl").open("w", encoding="utf-8") as fh:
            for r in rows:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")
        manifest[name] = {"n": len(rows), "n_labels": len(labels), "labels": labels}
        print(f"{name}: {len(rows)} examples, {len(labels)} labels")
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest
