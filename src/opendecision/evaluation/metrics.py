"""Run a model on an evaluation set and score prediction files (ID-paired).

An evaluation set is a directory of ``{task}.jsonl`` files (one decision per row, same labels
in every row). Single-label tasks are scored with macro-F1; multi-label tasks with example
exact match, micro-F1 and macro-F1 over labels.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

SPLITS = ("ag_news", "clinc_oos", "imdb", "rotten_tomatoes", "xnli")


def _jsonl(path: Path) -> list[dict]:
    return [json.loads(x) for x in path.read_text(encoding="utf-8").split("\n") if x.strip()]


def _tasks(set_dir: str | Path) -> list[str]:
    return sorted(p.stem for p in Path(set_dir).glob("*.jsonl"))


def macro_f1(gold: list[str], pred: list[str | None]) -> float:
    """Macro-F1 (%) over the gold classes present; ``None`` predictions count as wrong."""
    gold_n, pred_n, tp = Counter(gold), Counter(p for p in pred if p is not None), Counter()
    for g, p in zip(gold, pred):
        if p == g:
            tp[g] += 1
    total = 0.0
    for c in gold_n:
        precision = tp[c] / pred_n[c] if pred_n[c] else 0.0
        recall = tp[c] / gold_n[c]
        total += 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return 100 * total / len(gold_n)


def multi_label_scores(gold: list[list[str]], pred: list[list[str] | None]) -> dict:
    """Exact match, micro-F1 and macro-F1 (%) over the labels that occur in gold."""
    pred = [p or [] for p in pred]
    labels = sorted({x for g in gold for x in g})
    tp = fp = fn = 0
    per_label = []
    for lab in labels:
        t = sum(lab in g and lab in p for g, p in zip(gold, pred))
        f_p = sum(lab not in g and lab in p for g, p in zip(gold, pred))
        f_n = sum(lab in g and lab not in p for g, p in zip(gold, pred))
        tp, fp, fn = tp + t, fp + f_p, fn + f_n
        per_label.append(2 * t / (2 * t + f_p + f_n) if t else 0.0)
    exact = sum(set(g) == set(p) for g, p in zip(gold, pred)) / len(gold)
    return {
        "exact_match": round(100 * exact, 2),
        "micro_f1": round(100 * 2 * tp / (2 * tp + fp + fn), 2),
        "macro_f1": round(100 * sum(per_label) / len(per_label), 2),
    }


def predict(
    model,
    set_dir: str | Path,
    out_dir: str | Path,
    *,
    limit: int | None = None,
    batch_size: int = 8,
) -> None:
    """Write ``{task}_preds.jsonl`` rows ``{"id", "pred", "gold", "status"}`` for every task."""
    from ..schema import Choice, Decision, MultiLabel

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for task in _tasks(set_dir):
        rows = _jsonl(Path(set_dir) / f"{task}.jsonl")[:limit]
        d = rows[0]["decision"]
        labels = [c["name"] for c in rows[0]["candidates"]]
        multi = rows[0].get("multi_label", False)
        spec = MultiLabel(labels) if multi else Choice(labels)
        decision = Decision(d["name"], spec, description=d["description"])
        results = model.decide(
            [r["state"] for r in rows], {d["name"]: decision}, batch_size=batch_size
        )
        with (out / f"{task}_preds.jsonl").open("w", encoding="utf-8") as fh:
            for r, res in zip(rows, results):
                ans = res[d["name"]]
                pred = ans.selected if multi else ans.best
                row = {"id": r["example_id"], "pred": pred, "gold": r["gold"], "status": "ok"}
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def score(pred_dir: str | Path, set_dir: str | Path) -> dict:
    """Per-task scores and the average single-label macro-F1.

    Checks that prediction IDs and gold labels match the evaluation set.
    """
    report, single = {}, []
    for task in _tasks(set_dir):
        rows = _jsonl(Path(set_dir) / f"{task}.jsonl")
        gold = {r["example_id"]: r["gold"] for r in rows}
        preds = {r["id"]: r for r in _jsonl(Path(pred_dir) / f"{task}_preds.jsonl")}
        if set(preds) != set(gold):
            raise ValueError(f"{task}: prediction ids do not match the evaluation set")
        ids = sorted(gold)
        if any(preds[i].get("gold", gold[i]) != gold[i] for i in ids):
            raise ValueError(f"{task}: gold labels differ from the evaluation set")
        pred = [preds[i]["pred"] if preds[i].get("status", "ok") == "ok" else None for i in ids]
        if rows[0].get("multi_label", False):
            report[task] = multi_label_scores([gold[i] for i in ids], pred)
        else:
            single.append(macro_f1([gold[i] for i in ids], pred))
            report[task] = round(single[-1], 2)
    if single:
        report["average"] = round(sum(single) / len(single), 2)
    return report
