"""Evaluate a model on an evaluation set.

    python -m opendecision.cli.evaluate --model Tokz-labs/OpenDecision-Large --set suite
    python -m opendecision.cli.evaluate --model Tokz-labs/OpenDecision-Large --set suite-full
    python -m opendecision.cli.evaluate --model Tokz-labs/OpenDecision-Large --set heldout

Sets (built on first use under ``data/<set>``):

* ``suite``: AG News, CLINC150, IMDb, Rotten Tomatoes, XNLI; 1,000 seeded test examples each;
* ``suite-full``: the same five tasks, complete test splits;
* ``heldout``: label sets no training data targeted (TREC, Emotion, Subjectivity, GoEmotions),
  plus TREC and Emotion with renamed labels.

Writes ``{task}_preds.jsonl`` and prints the scores. ``--preds DIR`` scores existing predictions.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ..evaluation import build, heldout, predict, score

BUILDERS = {
    "suite": lambda out: build(out),
    "suite-full": lambda out: build(out, n=None),
    "heldout": heldout.build,
}


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--model", default=None, help="Hub repo id or local model dir")
    ap.add_argument("--preds", default=None, help="score an existing predictions directory")
    ap.add_argument("--set", default="suite", choices=sorted(BUILDERS))
    ap.add_argument("--data", default=None, help="evaluation set directory (default data/<set>)")
    ap.add_argument("--out", default=None, help="predictions directory (default runs/eval/<set>)")
    ap.add_argument("--limit", type=int, default=None, help="examples per task (default all)")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--batch-size", type=int, default=8)
    args = ap.parse_args()
    data = Path(args.data or f"data/{args.set}")
    if not any(data.glob("*.jsonl")):
        BUILDERS[args.set](data)
    pred_dir = args.preds
    if pred_dir is None:
        if args.model is None:
            ap.error("give --model or --preds")
        from ..engine import OpenDecision

        model = OpenDecision.from_pretrained(args.model, device=args.device)
        pred_dir = args.out or f"runs/eval/{args.set}"
        predict(model, data, pred_dir, limit=args.limit, batch_size=args.batch_size)
    if args.limit is None:
        print(json.dumps(score(pred_dir, data), indent=2))
    else:
        print(
            json.dumps({"predictions": pred_dir, "note": "partial run; score needs all examples"})
        )


if __name__ == "__main__":
    main()
