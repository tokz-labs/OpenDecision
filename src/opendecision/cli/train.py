"""Fine-tune an OpenDecision model on your own decisions (one GPU).

    python -m opendecision.cli.train --train my_decisions.jsonl --out runs/mine \
        --init Tokz-labs/OpenDecision-Large --steps 2000

``--train`` is JSONL in the simple format
``{"text", "name", "question"?, "labels", "gold"}`` (``gold`` may be a list for
multi-label decisions) or the full ``Example`` format (``opendecision.training.examples``).
Defaults reproduce the released recipe. The model is written to ``<out>/model`` and loads
with ``OpenDecision.from_pretrained``.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from ..engine import OpenDecision
from ..training import TrainConfig, read_examples, train


def main() -> None:
    d = TrainConfig()
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--train", required=True, nargs="+", help="one or more JSONL files")
    ap.add_argument("--val", default=None, help="validation JSONL (default: 2%% of --train)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--init", default="Tokz-labs/OpenDecision-Large", help="Hub id or local dir")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--steps", type=int, default=d.steps)
    ap.add_argument("--batch-size", type=int, default=d.batch_size)
    ap.add_argument("--grad-accum", type=int, default=d.grad_accum)
    ap.add_argument("--backbone-lr", type=float, default=d.backbone_lr)
    ap.add_argument("--head-lr", type=float, default=d.head_lr)
    ap.add_argument("--warmup-steps", type=int, default=d.warmup_steps)
    ap.add_argument("--lambda-value", type=float, default=d.lambda_value)
    ap.add_argument("--temperature", type=float, default=d.temperature)
    ap.add_argument("--seed", type=int, default=d.seed)
    ap.add_argument("--eval-every", type=int, default=d.eval_every)
    args = ap.parse_args()

    cfg = TrainConfig(
        steps=args.steps,
        batch_size=args.batch_size,
        grad_accum=args.grad_accum,
        backbone_lr=args.backbone_lr,
        head_lr=args.head_lr,
        warmup_steps=args.warmup_steps,
        lambda_value=args.lambda_value,
        temperature=args.temperature,
        seed=args.seed,
        eval_every=args.eval_every,
    )
    train_ex = [ex for path in args.train for ex in read_examples(path)]
    if args.val:
        val_ex = read_examples(args.val)
    else:
        random.Random(cfg.seed).shuffle(train_ex)
        k = max(1, len(train_ex) // 50)
        val_ex, train_ex = train_ex[:k], train_ex[k:]

    engine = OpenDecision.from_pretrained(args.init, device="cpu")
    summary = train(engine.model, engine.encoder, train_ex, val_ex, cfg, args.out, args.device)
    engine.model.float().cpu().eval()
    engine.save_pretrained(Path(args.out) / "model")
    print(json.dumps({"model": str(Path(args.out) / "model"), "eval": summary["eval"][-1:]}))


if __name__ == "__main__":
    main()
