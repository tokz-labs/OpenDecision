# Fine-tuning on your own decisions

`train.jsonl` shows the simple format: one decision per line with the input `text`, a decision
`name`, an optional `question`, the `labels` to choose from and the `gold` answer (a list for
multi-label decisions). Twelve lines are only enough to check that the pipeline runs; real
fine-tuning needs a few hundred labeled decisions per schema.

```bash
python -m opendecision.cli.train \
    --train examples/finetune/train.jsonl \
    --out runs/example \
    --init Tokz-labs/OpenDecision-Large \
    --steps 50 --warmup-steps 5 --eval-every 25
```

The fine-tuned model is written to `runs/example/model`:

```python
from opendecision import OpenDecision

model = OpenDecision.from_pretrained("runs/example/model")
```

A GPU with 24 GB of memory is enough; training uses bf16 and activation checkpointing.
