"""Fine-tuning loop. Defaults are the recipe of the released models."""

from __future__ import annotations

import json
import math
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch

from ..encoding import Encoder
from .collator import Collator
from .examples import Example
from .losses import decision_loss


@dataclass
class TrainConfig:
    steps: int = 2000
    batch_size: int = 2  # decisions per micro-batch
    grad_accum: int = 4  # effective batch = batch_size * grad_accum decisions
    backbone_lr: float = 5e-6
    head_lr: float = 5e-5
    weight_decay: float = 0.01
    betas: tuple[float, float] = (0.9, 0.98)
    warmup_steps: int = 200
    min_lr_ratio: float = 0.1
    grad_clip: float = 1.0
    lambda_value: float = 0.25
    temperature: float = 0.5
    max_options: int = 96
    seed: int = 1701
    bf16: bool = True
    activation_checkpointing: bool = True
    log_every: int = 50
    eval_every: int = 500


def lr_factor(step: int, warmup: int, total: int, min_ratio: float) -> float:
    """Linear warmup, then cosine decay to ``min_ratio``."""
    if step < warmup:
        return (step + 1) / max(1, warmup)
    progress = min(1.0, (step - warmup) / max(1, total - warmup))
    return min_ratio + (1.0 - min_ratio) * 0.5 * (1.0 + math.cos(math.pi * progress))


class EpochLoader:
    """Infinite batches over pre-tokenized records; order reshuffled each epoch from ``seed + epoch``."""

    def __init__(self, records: list[dict], collator: Collator, batch_size: int, seed: int):
        self.records = records
        self.collator = collator
        self.batch_size = batch_size
        self.seed = seed

    def __iter__(self):
        epoch = 0
        while True:
            order = torch.randperm(
                len(self.records), generator=torch.Generator().manual_seed(self.seed + epoch)
            ).tolist()
            for i in range(0, len(order), self.batch_size):
                yield self.collator.assemble(
                    [self.records[j] for j in order[i : i + self.batch_size]]
                )
            epoch += 1


@torch.no_grad()
def policy_accuracy(
    model, records: list[dict], collator: Collator, device, batch_size: int, bf16: bool
) -> float:
    """Top-1 policy accuracy over records that have a gold option."""
    model.eval()
    hit = total = 0
    for i in range(0, len(records), batch_size):
        batch = collator.assemble(records[i : i + batch_size]).to(device)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=bf16):
            out = model(
                batch.state_ids,
                batch.state_mask,
                batch.cand_ids,
                batch.cand_mask,
                batch.cand_state_index,
                batch.group_sizes,
            )
        pred = out.policy_logits.float().argmax(dim=-1)
        known = batch.gold_index >= 0
        hit += int((pred[known] == batch.gold_index[known]).sum())
        total += int(known.sum())
    model.train()
    return hit / max(1, total)


def train(
    model,
    encoder: Encoder,
    train_examples: list[Example],
    val_examples: list[Example],
    cfg: TrainConfig,
    out_dir: str | Path,
    device: str = "cuda",
) -> dict:
    """Train ``model`` in place; writes ``metrics.jsonl`` and returns a summary."""
    device = torch.device(device)
    bf16 = cfg.bf16 and device.type == "cuda" and torch.cuda.is_bf16_supported()
    torch.manual_seed(cfg.seed)
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    log = (out / "metrics.jsonl").open("a", encoding="utf-8")

    model.to(device).train()
    if cfg.activation_checkpointing:
        model.backbone.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
    collator = Collator(encoder, max_options=cfg.max_options, shuffle_options=True, seed=cfg.seed)
    val_collator = Collator(encoder, max_options=cfg.max_options, shuffle_options=False)
    batches = iter(
        EpochLoader(
            [collator.encode(ex) for ex in train_examples], collator, cfg.batch_size, cfg.seed
        )
    )
    val_records = [val_collator.encode(ex) for ex in val_examples]

    backbone = [p for n, p in model.named_parameters() if n.startswith("backbone.")]
    heads = [p for n, p in model.named_parameters() if not n.startswith("backbone.")]
    optimizer = torch.optim.AdamW(
        [{"params": backbone, "lr": cfg.backbone_lr}, {"params": heads, "lr": cfg.head_lr}],
        weight_decay=cfg.weight_decay,
        betas=cfg.betas,
        fused=device.type == "cuda",
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda s: lr_factor(s, cfg.warmup_steps, cfg.steps, cfg.min_lr_ratio)
    )

    t0 = time.time()
    running = {"policy": 0.0, "value": 0.0, "n": 0}
    history = []
    for step in range(1, cfg.steps + 1):
        for _ in range(cfg.grad_accum):
            batch = next(batches).to(device)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=bf16):
                output = model(
                    batch.state_ids,
                    batch.state_mask,
                    batch.cand_ids,
                    batch.cand_mask,
                    batch.cand_state_index,
                    batch.group_sizes,
                )
                loss, parts = decision_loss(
                    output, batch, lambda_value=cfg.lambda_value, temperature=cfg.temperature
                )
            if not torch.isfinite(loss):
                raise FloatingPointError(f"non-finite loss at step {step}")
            (loss / cfg.grad_accum).backward()
            running["policy"] += float(parts["policy"])
            running["value"] += float(parts["value"])
            running["n"] += 1
        if cfg.grad_clip:
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)

        if step % cfg.log_every == 0 or step == cfg.steps:
            n = max(1, running["n"])
            rec = {
                "step": step,
                "policy_loss": running["policy"] / n,
                "value_loss": running["value"] / n,
                "lr": scheduler.get_last_lr()[0],
                "sec": round(time.time() - t0, 1),
            }
            running = {"policy": 0.0, "value": 0.0, "n": 0}
            print(json.dumps(rec), flush=True)
            log.write(json.dumps(rec) + "\n")
        if val_records and (step % cfg.eval_every == 0 or step == cfg.steps):
            acc = policy_accuracy(model, val_records, val_collator, device, cfg.batch_size, bf16)
            rec = {"step": step, "val_top1": acc}
            history.append(rec)
            print(json.dumps(rec), flush=True)
            log.write(json.dumps(rec) + "\n")
    log.close()
    summary = {
        "steps": cfg.steps,
        "seconds": round(time.time() - t0, 1),
        "eval": history,
        "config": asdict(cfg),
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary
