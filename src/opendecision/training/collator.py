"""Examples -> padded tensors.

``P`` option rows are grouped contiguously into ``G`` decision groups; each option row
points at the state row it reads, so one encoded state serves all of its options.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

import torch
from torch import Tensor

from ..encoding import Encoder, header_text, option_text
from .examples import Example


@dataclass
class Batch:
    state_ids: Tensor
    state_mask: Tensor
    cand_ids: Tensor
    cand_mask: Tensor
    cand_state_index: Tensor  # (P,) state row of each option row
    group_sizes: Tensor  # (G,)
    gold_index: Tensor  # (G,) position of the gold option, -1 if absent
    policy_targets: Tensor  # (P,)
    value_targets: Tensor  # (P,)
    value_valid: Tensor  # (P,) bool

    def to(self, device) -> Batch:
        return Batch(**{k: v.to(device) for k, v in self.__dict__.items()})


def _pad(seqs: list[list[int]], pad_id: int) -> tuple[Tensor, Tensor]:
    width = max((len(s) for s in seqs), default=1)
    ids = torch.full((len(seqs), width), pad_id, dtype=torch.long)
    mask = torch.zeros((len(seqs), width), dtype=torch.bool)
    for i, s in enumerate(seqs):
        if s:
            ids[i, : len(s)] = torch.tensor(s, dtype=torch.long)
            mask[i, : len(s)] = True
    return ids, mask


class Collator:
    """Tokenizes examples once (:meth:`encode`) and assembles batches (:meth:`assemble`).

    With ``shuffle_options`` the option order of every record is permuted at assembly
    time, so the same cached record is seen in a fresh order each epoch.
    """

    def __init__(
        self,
        encoder: Encoder,
        *,
        max_options: int = 96,
        shuffle_options: bool = True,
        seed: int = 0,
    ):
        self.encoder = encoder
        self.max_options = max_options
        self.shuffle_options = shuffle_options
        self.rng = random.Random(seed)

    def encode(self, ex: Example) -> dict:
        header = header_text(ex.decision.get("name", "decision"), ex.decision.get("description"))
        rows = list(ex.candidates)[: self.max_options]
        value_ok = ex.has_value_targets
        return {
            "state": self.encoder.state(ex.state),
            "options": [
                self.encoder.option(header, option_text(c["name"], c.get("description")))
                for c in rows
            ],
            "ids": [str(c["id"]) for c in rows],
            "policy": [float(ex.policy.get(str(c["id"]), 0.0)) for c in rows],
            "values": [float(ex.values.get(str(c["id"]), 0.0)) for c in rows],
            "value_valid": [bool(c.get("value_observed", value_ok)) for c in rows],
            "gold": str(ex.gold) if ex.gold is not None else None,
        }

    def _permute(self, rec: dict) -> dict:
        n = len(rec["options"])
        if n < 2:
            return rec
        order = list(range(n))
        self.rng.shuffle(order)
        keyed = ("options", "ids", "policy", "values", "value_valid")
        return {**rec, **{k: [rec[k][i] for i in order] for k in keyed}}

    def assemble(self, records: list[dict]) -> Batch:
        if self.shuffle_options:
            records = [self._permute(r) for r in records]
        states, options, owner, sizes, gold_index = [], [], [], [], []
        policy, values, valid = [], [], []
        for i, r in enumerate(records):
            states.append(r["state"])
            options.extend(r["options"])
            owner.extend([i] * len(r["options"]))
            sizes.append(len(r["options"]))
            gold_index.append(r["ids"].index(r["gold"]) if r["gold"] in r["ids"] else -1)
            policy.extend(r["policy"])
            values.extend(r["values"])
            valid.extend(r["value_valid"])
        pad = self.encoder.pad_id
        state_ids, state_mask = _pad(states, pad)
        cand_ids, cand_mask = _pad(options, pad)
        return Batch(
            state_ids=state_ids,
            state_mask=state_mask,
            cand_ids=cand_ids,
            cand_mask=cand_mask,
            cand_state_index=torch.tensor(owner, dtype=torch.long),
            group_sizes=torch.tensor(sizes, dtype=torch.long),
            gold_index=torch.tensor(gold_index, dtype=torch.long),
            policy_targets=torch.tensor(policy, dtype=torch.float32),
            value_targets=torch.tensor(values, dtype=torch.float32),
            value_valid=torch.tensor(valid, dtype=torch.bool),
        )

    def __call__(self, examples: list[Example]) -> Batch:
        return self.assemble([self.encode(ex) for ex in examples])
