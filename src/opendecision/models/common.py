"""Shared pieces of the two architectures: configuration, outputs and group padding."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from torch import Tensor

CROSS_ENCODER = "cross_encoder"
PACKED = "packed"
ARCHITECTURES = (CROSS_ENCODER, PACKED)


@dataclass
class ModelConfig:
    """Everything needed to rebuild a model and its input serialization."""

    architecture: str = CROSS_ENCODER
    backbone: str = "microsoft/deberta-v3-large"
    max_state_tokens: int = 512
    max_candidate_tokens: int = 48
    max_header_tokens: int = 20
    # Cross-encoder: option tokens kept per joint row, rows per backbone call.
    cross_candidate_tokens: int = 32
    row_chunk_size: int = 128
    # Packed encoder: options per pack, option tokens per pack, attention cells per call.
    pack_size: int = 16
    pack_candidate_tokens: int = 3072
    pack_cell_budget: int = 6_000_000

    def __post_init__(self) -> None:
        if self.architecture not in ARCHITECTURES:
            raise ValueError(f"architecture must be one of {ARCHITECTURES}")

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(asdict(self), indent=2) + "\n", encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> ModelConfig:
        return cls(**json.loads(Path(path).read_text(encoding="utf-8")))


@dataclass
class ModelOutput:
    policy_logits: Tensor  # (G, K) per decision group, padded slots = finfo.min
    policy_valid: Tensor  # (G, K) bool
    value_logits: Tensor  # (P,) one per candidate row, candidate-local


def pad_groups(z: Tensor, group_sizes: Tensor) -> tuple[Tensor, Tensor]:
    """Pad a flat (P, ...) tensor of contiguous groups into (G, K, ...) plus a validity mask."""
    sizes = group_sizes.tolist()
    kmax = max(sizes) if sizes else 0
    padded = z.new_zeros((len(sizes), kmax, *z.shape[1:]))
    valid = torch.zeros((len(sizes), kmax), dtype=torch.bool, device=z.device)
    offset = 0
    for i, n in enumerate(sizes):
        padded[i, :n] = z[offset : offset + n]
        valid[i, :n] = True
        offset += n
    return padded, valid


def group_policy_logits(scores: Tensor, group_sizes: Tensor) -> tuple[Tensor, Tensor]:
    """(P,) candidate scores -> (G, K) policy logits with padded slots at finfo.min."""
    padded, valid = pad_groups(scores.unsqueeze(1), group_sizes)
    logits = padded.squeeze(-1)
    neg = torch.finfo(logits.dtype).min
    return torch.where(valid, logits, torch.full_like(logits, neg)), valid


def build_backbone(backbone: str, backbone_config=None):
    """Pretrained encoder, or an uninitialized one from ``backbone_config`` when fine-tuned
    weights are loaded right after (avoids downloading the base checkpoint)."""
    from transformers import AutoModel

    if backbone_config is not None:
        return AutoModel.from_config(backbone_config).float()
    return AutoModel.from_pretrained(backbone).float()
