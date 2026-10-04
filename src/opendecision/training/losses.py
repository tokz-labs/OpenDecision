"""Policy and outcome losses.

The policy is a softmax over the options of one decision, trained with soft cross-entropy
against the (temperature-sharpened) target distribution. The outcome head is trained with
per-option binary cross-entropy on rows whose outcome targets are real. The two never
share a normalization.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor

from ..models import ModelOutput, pad_groups


def policy_cross_entropy(logits: Tensor, valid: Tensor, targets: Tensor) -> Tensor:
    """Mean over groups of -sum_k t_k log softmax(logits)_k on valid slots."""
    masked = logits.masked_fill(~valid, torch.finfo(logits.dtype).min)
    logp = (logits - torch.logsumexp(masked, dim=-1, keepdim=True)).clamp_min(-100.0)
    loss = -(targets.to(logp.dtype) * logp).sum(dim=-1)
    return loss.mean()


def value_bce(logits: Tensor, targets: Tensor, valid: Tensor) -> Tensor:
    if valid.numel() == 0 or not bool(valid.any()):
        return logits.sum() * 0.0  # zero loss and gradient, graph-safe
    return F.binary_cross_entropy_with_logits(logits[valid].float(), targets[valid].float())


def decision_loss(
    output: ModelOutput,
    batch,
    *,
    lambda_value: float = 0.25,
    temperature: float = 0.5,
) -> tuple[Tensor, dict[str, Tensor]]:
    targets, _ = pad_groups(batch.policy_targets.unsqueeze(1), batch.group_sizes)
    targets = targets.squeeze(-1)
    targets = targets / targets.sum(dim=-1, keepdim=True).clamp_min(1e-8)
    if temperature != 1.0:
        # Sharpen (<1) soft targets so many-similar-label tasks keep a clear argmax.
        targets = targets.clamp_min(1e-8) ** (1.0 / temperature)
        targets = targets / targets.sum(dim=-1, keepdim=True).clamp_min(1e-8)
    lp = policy_cross_entropy(output.policy_logits, output.policy_valid, targets)
    lv = value_bce(output.value_logits, batch.value_targets, batch.value_valid)
    return lp + lambda_value * lv, {"policy": lp.detach(), "value": lv.detach()}
