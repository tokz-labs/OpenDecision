"""Cross-encoder: one joint ``[CLS] state [SEP] option [SEP]`` sequence per option."""

from __future__ import annotations

import torch
from torch import Tensor, nn

from .common import ModelConfig, ModelOutput, build_backbone, group_policy_logits


class CrossEncoder(nn.Module):
    """Scores each option from the [CLS] vector of its joint sequence.

    The option keeps its first ``cross_candidate_tokens`` tokens; when the joint row would
    exceed the backbone's position limit, the state is trimmed from the left. Score and
    value heads read only that option's [CLS], so both are candidate-local; the policy is
    the softmax of the scores within each decision group.
    """

    def __init__(self, cfg: ModelConfig, backbone_config=None):
        super().__init__()
        self.cfg = cfg
        self.backbone = build_backbone(cfg.backbone, backbone_config)
        hidden = self.backbone.config.hidden_size
        self.score_head = nn.Linear(hidden, 1)
        self.value_head = nn.Linear(hidden, 1)
        self.max_pos = int(self.backbone.config.max_position_embeddings)
        self.cls_token_id = self.sep_token_id = -1
        self.pad_token_id = 0

    def set_special_ids(self, cls_id: int, sep_id: int, pad_id: int) -> None:
        self.cls_token_id = int(cls_id)
        self.sep_token_id = int(sep_id)
        self.pad_token_id = int(pad_id)

    def forward(
        self,
        state_ids: Tensor,
        state_mask: Tensor,
        cand_ids: Tensor,
        cand_mask: Tensor,
        cand_state_index: Tensor,
        group_sizes: Tensor,
    ) -> ModelOutput:
        device = state_ids.device
        n_rows = cand_ids.shape[0]
        cand_width = min(cand_ids.shape[1], self.cfg.cross_candidate_tokens)
        cand = cand_ids[:, :cand_width]
        cand_len = cand_mask[:, :cand_width].sum(dim=1, dtype=torch.long)

        state_rows = state_ids[cand_state_index]
        state_len = state_mask[cand_state_index].sum(dim=1, dtype=torch.long)
        kept_state_len = torch.minimum(state_len, self.max_pos - cand_len - 3)
        state_start = state_len - kept_state_len
        row_len = kept_state_len + cand_len + 2
        width = int(row_len.max().item())

        # Assemble all joint rows on-device: [CLS] state [SEP] option, then padding.
        pos = torch.arange(width, device=device).unsqueeze(0)
        ids = torch.full((n_rows, width), self.pad_token_id, dtype=torch.long, device=device)
        ids[:, 0] = self.cls_token_id
        in_state = (pos >= 1) & (pos < 1 + kept_state_len.unsqueeze(1))
        state_src = (state_start.unsqueeze(1) + (pos - 1).clamp(min=0)).clamp(
            max=state_rows.shape[1] - 1
        )
        ids = torch.where(in_state, torch.gather(state_rows, 1, state_src), ids)
        ids.scatter_(1, (1 + kept_state_len).unsqueeze(1), self.sep_token_id)
        cand_start = kept_state_len + 2
        in_cand = (pos >= cand_start.unsqueeze(1)) & (pos < (cand_start + cand_len).unsqueeze(1))
        cand_src = (pos - cand_start.unsqueeze(1)).clamp(min=0).clamp(max=cand_width - 1)
        ids = torch.where(in_cand, torch.gather(cand, 1, cand_src), ids)
        mask = pos < row_len.unsqueeze(1)

        # Sort rows by length so each chunk is padded only to its own longest row.
        order = torch.argsort(row_len, stable=True)
        ids, mask = ids[order], mask[order]
        chunk = self.cfg.row_chunk_size
        widths = torch.stack([part.max() for part in row_len[order].split(chunk)]).cpu().tolist()
        pooled = []
        for i, start in enumerate(range(0, n_rows, chunk)):
            w = widths[i]
            hidden = self.backbone(
                input_ids=ids[start : start + chunk, :w],
                attention_mask=mask[start : start + chunk, :w].long(),
            ).last_hidden_state
            pooled.append(hidden[:, 0].float())
        pooled = torch.cat(pooled, dim=0)[torch.argsort(order)]

        scores = self.score_head(pooled).squeeze(-1)
        policy_logits, valid = group_policy_logits(scores, group_sizes)
        return ModelOutput(policy_logits, valid, self.value_head(pooled).squeeze(-1))
