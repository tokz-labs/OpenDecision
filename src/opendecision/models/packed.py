"""Packed candidate encoder: one forward pass per pack of up to ``pack_size`` options.

Layout of one packed sequence (DeBERTa-v2/v3 backbone)::

    [state tokens S] [CLS o_1 SEP] [CLS o_2 SEP] ... [CLS o_n SEP]

Attention (3D mask, per sequence): state tokens attend to the state only; option tokens
attend to the state and to their own segment only.

Relative positions: state token i sits at position i; token j of every segment sits at
position S + j. Every segment therefore sees exactly the geometry it would see as the only
option, independent of its slot and of its siblings, so outputs are invariant to option
order and pack composition. Score and value heads read each segment's [CLS]; both are
candidate-local and the policy is the group softmax of the scores.

Parameter names (backbone, score_head, value_head) match :class:`CrossEncoder`, so either
architecture can warm-start the other.
"""

from __future__ import annotations

import types

import torch
from torch import Tensor, nn

from .common import ModelConfig, ModelOutput, build_backbone, group_policy_logits


def _batched_disentangled_bias(
    self, query_layer, key_layer, relative_pos, rel_embeddings, scale_factor
):
    """DeBERTa-v2 disentangled attention bias with one relative-position matrix per sequence.

    Mirrors transformers' implementation (share_att_key, c2p + p2c) but indexes positions
    per batch element: ``relative_pos`` is (B, q, k) and ``query_layer`` is (B * H, q, d).
    """
    from transformers.models.deberta_v2.modeling_deberta_v2 import scaled_size_sqrt

    heads = self.num_attention_heads
    rel = relative_pos.to(device=query_layer.device, dtype=torch.long)
    if rel.dim() == 4:
        rel = rel.squeeze(1)
    b, q, k = rel.shape
    per = query_layer.size(0) // b  # heads per sequence
    att_span = self.pos_ebd_size
    rel_embeddings = rel_embeddings[0 : att_span * 2, :].unsqueeze(0)
    reps = query_layer.size(0) // heads
    pos_query = self.transpose_for_scores(self.query_proj(rel_embeddings), heads).repeat(reps, 1, 1)
    pos_key = self.transpose_for_scores(self.key_proj(rel_embeddings), heads).repeat(reps, 1, 1)
    score = 0
    # Index through zero-copy (b, per, q, k) expanded views instead of materializing one
    # int64 position matrix per head.
    if "c2p" in self.pos_att_type:
        scale = scaled_size_sqrt(pos_key, scale_factor)
        c2p = torch.bmm(query_layer, pos_key.transpose(-1, -2))
        c2p_pos = torch.clamp(rel + att_span, 0, att_span * 2 - 1)
        c2p = torch.gather(
            c2p.view(b, per, q, -1), dim=-1, index=c2p_pos.unsqueeze(1).expand(b, per, q, k)
        ).view(b * per, q, k)
        score += c2p / scale.to(dtype=c2p.dtype)
    if "p2c" in self.pos_att_type:
        scale = scaled_size_sqrt(pos_query, scale_factor)
        p2c_pos = torch.clamp(-rel + att_span, 0, att_span * 2 - 1)
        p2c = torch.bmm(key_layer, pos_query.transpose(-1, -2))
        # Same indexing as transformers: rows of p2c are keys; transpose afterwards.
        p2c = (
            torch.gather(
                p2c.view(b, per, k, -1), dim=-1, index=p2c_pos.unsqueeze(1).expand(b, per, q, k)
            )
            .view(b * per, q, k)
            .transpose(-1, -2)
        )
        score += p2c / scale.to(dtype=p2c.dtype)
    return score


def install_batched_relative_positions(backbone) -> None:
    """Bind the per-sequence attention bias to every attention module of one backbone."""
    for layer in backbone.encoder.layer:
        attn = layer.attention.self
        if not attn.share_att_key:
            raise ValueError("packed encoder requires share_att_key=True")
        attn.disentangled_attention_bias = types.MethodType(_batched_disentangled_bias, attn)


class PackedEncoder(nn.Module):
    def __init__(self, cfg: ModelConfig, backbone_config=None):
        super().__init__()
        self.cfg = cfg
        self.backbone = build_backbone(cfg.backbone, backbone_config)
        if getattr(self.backbone.config, "model_type", "") != "deberta-v2":
            raise ValueError("packed encoder supports DeBERTa-v2/v3 backbones only")
        install_batched_relative_positions(self.backbone)
        hidden = self.backbone.config.hidden_size
        self.score_head = nn.Linear(hidden, 1)
        self.value_head = nn.Linear(hidden, 1)
        self.cls_token_id = self.sep_token_id = -1
        self.pad_token_id = 0

    def set_special_ids(self, cls_id: int, sep_id: int, pad_id: int) -> None:
        self.cls_token_id = int(cls_id)
        self.sep_token_id = int(sep_id)
        self.pad_token_id = int(pad_id)

    def _bucket(self, rel: Tensor) -> Tensor:
        from transformers.models.deberta_v2.modeling_deberta_v2 import make_log_bucket_position

        enc = self.backbone.encoder
        if enc.position_buckets > 0 and enc.max_relative_positions > 0:
            return make_log_bucket_position(rel, enc.position_buckets, enc.max_relative_positions)
        return rel

    def _plan_packs(self, state_mask: Tensor, cand_mask: Tensor, cand_state_index: Tensor):
        """Split each state's contiguous option rows into packs bounded by size and tokens."""
        state_len = state_mask.sum(1).tolist()
        cand_len = cand_mask.sum(1).clamp(max=self.cfg.max_candidate_tokens).tolist()
        owner = cand_state_index.tolist()
        packs: list[tuple[int, list[int], int]] = []  # (state, option rows, sequence length)
        p, n = 0, len(owner)
        while p < n:
            s, q = owner[p], p
            while q < n and owner[q] == s:
                q += 1
            rows: list[int] = []
            tokens = 0
            for r in range(p, q):
                need = int(cand_len[r]) + 2
                if rows and (
                    len(rows) >= self.cfg.pack_size
                    or tokens + need > self.cfg.pack_candidate_tokens
                ):
                    packs.append((s, rows, int(state_len[s]) + tokens))
                    rows, tokens = [], 0
                rows.append(r)
                tokens += need
            if rows:
                packs.append((s, rows, int(state_len[s]) + tokens))
            p = q
        return packs, state_len, cand_len

    def _materialize(self, packs, state_ids, cand_ids, state_len, cand_len):
        """Token ids, token mask, 3D attention mask, bucketed relative positions, row slots."""
        device = state_ids.device
        width = max(length for _, _, length in packs)
        b = len(packs)
        ids = torch.full((b, width), self.pad_token_id, dtype=torch.long, device=device)
        seg = torch.full((b, width), -2, dtype=torch.long, device=device)  # -2 pad, -1 state
        pos = torch.zeros((b, width), dtype=torch.long, device=device)
        slots: dict[int, tuple[int, int]] = {}
        for i, (s, rows, _) in enumerate(packs):
            sl = int(state_len[s])
            ids[i, :sl] = state_ids[s, :sl]
            seg[i, :sl] = -1
            pos[i, :sl] = torch.arange(sl, device=device)
            off = sl
            for j, r in enumerate(rows):
                cl = int(cand_len[r])
                ids[i, off] = self.cls_token_id
                ids[i, off + 1 : off + 1 + cl] = cand_ids[r, :cl]
                ids[i, off + 1 + cl] = self.sep_token_id
                seg[i, off : off + cl + 2] = j
                pos[i, off : off + cl + 2] = sl + torch.arange(cl + 2, device=device)
                slots[r] = (i, off)
                off += cl + 2
        token_mask = seg > -2
        q_seg, k_seg = seg.unsqueeze(2), seg.unsqueeze(1)
        attn = (k_seg == -1) | ((q_seg >= 0) & (q_seg == k_seg))
        attn = attn & token_mask.unsqueeze(1) & token_mask.unsqueeze(2)
        rel = self._bucket(pos.unsqueeze(2) - pos.unsqueeze(1))
        return ids, token_mask, attn, rel, slots

    def forward(
        self,
        state_ids: Tensor,
        state_mask: Tensor,
        cand_ids: Tensor,
        cand_mask: Tensor,
        cand_state_index: Tensor,
        group_sizes: Tensor,
    ) -> ModelOutput:
        packs, state_len, cand_len = self._plan_packs(state_mask, cand_mask, cand_state_index)
        # Greedy chunks bounded by the attention-cell budget so long inputs never
        # materialize every pack's (L x L) masks at once.
        chunks, cur, cur_w = [], [], 0
        for pk in packs:
            w = max(cur_w, pk[2])
            if cur and (len(cur) + 1) * w * w > self.cfg.pack_cell_budget:
                chunks.append(cur)
                cur, w = [], pk[2]
            cur.append(pk)
            cur_w = w
        if cur:
            chunks.append(cur)
        pooled = None
        for chunk in chunks:
            ids, token_mask, attn, rel, slots = self._materialize(
                chunk, state_ids, cand_ids, state_len, cand_len
            )
            emb = self.backbone.embeddings(input_ids=ids, mask=token_mask.long())
            hidden = self.backbone.encoder(
                emb, attn.long(), output_hidden_states=False, relative_pos=rel, return_dict=True
            ).last_hidden_state
            rows = sorted(slots)
            idx = torch.tensor([slots[r][0] for r in rows], device=hidden.device)
            off = torch.tensor([slots[r][1] for r in rows], device=hidden.device)
            feats = hidden[idx, off].float()
            if pooled is None:
                pooled = feats.new_zeros((cand_state_index.shape[0], feats.shape[-1]))
            pooled[torch.tensor(rows, device=hidden.device)] = feats
        scores = self.score_head(pooled).squeeze(-1)
        policy_logits, valid = group_policy_logits(scores, group_sizes)
        return ModelOutput(policy_logits, valid, self.value_head(pooled).squeeze(-1))
