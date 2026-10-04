"""Packed encoder invariants: option order, pack size, candidate locality, exact attention bias."""

from __future__ import annotations

import torch
from conftest import tiny_model

CANDS = [[5, 20, 6, 30], [5, 20, 6, 31, 32], [5, 20, 6, 33], [5, 20, 6, 34, 35, 36]]


def _batch(cands):
    state = torch.tensor([[1, 10, 11, 12, 13, 2]])
    width = max(len(c) for c in cands)
    ids = torch.zeros((len(cands), width), dtype=torch.long)
    mask = torch.zeros_like(ids, dtype=torch.bool)
    for i, c in enumerate(cands):
        ids[i, : len(c)] = torch.tensor(c)
        mask[i, : len(c)] = True
    owner = torch.zeros(len(cands), dtype=torch.long)
    return (
        state,
        torch.ones_like(state, dtype=torch.bool),
        ids,
        mask,
        owner,
        torch.tensor([len(cands)]),
    )


@torch.no_grad()
def test_pack_size_and_permutation_invariance():
    full = tiny_model("packed", pack_size=16)(*_batch(CANDS))
    single = tiny_model("packed", pack_size=1)(*_batch(CANDS))
    assert torch.allclose(full.value_logits, single.value_logits, atol=1e-5)
    perm = [2, 0, 3, 1]
    permuted = tiny_model("packed")(*_batch([CANDS[i] for i in perm]))
    assert torch.allclose(permuted.value_logits, full.value_logits[perm], atol=1e-5)


@torch.no_grad()
def test_value_is_candidate_local():
    for architecture in ("packed", "cross_encoder"):
        model = tiny_model(architecture)
        full, fewer = model(*_batch(CANDS)), model(*_batch(CANDS[:2]))
        assert torch.allclose(full.value_logits[:2], fewer.value_logits, atol=1e-5)


@torch.no_grad()
def test_batched_bias_matches_stock_deberta():
    """On a plain 0..L-1 layout with a full mask the per-sequence bias equals stock DeBERTa."""
    from transformers import DebertaV2Model

    model = tiny_model("packed")
    stock = DebertaV2Model(model.backbone.config).eval()
    stock.load_state_dict(model.backbone.state_dict())
    ids = torch.tensor([[1, 10, 11, 12, 2], [1, 13, 14, 15, 2]])
    rel = model._bucket(torch.arange(5)[:, None] - torch.arange(5)[None, :]).expand(2, 5, 5)
    emb = model.backbone.embeddings(input_ids=ids, mask=torch.ones_like(ids))
    out = model.backbone.encoder(
        emb,
        torch.ones((2, 5, 5), dtype=torch.long),
        output_hidden_states=False,
        relative_pos=rel,
        return_dict=True,
    ).last_hidden_state
    assert torch.allclose(out, stock(input_ids=ids).last_hidden_state, atol=1e-5)


@torch.no_grad()
def test_chunked_packs_match_single_pass():
    whole = tiny_model("packed", pack_size=1)(*_batch(CANDS))
    chunked = tiny_model("packed", pack_size=1, pack_cell_budget=1)(*_batch(CANDS))
    assert torch.allclose(whole.value_logits, chunked.value_logits, atol=1e-5)
    assert torch.allclose(whole.policy_logits, chunked.policy_logits, atol=1e-5)
