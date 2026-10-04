"""Tiny random DeBERTa-v2 models and a word-level tokenizer: tests never download weights."""

from __future__ import annotations

import pytest
import torch

transformers = pytest.importorskip("transformers")


def tiny_backbone_config(vocab_size: int = 64):
    from transformers import DebertaV2Config

    return DebertaV2Config(
        vocab_size=vocab_size,
        hidden_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        intermediate_size=64,
        max_position_embeddings=512,
        relative_attention=True,
        position_buckets=256,
        pos_att_type=["p2c", "c2p"],
        position_biased_input=False,
        type_vocab_size=0,
        norm_rel_ebd="layer_norm",
        share_att_key=True,
    )


def tiny_model(architecture: str, vocab_size: int = 64, **cfg):
    from opendecision.models import ModelConfig, build_model

    torch.manual_seed(0)
    model = build_model(
        ModelConfig(architecture=architecture, backbone="tiny", **cfg),
        backbone_config=tiny_backbone_config(vocab_size),
    )
    model.set_special_ids(1, 2, 0)
    return model.eval()


class WordTokenizer:
    """Minimal stand-in for a Hugging Face tokenizer: one id per lowercase word."""

    def __init__(self, size: int = 200):
        self.vocab = {"[PAD]": 0, "[CLS]": 1, "[SEP]": 2}
        self.size = size
        self.cls_token_id, self.sep_token_id, self.pad_token_id = 1, 2, 0

    def add_special_tokens(self, spec):
        for token in spec["additional_special_tokens"]:
            self.vocab.setdefault(token, len(self.vocab))

    def convert_tokens_to_ids(self, token):
        return self.vocab[token]

    def encode(self, text, add_special_tokens=False):
        ids = []
        for word in text.lower().split():
            if word not in self.vocab:
                self.vocab[word] = 5 + sum(map(ord, word)) % (self.size - 5)
            ids.append(self.vocab[word])
        return ids

    def __len__(self):
        return self.size
