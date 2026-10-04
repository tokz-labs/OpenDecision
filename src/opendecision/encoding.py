"""Input serialization shared by inference and training.

A state is ``[CLS] text [SEP]``; an option is ``[DEC] header [CAND] option-text``, where the
header is the decision name (plus `` :: description``) cut to ``max_header_tokens``. Both
are trimmed to their length limits keeping the final token, exactly as during training.
"""

from __future__ import annotations

from dataclasses import dataclass

DEC_TOKEN = "[DEC]"
CAND_TOKEN = "[CAND]"


def trim(ids: list[int], max_len: int) -> list[int]:
    """Keep the first ``max_len - 1`` tokens and the last one (preserves a closing [SEP])."""
    if len(ids) <= max_len:
        return ids
    return ids[: max_len - 1] + [ids[-1]] if max_len > 1 else ids[:max_len]


def header_text(name: str, description: str | None) -> str:
    return f"{name} :: {description}" if description else name


def option_text(name: str, description: str | None) -> str:
    return f"{name} :: {description}" if description else name


@dataclass
class Encoder:
    """Tokenizes states and options for one tokenizer and length configuration."""

    tokenizer: object
    max_state_tokens: int = 512
    max_candidate_tokens: int = 48
    max_header_tokens: int = 20

    def __post_init__(self) -> None:
        tok = self.tokenizer
        tok.add_special_tokens({"additional_special_tokens": [DEC_TOKEN, CAND_TOKEN]})
        self.dec_id = tok.convert_tokens_to_ids(DEC_TOKEN)
        self.cand_id = tok.convert_tokens_to_ids(CAND_TOKEN)
        self.cls_id = tok.cls_token_id
        self.sep_id = tok.sep_token_id
        self.pad_id = tok.pad_token_id if tok.pad_token_id is not None else 0

    def state(self, text: str) -> list[int]:
        ids = [self.cls_id, *self.tokenizer.encode(text, add_special_tokens=False), self.sep_id]
        return trim(ids, self.max_state_tokens)

    def option(self, header: str, text: str) -> list[int]:
        head = self.tokenizer.encode(header, add_special_tokens=False)[: self.max_header_tokens]
        body = self.tokenizer.encode(text, add_special_tokens=False)
        return trim([self.dec_id, *head, self.cand_id, *body], self.max_candidate_tokens)
