"""``OpenDecision``: load a model and answer decisions."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path

import torch
from torch import Tensor

from .encoding import Encoder, header_text
from .models import ModelConfig, build_model
from .schema import Candidate, Decision, DecisionKind, MultiLabel, Score, as_decision


@dataclass
class DecisionResult:
    """Answer to one decision.

    ``policy`` is a distribution over the option set (sums to 1; use it to choose).
    ``value`` holds a candidate-local probability per option that does not depend on the
    other options (use it for multi-label decisions and thresholds).
    """

    name: str
    kind: DecisionKind
    policy: dict[str, float]
    value: dict[str, float]
    selected: list[str] = field(default_factory=list)
    score_expectation: float | None = None

    @property
    def best(self) -> str:
        return max(self.policy, key=self.policy.get)

    def ranked(self) -> list[tuple[str, float]]:
        return sorted(self.policy.items(), key=lambda kv: kv[1], reverse=True)


class DecisionBatch(dict):
    """Results for one input, keyed by decision name."""


def _pad(seqs: list[list[int]], pad: int, device) -> tuple[Tensor, Tensor]:
    width = max((len(s) for s in seqs), default=1)
    ids = torch.full((len(seqs), width), pad, dtype=torch.long)
    mask = torch.zeros((len(seqs), width), dtype=torch.bool)
    for i, s in enumerate(seqs):
        ids[i, : len(s)] = torch.tensor(s, dtype=torch.long)
        mask[i, : len(s)] = True
    return ids.to(device), mask.to(device)


class OpenDecision:
    """A decision model plus its tokenizer.

    >>> model = OpenDecision.from_pretrained("Tokz-labs/OpenDecision-Large")
    >>> model.decide("Refund my order", {"intent": ["refund", "shipping", "other"]})["intent"].best
    """

    def __init__(self, model, tokenizer, config: ModelConfig, device: str = "auto"):
        self.config = config
        self.encoder = Encoder(
            tokenizer,
            max_state_tokens=config.max_state_tokens,
            max_candidate_tokens=config.max_candidate_tokens,
            max_header_tokens=config.max_header_tokens,
        )
        self.tokenizer = tokenizer
        model.set_special_ids(self.encoder.cls_id, self.encoder.sep_id, self.encoder.pad_id)
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self.model = model.to(self.device).eval()
        # bf16 autocast on CUDA (2.6-2.9x faster than fp32 on an L4); set "fp32" to disable.
        self.precision = "bf16"

    # ------------------------------------------------------------------ persistence

    @classmethod
    def from_pretrained(
        cls, path: str | Path, device: str = "auto", revision: str | None = None
    ) -> OpenDecision:
        """Load from a local directory or a Hugging Face Hub repo id."""
        from safetensors.torch import load_file
        from transformers import AutoConfig, AutoTokenizer

        root = Path(path)
        if not (root / "config.json").exists():
            from huggingface_hub import snapshot_download

            root = Path(snapshot_download(repo_id=str(path), revision=revision))
        config = ModelConfig.load(root / "config.json")
        tokenizer = AutoTokenizer.from_pretrained(str(root / "tokenizer"))
        backbone_config = AutoConfig.from_pretrained(str(root / "backbone_config.json"))
        model = build_model(config, backbone_config=backbone_config)
        model.backbone.resize_token_embeddings(len(tokenizer))
        model.load_state_dict(load_file(str(root / "model.safetensors"), device="cpu"))
        return cls(model, tokenizer, config, device=device)

    def save_pretrained(self, path: str | Path) -> None:
        from safetensors.torch import save_file

        out = Path(path)
        out.mkdir(parents=True, exist_ok=True)
        self.config.save(out / "config.json")
        self.model.backbone.config.to_json_file(str(out / "backbone_config.json"))
        self.tokenizer.save_pretrained(str(out / "tokenizer"))
        state = {k: v.detach().contiguous().cpu() for k, v in self.model.state_dict().items()}
        save_file(state, str(out / "model.safetensors"))

    # ------------------------------------------------------------------ inference

    @torch.no_grad()
    def decide(
        self,
        inputs: str | object | list,
        decisions: Mapping[str, object],
        *,
        batch_size: int = 8,
    ) -> DecisionBatch | list[DecisionBatch]:
        """Answer every decision for one input (or each of a list of inputs).

        ``decisions`` maps a name to a :class:`Decision`, a spec (``Choice``, ``Bool``,
        ``MultiLabel``, ``Score``) or a list of option strings. Non-string inputs are
        serialized as JSON.
        """
        if not decisions:
            raise ValueError("at least one decision is required")
        specs = [as_decision(name, value) for name, value in decisions.items()]
        many = isinstance(inputs, list)
        items = inputs if many else [inputs]
        results: list[DecisionBatch] = []
        for start in range(0, len(items), max(1, batch_size)):
            results.extend(self._decide_batch(items[start : start + batch_size], specs))
        return results if many else results[0]

    def _decide_batch(self, items: list, specs: list[Decision]) -> list[DecisionBatch]:
        enc = self.encoder
        states = [
            enc.state(x if isinstance(x, str) else json.dumps(x, ensure_ascii=False, default=str))
            for x in items
        ]
        option_rows: list[list[int]] = []
        owner: list[int] = []
        sizes: list[int] = []
        layout: list[tuple[int, Decision, tuple[Candidate, ...]]] = []
        for si in range(len(states)):
            for dec in specs:
                header = header_text(dec.name, dec.description)
                for c in dec.options:
                    option_rows.append(enc.option(header, c.text()))
                    owner.append(si)
                sizes.append(len(dec.options))
                layout.append((si, dec, dec.options))

        state_ids, state_mask = _pad(states, enc.pad_id, self.device)
        cand_ids, cand_mask = _pad(option_rows, enc.pad_id, self.device)
        use_bf16 = self.device.type == "cuda" and self.precision == "bf16"
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_bf16):
            out = self.model(
                state_ids,
                state_mask,
                cand_ids,
                cand_mask,
                torch.tensor(owner, dtype=torch.long, device=self.device),
                torch.tensor(sizes, dtype=torch.long, device=self.device),
            )
        probs = torch.softmax(out.policy_logits.float(), dim=-1)
        values = torch.sigmoid(out.value_logits.float())

        batches = [DecisionBatch() for _ in states]
        row = 0
        for g, (si, dec, options) in enumerate(layout):
            n = len(options)
            p = probs[g, :n].tolist()
            v = values[row : row + n].tolist()
            row += n
            policy = {c.id: float(x) for c, x in zip(options, p)}
            value = {c.id: float(x) for c, x in zip(options, v)}
            best = max(policy, key=policy.get)
            selected = [best]
            if isinstance(dec.spec, MultiLabel):
                selected = [c.id for c, x in zip(options, v) if x >= dec.spec.threshold] or [best]
            expectation = None
            if isinstance(dec.spec, Score):
                expectation = float(sum(x * s for x, s in zip(p, dec.spec.values)))
            batches[si][dec.name] = DecisionResult(
                dec.name, dec.kind, policy, value, selected, expectation
            )
        return batches
