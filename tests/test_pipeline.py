"""decide() and the training loop end to end on a tiny random model."""

from __future__ import annotations

import pytest
from conftest import WordTokenizer, tiny_model

from opendecision import Bool, Choice, Decision, MultiLabel, OpenDecision, Score
from opendecision.encoding import Encoder, trim
from opendecision.training import TrainConfig, from_labels, train


def _engine(architecture: str) -> OpenDecision:
    tok = WordTokenizer()
    model = tiny_model(architecture, vocab_size=len(tok))
    return OpenDecision(model, tok, model.cfg, device="cpu")


def test_trim_keeps_last_token():
    assert trim([1, 2, 3, 4, 5], 3) == [1, 2, 5]
    assert trim([1, 2], 3) == [1, 2]


@pytest.mark.parametrize("architecture", ["cross_encoder", "packed"])
def test_decide_returns_every_kind(architecture):
    out = _engine(architecture).decide(
        ["the parcel arrived broken", "refund please"],
        {
            "queue": Decision(
                "queue", Choice(["billing", "shipping", "other"]), description="Which team?"
            ),
            "tags": MultiLabel(["damaged", "late", "refund"]),
            "urgent": Bool(),
            "severity": Score(1, 3),
        },
        batch_size=1,
    )
    assert len(out) == 2
    first = out[0]
    assert abs(sum(first["queue"].policy.values()) - 1) < 1e-5
    assert first["queue"].best in {"billing", "shipping", "other"}
    assert set(first["urgent"].policy) == {"yes", "no"}
    assert first["tags"].selected and set(first["tags"].selected) <= {"damaged", "late", "refund"}
    assert 1 <= first["severity"].score_expectation <= 3


def test_option_order_does_not_change_packed_answers():
    engine = _engine("packed")
    a = engine.decide("my card was charged twice", {"q": ["billing", "shipping", "account"]})["q"]
    b = engine.decide("my card was charged twice", {"q": ["account", "billing", "shipping"]})["q"]
    assert all(abs(a.policy[k] - b.policy[k]) < 1e-5 for k in a.policy)


@pytest.mark.parametrize("checkpointing", [False, True])
@pytest.mark.parametrize("architecture", ["cross_encoder", "packed"])
def test_training_steps_run(tmp_path, architecture, checkpointing):
    tok = WordTokenizer()
    model = tiny_model(architecture, vocab_size=len(tok))
    pairs = [("money back please", "refund"), ("where is my box", "shipping")] * 4
    examples = [
        from_labels(f"e{i}", text, "intent", ["refund", "shipping", "other"], gold)
        for i, (text, gold) in enumerate(pairs)
    ]
    cfg = TrainConfig(
        steps=2,
        warmup_steps=1,
        bf16=False,
        activation_checkpointing=checkpointing,
        log_every=1,
        eval_every=1,
    )
    summary = train(
        model.train(), Encoder(tok), examples, examples[:2], cfg, tmp_path, device="cpu"
    )
    assert summary["steps"] == 2
    assert 0.0 <= summary["eval"][-1]["val_top1"] <= 1.0
