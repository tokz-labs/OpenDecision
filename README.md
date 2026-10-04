# OpenDecision

[![CI](https://github.com/tokz-labs/OpenDecision/actions/workflows/ci.yml/badge.svg)](https://github.com/tokz-labs/OpenDecision/actions/workflows/ci.yml)
[![License: Apache-2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![Models on Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97-Tokz--labs-yellow)](https://huggingface.co/Tokz-labs)

**Open encoder models for fast decisions over options you define at call time.**

📄 Technical report: [*OpenDecision: Fast Schema-Conditioned Decisions with Packed Candidate Encoding*](https://github.com/tokz-labs/OpenDecision/releases/latest/download/OpenDecision.pdf) (PDF; arXiv version coming soon)

Route a ticket, pick an intent, tag a document, answer yes/no — several questions about the
same text in one call, run locally, with a probability for every option instead of
generated text.

- **Any label set at inference time** — no fixed vocabulary, no retraining.
- **Several decisions per call** — single choice, yes/no, multi-label and ordinal scores.
- **Order-invariant** — the packed model's answers do not depend on the order of the options.
- **Fast** — 3.5× less GPU time than 4B LLM-based decision models, with a higher average on five standard benchmarks.
- **Apache-2.0** code and weights.

## Installation

```bash
pip install "opendecision @ git+https://github.com/tokz-labs/OpenDecision"
```

Python 3.10+, PyTorch 2.2+, Transformers 4.48+.

## Quick start

```python
from opendecision import OpenDecision, Choice, Bool, MultiLabel, Decision

model = OpenDecision.from_pretrained("Tokz-labs/OpenDecision-Large")

ticket = "I was charged twice for my May invoice and the app keeps logging me out."
out = model.decide(ticket, {
    "queue": Decision("queue", Choice(["billing", "technical", "account", "sales"]),
                      description="Which team should handle this ticket?"),
    "tags": MultiLabel(["double_charge", "login_issue", "refund_request", "outage"]),
    "urgent": Bool(),
})

out["queue"].best        # 'billing'
out["tags"].selected     # ['double_charge', 'login_issue']
out["urgent"].policy     # {'yes': 0.096, 'no': 0.904}
```

Pass a list of texts to `decide()` to get one result per text (`batch_size` controls how
many are encoded together).

More examples: [`examples/quickstart.py`](examples/quickstart.py),
[`examples/batch_routing.py`](examples/batch_routing.py) (confidence thresholds) and
[`examples/finetune/`](examples/finetune) (training on your own labels).

## Models

| Model | Architecture | Parameters | Use it for |
|---|---|---|---|
| [`Tokz-labs/OpenDecision-Large`](https://huggingface.co/Tokz-labs/OpenDecision-Large) | cross-encoder: one pass per option | 433.9M | best accuracy |
| [`Tokz-labs/OpenDecision-Large-Packed`](https://huggingface.co/Tokz-labs/OpenDecision-Large-Packed) | packed encoder: up to 16 options per pass | 433.9M | lowest cost on GPU, long inputs |
| [`Tokz-labs/OpenDecision-Small`](https://huggingface.co/Tokz-labs/OpenDecision-Small) | cross-encoder | 70.6M | CPU, small footprint |

Large models use DeBERTa-v3-large, Small uses DeBERTa-v3-xsmall. English, inputs up to 512 tokens.

## Decision types

| Spec | Answers | Read |
|---|---|---|
| `Choice(options)` | exactly one option | `.best`, `.ranked()`, `.policy` |
| `Bool()` | `yes` or `no` | `.best`, `.policy` |
| `MultiLabel(options, threshold=0.5)` | any subset | `.selected`, `.value` |
| `Score(low, high, step=1)` | a point on a scale | `.best`, `.score_expectation` |

`Decision(name, spec, description=...)` adds a natural-language question; options can be
`Candidate(id, name, description)` to add per-option descriptions.

Every result has two views. `.policy` is a distribution **over the option set** (sums to
1) — use it to choose. `.value[option]` is a **per-option** probability that does not
depend on which other options are present — use it for multi-label decisions and
thresholds.

## Results

All systems below are open models, run through their own packages on the same inputs, label names and
questions (GLiNER2.5 with its best interface per dataset). Full details are in the technical report.

**Classification suite** (complete test sets, macro-F1):

| Model | Size | AG News | CLINC150 | IMDb | Rotten Tomatoes | XNLI | Average |
|---|---|---|---|---|---|---|---|
| **OpenDecision-Large** | 434M | 87.37 | 76.38 | 94.84 | **92.03** | **88.93**\* | **87.91** |
| **OpenDecision-Large-Packed** | 434M | 86.70 | 75.95 | 94.59 | 89.96 | 88.97\* | 87.23 |
| SemIf (Qwen3.5-4B) | 4B | 87.28 | 78.22 | **95.27** | 88.46 | 82.36 | 86.32 |
| JevK5 v0.3 | 4B | 85.45 | **79.86** | 95.06 | 87.60 | 83.40 | 86.27 |
| Laya | 421M | **92.42** | 47.57 | 92.16 | 87.99 | 87.22 | 81.47 |
| GLiFormer large-v1 | 576M | 82.88 | 59.49 | 94.19 | 84.06 | 82.03 | 80.53 |
| **OpenDecision-Small** | 71M | 82.32 | 63.18 | 88.60 | 83.58 | 78.61\* | 79.26 |
| GLiNER2.5-small | 74M | 72.21 | 56.72 | 87.56 | 77.20 | 78.24 | 74.39 |
| GLiNER2.5-Decide | 340M | 71.37 | 66.19 | 90.02 | 85.83 | 47.65 | 72.21 |

\* NLI data (SNLI, MultiNLI, WANLI) is in training, so XNLI is not zero-shot. Without XNLI,
OpenDecision-Large still has the highest average (87.66 vs 87.31 for
SemIf). No text or training split of AG News, CLINC150, IMDb or Rotten Tomatoes is used, but the suite
guided our choice of training data.

**New label sets.** Tasks whose label sets no training data targeted (macro-F1; GoEmotions:
multi-label exact match). Here the 4B models and Laya generalize better:

| Model | TREC | Emotion | Subjectivity | Avg of 3 | GoEmotions |
|---|---|---|---|---|---|
| JevK5 v0.3 (4B) | **88.00** | 51.66 | **76.54** | **72.07** | 3.85 |
| SemIf (4B) | 83.36 | 47.99 | 61.53 | 64.29 | 0.87 |
| Laya | 83.00 | 48.54 | 54.79 | 62.11 | 10.26 |
| OpenDecision-Large | 65.13 | 49.15 | 56.71 | 57.00 | 10.83 |
| GLiNER2.5-Decide | 59.09 | **53.33** | 55.93 | 56.12 | 30.85 |
| OpenDecision-Large-Packed | 62.95 | 48.30 | 55.68 | 55.64 | 13.30 |
| GLiFormer large-v1 | 57.97 | 41.71 | 50.50 | 50.06 | **34.51** |
| OpenDecision-Small | 45.96 | 46.36 | 40.15 | 44.16 | 17.49 |
| GLiNER2.5-small | 42.15 | 42.50 | 46.86 | 43.84 | 27.14 |

OpenDecision is strongest on decision types covered by its training data (routing, intents, topics,
sentiment, triage, NLI). For a new kind of decision, fine-tune it on a few hundred labeled examples
(see below).

**Speed.** GPU-seconds per 1,000 decisions on the complete suite test sets (one NVIDIA L4, each system
run its standard way), the corresponding cost at L4 list price, and CPU latency for one short text
(AMD Ryzen 5 7535HS with Radeon Graphics, fp32, median):

| | GPU-s / 1k decisions | $ / 1k decisions | CPU, 5 labels | CPU, 50 labels |
|---|---|---|---|---|
| OpenDecision-Small | **17.5** | 0.0039 | — | — |
| Laya | 43.5 | 0.0097 | — | — |
| OpenDecision-Large-Packed | 69.1 | 0.0153 | 438 ms | 1937 ms |
| OpenDecision-Large | 77.4 | 0.0172 | 735 ms | 4625 ms |
| GLiFormer large-v1 | 109.5 | 0.0243 | — | — |
| JevK5 v0.3 (4B) | 267.4 | 0.0594 | — | — |
| SemIf (4B) | 275.0 | 0.0611 | — | — |
| GLiNER2.5-Decide | — | — | 455 ms | 924 ms |

The packed model is fastest when questions have up to ~16 options (34.8 GPU-s per 1,000
without CLINC150's 151-option questions); with many options use the cross-encoder. On CPU,
GLiNER2.5-Decide scores all labels in one pass and is faster at high label counts.

## Training your own model

```bash
python -m opendecision.cli.train --train my_decisions.jsonl --out runs/mine \
    --init Tokz-labs/OpenDecision-Large --steps 2000
```

One decision per line:

```jsonl
{"text": "Card was declined at checkout", "name": "intent", "question": "What does the customer want?", "labels": ["payment_issue", "refund", "shipping"], "gold": "payment_issue"}
{"text": "Box arrived crushed, I want my money back", "name": "tags", "labels": ["damaged", "refund", "late"], "gold": ["damaged", "refund"]}
```

The model is saved to `runs/mine/model`; load it with `OpenDecision.from_pretrained`.
The defaults are the released recipe: AdamW (encoder 5e-6, heads 5e-5), 200 warmup steps,
cosine decay, effective batch 8, policy cross-entropy + 0.25 × outcome BCE, bf16. A 24 GB GPU
is enough; training uses activation checkpointing.

### Training data

The released models were trained largely on decisions written and labeled by OpenAI models,
plus public datasets; the technical report describes each stage and its sources. The training
data is not distributed.

## Evaluation

```bash
pip install "opendecision[eval] @ git+https://github.com/tokz-labs/OpenDecision"
python -m opendecision.cli.evaluate --model Tokz-labs/OpenDecision-Large --set suite-full
python -m opendecision.cli.evaluate --model Tokz-labs/OpenDecision-Large --set heldout
```

`--set` builds the evaluation set on first use: `suite` (1,000 seeded test examples per dataset),
`suite-full` (complete test sets) or `heldout` (TREC, Emotion, Subjectivity, GoEmotions and the
renamed-label variants). Expected averages (macro-F1): classification suite, complete test sets —
Large 87.91, Large-Packed 87.23, Small 79.26;
held-out TREC/Emotion/Subjectivity — 57.00, 55.64,
44.16. Our predictions were produced on NVIDIA L4 GPUs in bf16; CPU runs agree up to
floating-point noise.

Every prediction behind the tables is in the
[OpenDecision-predictions](https://huggingface.co/datasets/Tokz-labs/OpenDecision-predictions) dataset
and can be re-scored without a GPU:

```bash
huggingface-cli download Tokz-labs/OpenDecision-predictions --repo-type dataset --local-dir preds
python -m opendecision.cli.evaluate --set heldout --preds preds/predictions/opendecision-large/heldout
```

## Code layout

| Path | Contents |
|---|---|
| `src/opendecision/schema.py` | decision types (`Choice`, `Bool`, `MultiLabel`, `Score`, `Decision`) |
| `src/opendecision/engine.py` | `OpenDecision`: loading, saving, `decide()` |
| `src/opendecision/encoding.py` | how states and options are tokenized (shared by inference and training) |
| `src/opendecision/models/` | cross-encoder and packed encoder |
| `src/opendecision/training/` | examples, collator, losses, training loop |
| `src/opendecision/evaluation/` | evaluation sets (suite, held-out) and scoring |
| `src/opendecision/cli/` | `train`, `evaluate` (also installed as `opendecision-train`, `opendecision-evaluate`) |
| `examples/` | runnable examples |

## Limitations

English only; inputs up to 512 tokens; option names are truncated to 32 (Large) or 48
(Packed) tokens. The evaluation suites above were also used to choose training mixtures, so
treat their scores as optimistic. Most training text was generated or labeled by OpenAI
models. The models do not reason or explain; they score the options you give them.

## License

Apache-2.0 for code and weights.

## Citation

```bibtex
@misc{alwarawreh2026opendecision,
  title  = {OpenDecision: Fast Schema-Conditioned Decisions with Packed Candidate Encoding},
  author = {Alwarawreh, Abdallah},
  year   = {2026},
  note   = {Tokz Labs technical report},
  url    = {https://github.com/tokz-labs/OpenDecision/releases/latest/download/OpenDecision.pdf}
}
```
