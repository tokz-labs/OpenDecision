"""Evaluation sets (classification suite, held-out label sets): build inputs, run a model, score."""

from . import heldout
from .metrics import SPLITS, macro_f1, multi_label_scores, predict, score
from .suite import build

__all__ = ["SPLITS", "build", "heldout", "macro_f1", "multi_label_scores", "predict", "score"]
