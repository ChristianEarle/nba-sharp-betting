"""Shared scoring metrics for Phase 4 (model heads + baselines alike).

Every function here takes plain arrays/Series, not a model or a fold — kept
position-agnostic and dependency-light so both ``src.models.train_wr`` and
``tests/test_models.py`` import the identical implementation rather than
each rolling their own precision@k or calibration table.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss


def pr_auc(y_true, score) -> float:
    """Average precision (area under the PR curve). NaN if y_true has no positives."""
    y_true = np.asarray(y_true)
    if y_true.sum() == 0:
        return float("nan")
    return float(average_precision_score(y_true, score))


def rmse(y_true, y_pred) -> float:
    """Root mean squared error. Used for the finish_rank_delta regression head."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def brier(y_true, prob) -> float:
    """Brier score. Only meaningful for genuine probabilities in [0, 1] (the calibrated model),
    never for a baseline's unbounded score.
    """
    return float(brier_score_loss(np.asarray(y_true), np.asarray(prob)))


def top_k_indices(score, k: int = 10) -> np.ndarray:
    """Positions of the k highest-score rows, descending, ties broken by original (stable) order."""
    score = np.asarray(score)
    k = min(k, len(score))
    order = np.argsort(-score, kind="stable")
    return order[:k]


def top_k_precision(y_true, score, k: int = 10) -> float:
    """Of the k highest-score rows, the fraction with y_true == 1."""
    y_true = np.asarray(y_true)
    idx = top_k_indices(score, k)
    if len(idx) == 0:
        return float("nan")
    return float(y_true[idx].mean())


def calibration_decile_table(y_true, prob, n_bins: int = 10) -> pd.DataFrame:
    """Predicted-probability decile vs realized breakout rate, for a calibration check.

    Buckets are quantiles of ``prob`` (equal-count, not equal-width) — with
    a 3-6% base rate, equal-width bins would dump almost everything in the
    bottom bin. ``duplicates="drop"`` collapses bins with duplicate edges
    (common near a distribution's mode) rather than raising.
    """
    frame = pd.DataFrame({"y": np.asarray(y_true, dtype=float), "p": np.asarray(prob, dtype=float)})
    frame["bucket"] = pd.qcut(frame["p"], q=n_bins, duplicates="drop")
    out = (
        frame.groupby("bucket", observed=True)
        .agg(n=("y", "size"), mean_predicted=("p", "mean"), realized_rate=("y", "mean"))
        .reset_index()
    )
    return out
