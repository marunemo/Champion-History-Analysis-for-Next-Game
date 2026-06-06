"""Shared metrics and small IO helpers."""
from __future__ import annotations
import numpy as np
from sklearn.metrics import (accuracy_score, roc_auc_score, log_loss,
                             brier_score_loss)


def compute_metrics(y_true, p_pred) -> dict:
    """Binary classification metrics from probabilities p(blue win)."""
    y_true = np.asarray(y_true).astype(int)
    p = np.clip(np.asarray(p_pred, dtype=float), 1e-7, 1 - 1e-7)
    yhat = (p >= 0.5).astype(int)
    out = {
        "accuracy": accuracy_score(y_true, yhat),
        "log_loss": log_loss(y_true, p, labels=[0, 1]),
        "brier": brier_score_loss(y_true, p),
        "n": int(len(y_true)),
        "pos_rate": float(y_true.mean()),
    }
    # AUC undefined if a split is single-class
    out["roc_auc"] = (roc_auc_score(y_true, p)
                      if len(np.unique(y_true)) == 2 else float("nan"))
    return out


def fmt_metrics(m: dict) -> str:
    return (f"acc={m['accuracy']:.4f} auc={m['roc_auc']:.4f} "
            f"logloss={m['log_loss']:.4f} brier={m['brier']:.4f} (n={m['n']})")
