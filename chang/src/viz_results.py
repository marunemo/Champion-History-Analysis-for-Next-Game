"""Unified comparison & diagnostic figures across all methods.

Consumes results/metrics/{classical,bert,qwen}.csv and results/preds/*.npz to
produce:
  results/metrics/all_methods.csv        tidy comparison table
  figures/compare_accuracy.png           best-per-method acc/AUC bars x dataset
  figures/roc_<ds>.png                   ROC curves (best classical, BERT, Qwen)
  figures/calibration_<ds>.png           reliability curves
  figures/confusion_<ds>.png             confusion matrices
"""
from __future__ import annotations
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, confusion_matrix
from sklearn.calibration import calibration_curve

from config import DATASETS, METRICS_DIR, PREDS_DIR, FIGURES_DIR
from utils import compute_metrics

warnings.filterwarnings("ignore")
DSETS = list(DATASETS)


# ---------------------------------------------------------------- tables
def build_all_methods():
    rows = []
    cls = pd.read_csv(METRICS_DIR / "classical.csv")
    # best classical (by test AUC) per dataset, excluding baselines
    real = cls[~cls.model.str.startswith("baseline")]
    for ds in DSETS:
        sub = real[real.dataset == ds]
        best = sub.loc[sub.roc_auc.idxmax()]
        rows.append(dict(dataset=ds, method=f"classical:{best.model}+{best.feature}",
                         family="classical", accuracy=best.accuracy,
                         roc_auc=best.roc_auc, log_loss=best.log_loss,
                         brier=best.brier))
        # baselines
        for _, b in cls[(cls.dataset == ds) & cls.model.str.startswith("baseline")].iterrows():
            rows.append(dict(dataset=ds, method=b.model, family="baseline",
                             accuracy=b.accuracy, roc_auc=b.roc_auc,
                             log_loss=b.log_loss, brier=b.brier))
    for fn, fam in [("bert.csv", "bert"), ("qwen.csv", "llm")]:
        p = METRICS_DIR / fn
        if p.exists():
            for _, r in pd.read_csv(p).iterrows():
                rows.append(dict(dataset=r.dataset, method=r.model, family=fam,
                                 accuracy=r.accuracy, roc_auc=r.roc_auc,
                                 log_loss=r.log_loss, brier=r.brier))
    out = pd.DataFrame(rows)
    out.to_csv(METRICS_DIR / "all_methods.csv", index=False)
    return out


# ---------------------------------------------------------------- preds io
def load_preds(ds):
    """Return dict method -> (y_true, p_blue) for the best classical, BERT, Qwen."""
    out = {}
    cz = PREDS_DIR / f"classical_{ds}.npz"
    if cz.exists():
        d = np.load(cz)
        y = d["y_true"]
        cls = pd.read_csv(METRICS_DIR / "classical.csv")
        sub = cls[(cls.dataset == ds) & ~cls.model.str.startswith("baseline")]
        best = sub.loc[sub.roc_auc.idxmax()]
        key = f"{best.model}__{best.feature}"
        if key in d:
            out[f"classical ({best.model}+{best.feature})"] = (y, d[key])
        out["blue-side prior"] = (y, d["baseline_blueprior"])
    for fam, fn in [("BERT", f"bert_{ds}.npz"), ("Qwen3-32B zero-shot", f"qwen_{ds}.npz")]:
        p = PREDS_DIR / fn
        if p.exists():
            d = np.load(p)
            out[fam] = (d["y_true"], d["p_blue"])
    return out


# ---------------------------------------------------------------- plots
def plot_compare(tbl):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for ax, metric, lo in [(axes[0], "accuracy", 0.45), (axes[1], "roc_auc", 0.45)]:
        order = ["baseline", "classical", "bert", "llm"]
        fams = [f for f in order if f in set(tbl.family)]
        width = 0.8 / len(fams)
        x = np.arange(len(DSETS))
        for i, fam in enumerate(fams):
            vals = []
            for ds in DSETS:
                s = tbl[(tbl.dataset == ds) & (tbl.family == fam)]
                vals.append(s[metric].max() if len(s) else np.nan)
            ax.bar(x + i * width, vals, width, label=fam)
        ax.axhline(0.5, ls="--", c="gray", lw=1)
        ax.set_xticks(x + width * (len(fams) - 1) / 2)
        ax.set_xticklabels(DSETS)
        ax.set_ylim(lo, max(0.62, np.nanmax(tbl[metric]) + 0.02))
        ax.set_title(metric); ax.legend(fontsize=8)
    fig.suptitle("Best score per method family (test split)", fontsize=13)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "compare_accuracy.png", dpi=140)
    plt.close(fig)
    print("Saved -> figures/compare_accuracy.png")


def plot_roc(ds):
    preds = load_preds(ds)
    plt.figure(figsize=(6, 5.5))
    for name, (y, p) in preds.items():
        if len(np.unique(y)) < 2:
            continue
        fpr, tpr, _ = roc_curve(y, p)
        plt.plot(fpr, tpr, lw=1.8, label=f"{name} (AUC={auc(fpr,tpr):.3f})")
    plt.plot([0, 1], [0, 1], ls="--", c="gray")
    plt.xlabel("FPR"); plt.ylabel("TPR"); plt.legend(fontsize=8, loc="lower right")
    plt.title(f"ROC — {ds}")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / f"roc_{ds}.png", dpi=140)
    plt.close()


def plot_calibration(ds):
    preds = load_preds(ds)
    plt.figure(figsize=(6, 5.5))
    plt.plot([0, 1], [0, 1], ls="--", c="gray", label="perfect")
    for name, (y, p) in preds.items():
        if name == "blue-side prior":
            continue
        try:
            frac, mean = calibration_curve(y, p, n_bins=8, strategy="quantile")
            plt.plot(mean, frac, marker="o", lw=1.5, ms=4, label=name)
        except ValueError:
            pass
    plt.xlabel("predicted P(blue win)"); plt.ylabel("observed")
    plt.legend(fontsize=8, loc="upper left"); plt.title(f"Calibration — {ds}")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / f"calibration_{ds}.png", dpi=140)
    plt.close()


def plot_confusion(ds):
    preds = load_preds(ds)
    items = [(k, v) for k, v in preds.items() if k != "blue-side prior"]
    if not items:
        return
    fig, axes = plt.subplots(1, len(items), figsize=(4 * len(items), 3.6))
    if len(items) == 1:
        axes = [axes]
    for ax, (name, (y, p)) in zip(axes, items):
        cm = confusion_matrix(y, (p >= 0.5).astype(int), labels=[0, 1])
        im = ax.imshow(cm, cmap="Blues")
        for (r, c), v in np.ndenumerate(cm):
            ax.text(c, r, str(v), ha="center", va="center",
                    color="white" if v > cm.max() / 2 else "black")
        ax.set_xticks([0, 1]); ax.set_xticklabels(["pred R", "pred B"])
        ax.set_yticks([0, 1]); ax.set_yticklabels(["true R", "true B"])
        ax.set_title(name, fontsize=9)
    fig.suptitle(f"Confusion matrices — {ds}")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / f"confusion_{ds}.png", dpi=140)
    plt.close(fig)


def run():
    tbl = build_all_methods()
    pd.set_option("display.width", 160)
    print(tbl.to_string(index=False))
    plot_compare(tbl)
    for ds in DSETS:
        plot_roc(ds); plot_calibration(ds); plot_confusion(ds)
        print(f"Saved -> figures/roc_{ds}.png, calibration_{ds}.png, confusion_{ds}.png")


if __name__ == "__main__":
    run()
