"""UMAP 2-D visualisation of champion embeddings.

Four embedding sources, each reduced to 2-D and coloured by champion role:
  init   pretrained embedding_init.pt        [192, 32]
  bag    LogReg bag-of-champions coefficients (data-driven champion vector)
  bert   fine-tuned BERT champion-name embedding   [192, 768]
  qwen   Qwen3-32B champion-name hidden state       [192, 5120]

Also a draft-level UMAP (test drafts coloured by win/loss) to visualise how
weakly the classes separate in the bag-of-champions space.
"""
from __future__ import annotations
import warnings
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import umap
from sklearn.linear_model import LogisticRegression

from config import (EMBEDDING_INIT, EMB_DIR, FIGURES_DIR, N_CHAMPIONS, SEED,
                    TARGET)
from champion_meta import build_meta
from data import load_dataset, make_splits
from features import bag_of_champions

warnings.filterwarnings("ignore")

ROLE_COLORS = {
    "Fighter": "#d62728", "Tank": "#8c564b", "Mage": "#1f77b4",
    "Assassin": "#9467bd", "Marksman": "#2ca02c", "Support": "#e377c2",
    "Unknown": "#7f7f7f",
}


def _umap(X, seed=SEED):
    n = X.shape[0]
    reducer = umap.UMAP(n_neighbors=min(15, n - 1), min_dist=0.25,
                        metric="cosine", random_state=seed)
    return reducer.fit_transform(X)


def bag_logreg_vectors():
    """Per-champion coefficient learned by LogReg on solo_all bag-of-champions.

    A champion's coefficient is its learned contribution to blue-win logit ->
    a 1-D 'strength'; we expand to a small vector by stacking the coefficient
    across the three datasets so UMAP has structure to work with.
    """
    cols = []
    for ds in ("pro", "solo_all", "solo_high"):
        sp = make_splits(load_dataset(ds))
        X = bag_of_champions(sp["train"]); y = sp["train"][TARGET].values
        lr = LogisticRegression(C=1.0, max_iter=2000).fit(X, y)
        cols.append(lr.coef_[0])
    return np.stack(cols, axis=1)        # [N_CHAMPIONS, 3]


def load_sources():
    src = {}
    init = torch.load(EMBEDDING_INIT, map_location="cpu").numpy()
    if init.shape[0] < N_CHAMPIONS:
        init = np.vstack([init, np.zeros((N_CHAMPIONS - init.shape[0], init.shape[1]))])
    src["init"] = init
    src["bag"] = bag_logreg_vectors()
    for tag, fn in [("bert", "bert_champ_solo_all.npy"), ("qwen", "qwen_champ.npy")]:
        p = EMB_DIR / fn
        if p.exists():
            src[tag] = np.load(p)
    return src


def plot_champion_umaps():
    feats, roles, names, found = build_meta(N_CHAMPIONS)
    src = load_sources()
    titles = {"init": "Pretrained embedding_init",
              "bag": "LogReg bag-of-champions coef",
              "bert": "Fine-tuned BERT name emb",
              "qwen": "Qwen3-32B name hidden state"}
    keys = [k for k in ["init", "bag", "bert", "qwen"] if k in src]
    fig, axes = plt.subplots(1, len(keys), figsize=(5.2 * len(keys), 5))
    if len(keys) == 1:
        axes = [axes]
    for ax, k in zip(axes, keys):
        X = src[k].astype(np.float32)
        mask = found if X.shape[0] == N_CHAMPIONS else np.ones(X.shape[0], bool)
        emb = _umap(X[mask])
        rr = [roles[i] for i in range(N_CHAMPIONS) if mask[i]]
        for role in ROLE_COLORS:
            sel = [j for j, r in enumerate(rr) if r == role]
            if sel:
                ax.scatter(emb[sel, 0], emb[sel, 1], s=18,
                           c=ROLE_COLORS[role], label=role, alpha=0.8,
                           edgecolors="none")
        ax.set_title(f"{titles[k]}\n({X.shape[1]}-d)", fontsize=11)
        ax.set_xticks([]); ax.set_yticks([])
    axes[-1].legend(markerscale=1.4, fontsize=8, loc="upper right",
                    framealpha=0.9)
    fig.suptitle("Champion embeddings — UMAP 2-D, coloured by role", fontsize=13)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "umap_champion_embeddings.png", dpi=140)
    plt.close(fig)
    print("Saved -> figures/umap_champion_embeddings.png  (sources:", keys, ")")


def plot_draft_separability(ds="solo_all", n=4000):
    sp = make_splits(load_dataset(ds))
    test = sp["test"]
    if len(test) > n:
        test = test.sample(n, random_state=SEED)
    X = bag_of_champions(test)
    y = test[TARGET].values
    emb = _umap(X)
    plt.figure(figsize=(6.5, 5.5))
    for lab, col, name in [(1, "#1f77b4", "Blue win"), (0, "#d62728", "Red win")]:
        sel = y == lab
        plt.scatter(emb[sel, 0], emb[sel, 1], s=6, c=col, alpha=0.4, label=name)
    plt.legend(markerscale=2)
    plt.xticks([]); plt.yticks([])
    plt.title(f"Draft-level UMAP ({ds} test, bag-of-champions)\n"
              "classes overlap -> draft-only signal is weak")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / f"umap_draft_separability_{ds}.png", dpi=140)
    plt.close()
    print(f"Saved -> figures/umap_draft_separability_{ds}.png")


if __name__ == "__main__":
    plot_champion_umaps()
    plot_draft_separability("solo_all")
