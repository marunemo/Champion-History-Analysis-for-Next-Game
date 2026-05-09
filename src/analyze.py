import argparse
import json
import pickle
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from src.model import DraftEmbeddingFFNN, DraftEmbeddingCNN

# ---------- helpers ----------

TAG_COLORS = {
    "Fighter": "#e74c3c",
    "Tank": "#3498db",
    "Mage": "#9b59b6",
    "Assassin": "#2ecc71",
    "Marksman": "#f39c12",
    "Support": "#1abc9c",
}

DATA_FILES = {
    "A": "data/processed/df_pro_enc.csv",
    "B": "data/processed/df_solo_all_enc.csv",
    "C": "data/processed/df_solo_high_enc.csv",
    "D": "data/processed/df_solo_low_enc.csv",
    "A_tl": "data/processed/df_pro_enc.csv",
    "C_tl": "data/processed/df_solo_high_enc.csv",
    "D_tl": "data/processed/df_solo_low_enc.csv",
}

ARCH_MAP = {"ffnn": DraftEmbeddingFFNN, "cnn": DraftEmbeddingCNN}


def _model_dir(model_name):
    return "transfer" if model_name.endswith("_tl") else "baseline"


def load_checkpoint(arch, model_name):
    path = f"outputs/models/{_model_dir(model_name)}/{arch}_{model_name}.pt"
    ckpt = torch.load(path, weights_only=False, map_location="cpu")
    ModelClass = ARCH_MAP[arch]
    model = ModelClass()
    model.load_state_dict(ckpt["model_state_dict"])
    w_after = model.embed.weight.detach().numpy()
    w_before = ckpt["w_before"].numpy()
    return w_before, w_after


def load_tag_map():
    with open("data/raw/champion.json") as f:
        data = json.load(f)["data"]
    with open("weights/label_encoder.pkl", "rb") as f:
        le = pickle.load(f)
    tag_map = {}
    name_to_tags = {v["name"]: v["tags"] for v in data.values()}
    id_to_tags = {v["id"]: v["tags"] for v in data.values()}
    for idx, name in enumerate(le.classes_):
        tags = name_to_tags.get(name) or id_to_tags.get(name)
        tag_map[idx] = tags[0] if tags else "Unknown"
    return tag_map, le


def _prefix(arch, model_name):
    return f"{arch}_{model_name}"


# ---------- Analysis 1: t-SNE ----------

def plot_tsne(arch, model_name, tag_map, le):
    w_before, w_after = load_checkpoint(arch, model_name)
    pfx = _prefix(arch, model_name)

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    for ax, w, title in [
        (axes[0], w_before, f"{pfx} — Before (DDragon init)"),
        (axes[1], w_after, f"{pfx} — After training"),
    ]:
        tsne = TSNE(n_components=2, random_state=42, perplexity=30)
        coords = tsne.fit_transform(w)
        for idx in range(len(le.classes_)):
            tag = tag_map.get(idx, "Unknown")
            color = TAG_COLORS.get(tag, "#95a5a6")
            ax.scatter(coords[idx, 0], coords[idx, 1], c=color, s=20, alpha=0.7)
        ax.set_title(title, fontsize=12)
        ax.set_xticks([])
        ax.set_yticks([])

    for tag, color in TAG_COLORS.items():
        axes[1].scatter([], [], c=color, label=tag, s=40)
    axes[1].legend(loc="upper right", fontsize=8)

    plt.tight_layout()
    plt.savefig(f"outputs/figures/tsne/{pfx}.png", dpi=150)
    plt.close()
    print(f"  Saved tsne_{pfx}.png")


# ---------- Analysis 2: Co-occurrence vs Affinity ----------

def plot_heatmap(arch, model_name, tag_map, le):
    _, w_after = load_checkpoint(arch, model_name)
    df = pd.read_csv(DATA_FILES[model_name])
    pfx = _prefix(arch, model_name)

    blue_cols = [f"blue_p{i}" for i in range(1, 6)]
    red_cols = [f"red_p{i}" for i in range(1, 6)]
    pick_counts = pd.Series(
        np.concatenate([df[blue_cols].values.flatten(), df[red_cols].values.flatten()])
    ).value_counts()
    top30 = pick_counts.head(30).index.tolist()

    wins = df[df["result"] == 1]
    n = len(top30)
    co_matrix = np.zeros((n, n))
    for _, row in wins.iterrows():
        picks = [row[c] for c in blue_cols]
        for i, pi in enumerate(top30):
            if pi in picks:
                for j, pj in enumerate(top30):
                    if pj in picks and i != j:
                        co_matrix[i, j] += 1
    row_sums = co_matrix.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    co_matrix /= row_sums

    emb_top = torch.tensor(w_after[top30], dtype=torch.float32)
    affinity = F.cosine_similarity(emb_top.unsqueeze(0), emb_top.unsqueeze(1), dim=2).numpy()

    labels = [le.inverse_transform([idx])[0][:8] for idx in top30]

    fig, axes = plt.subplots(1, 2, figsize=(18, 7))
    sns.heatmap(co_matrix, ax=axes[0], xticklabels=labels, yticklabels=labels, cmap="YlOrRd", square=True)
    axes[0].set_title(f"{pfx} — Co-occurrence (Blue wins)", fontsize=11)
    axes[0].tick_params(labelsize=7)

    sns.heatmap(affinity, ax=axes[1], xticklabels=labels, yticklabels=labels, cmap="coolwarm",
                center=0, square=True, vmin=-1, vmax=1)
    axes[1].set_title(f"{pfx} — Embedding Affinity (cosine)", fontsize=11)
    axes[1].tick_params(labelsize=7)

    plt.tight_layout()
    plt.savefig(f"outputs/figures/heatmap/{pfx}.png", dpi=150)
    plt.close()
    print(f"  Saved heatmap_{pfx}.png")


# ---------- Analysis 3: Archetype Clustering ----------

def plot_archetype(arch, model_name, tag_map, le):
    _, w_after = load_checkpoint(arch, model_name)
    df = pd.read_csv(DATA_FILES[model_name])
    pfx = _prefix(arch, model_name)

    blue_cols = [f"blue_p{i}" for i in range(1, 6)]
    blue_picks = df[blue_cols].values
    team_embs = np.array([w_after[picks].mean(axis=0) for picks in blue_picks])

    km = KMeans(n_clusters=7, random_state=42, n_init=10)
    labels_km = km.fit_predict(team_embs)

    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    coords = tsne.fit_transform(team_embs)

    fig, ax = plt.subplots(figsize=(9, 7))
    for k in range(7):
        mask = labels_km == k
        ax.scatter(coords[mask, 0], coords[mask, 1], s=5, alpha=0.4, label=f"Cluster {k}")
    ax.legend(fontsize=8)
    ax.set_title(f"{pfx} — Archetype Clusters (k=7)", fontsize=12)
    ax.set_xticks([])
    ax.set_yticks([])
    plt.tight_layout()
    plt.savefig(f"outputs/figures/archetype/{pfx}.png", dpi=150)
    plt.close()

    print(f"  {pfx} archetype top-5 champions:")
    for k in range(7):
        mask = labels_km == k
        cluster_picks = blue_picks[mask].flatten()
        top5_ids = pd.Series(cluster_picks).value_counts().head(5).index.tolist()
        top5_names = [le.inverse_transform([i])[0] for i in top5_ids]
        print(f"    Cluster {k} ({mask.sum()} games): {top5_names}")
    print(f"  Saved archetype_{pfx}.png")


# ---------- Analysis 4: Delta Weight ----------

def plot_delta(arch, model_name, tag_map, le):
    w_before, w_after = load_checkpoint(arch, model_name)
    pfx = _prefix(arch, model_name)
    delta = w_after - w_before
    shift = np.linalg.norm(delta, axis=1)

    top_k = 20
    top_idx = np.argsort(shift)[::-1][:top_k]
    top_names = [le.inverse_transform([i])[0] for i in top_idx]
    top_shift = shift[top_idx]
    top_colors = [TAG_COLORS.get(tag_map.get(i, ""), "#95a5a6") for i in top_idx]

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.barh(range(top_k), top_shift[::-1], color=top_colors[::-1])
    ax.set_yticks(range(top_k))
    ax.set_yticklabels(top_names[::-1], fontsize=9)
    ax.set_xlabel("L2 norm of Δweight", fontsize=10)
    ax.set_title(f"{pfx} — Top {top_k} Meta-Shifted Champions (DDragon → Learned)", fontsize=12)

    for tag, color in TAG_COLORS.items():
        ax.barh([], [], color=color, label=tag)
    ax.legend(loc="lower right", fontsize=8)

    plt.tight_layout()
    plt.savefig(f"outputs/figures/delta_shift/{pfx}.png", dpi=150)
    plt.close()
    print(f"  Saved delta_shift_{pfx}.png")


# ---------- Analysis 5: PCA ----------

def plot_pca(arch, model_name, tag_map, le):
    _, w_after = load_checkpoint(arch, model_name)
    pfx = _prefix(arch, model_name)

    # DDragon info for correlation
    with open("data/raw/champion.json") as f:
        cdata = json.load(f)["data"]
    name_to_info = {}
    for v in cdata.values():
        name_to_info[v["name"]] = v["info"]
        name_to_info[v["id"]] = v["info"]

    pca = PCA(n_components=10)
    coords = pca.fit_transform(w_after)
    evr = pca.explained_variance_ratio_

    # Build attack values for coloring
    attack_vals = []
    for idx in range(len(le.classes_)):
        info = name_to_info.get(le.classes_[idx])
        attack_vals.append(info["attack"] if info else 5)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))

    axes[0].bar(range(1, 11), evr, color="#3498db", alpha=0.8)
    axes[0].set_xlabel("Principal Component")
    axes[0].set_ylabel("Explained Variance Ratio")
    axes[0].set_title(f"{pfx} — PCA Scree Plot")
    axes[0].set_xticks(range(1, 11))

    for idx in range(len(le.classes_)):
        tag = tag_map.get(idx, "Unknown")
        color = TAG_COLORS.get(tag, "#95a5a6")
        axes[1].scatter(coords[idx, 0], coords[idx, 1], c=color, s=25, alpha=0.7)
    axes[1].set_xlabel(f"PC1 ({evr[0]:.1%})")
    axes[1].set_ylabel(f"PC2 ({evr[1]:.1%})")
    axes[1].set_title(f"{pfx} — PC1 vs PC2 (by role)")
    for tag, color in TAG_COLORS.items():
        axes[1].scatter([], [], c=color, label=tag, s=40)
    axes[1].legend(fontsize=7, loc="upper right")

    sc = axes[2].scatter(coords[:, 0], coords[:, 1], c=attack_vals, cmap="RdYlBu_r", s=25, alpha=0.7)
    axes[2].set_xlabel(f"PC1 ({evr[0]:.1%})")
    axes[2].set_ylabel(f"PC2 ({evr[1]:.1%})")
    axes[2].set_title(f"{pfx} — PC1 vs PC2 (by DDragon attack)")
    plt.colorbar(sc, ax=axes[2], label="DDragon attack")

    plt.tight_layout()
    plt.savefig(f"outputs/figures/pca/{pfx}.png", dpi=150)
    plt.close()
    print(f"  Saved pca_{pfx}.png")


# ---------- main ----------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--arch", default="ffnn", choices=["ffnn", "cnn"])
    parser.add_argument("--models", nargs="+", default=["A", "B", "C", "D"])
    args = parser.parse_args()

    tag_map, le = load_tag_map()
    for name in args.models:
        print(f"\n=== {args.arch.upper()} {name} ===")
        plot_tsne(args.arch, name, tag_map, le)
        plot_archetype(args.arch, name, tag_map, le)
        plot_delta(args.arch, name, tag_map, le)
        plot_pca(args.arch, name, tag_map, le)
    print("\nAll analyses complete.")


if __name__ == "__main__":
    main()
