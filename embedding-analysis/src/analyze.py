import argparse
import json
import pickle
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
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


# ---------- Analysis 2: Archetype Clustering ----------

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


# ---------- Analysis 3: Delta Weight ----------

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


# ---------- Analysis 4: PCA ----------

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


# ---------- Analysis 5: PCA Annotated (PC1 vs PC2, PC1 vs PC3) ----------

def plot_pca_annotated(arch, model_name, tag_map, le):
    _, w_after = load_checkpoint(arch, model_name)
    pfx = _prefix(arch, model_name)

    pca = PCA(n_components=10)
    coords = pca.fit_transform(w_after)
    evr = pca.explained_variance_ratio_

    pc_labels = {
        0: "PC1: Engagement Range (ranged ↔ melee)",
        1: "PC2: Solo Agency (solo-carry ↔ team-dependent)",
        2: "PC3: Frontline Identity (tank ↔ carry)",
    }

    def _annotated_scatter(ax, pc_x, pc_y, n_label=5):
        for idx in range(len(le.classes_)):
            tag = tag_map.get(idx, "Unknown")
            color = TAG_COLORS.get(tag, "#95a5a6")
            ax.scatter(coords[idx, pc_x], coords[idx, pc_y],
                       c=color, s=25, alpha=0.6, zorder=2)

        # label extremes on both axes
        for pc in [pc_x, pc_y]:
            top = np.argsort(coords[:, pc])[-n_label:]
            bot = np.argsort(coords[:, pc])[:n_label]
            for idx in np.concatenate([top, bot]):
                ax.annotate(le.classes_[idx], (coords[idx, pc_x], coords[idx, pc_y]),
                            fontsize=5.5, alpha=0.85, zorder=3,
                            textcoords="offset points", xytext=(3, 3))

        ax.set_xlabel(f"{pc_labels[pc_x]}  ({evr[pc_x]:.1%})", fontsize=9)
        ax.set_ylabel(f"{pc_labels[pc_y]}  ({evr[pc_y]:.1%})", fontsize=9)

        for tag, color in TAG_COLORS.items():
            ax.scatter([], [], c=color, label=tag, s=40)
        ax.legend(fontsize=6, loc="upper right")

    fig, axes = plt.subplots(1, 2, figsize=(18, 7))

    _annotated_scatter(axes[0], 0, 1)
    axes[0].set_title(f"{pfx} — PC1 vs PC2 (annotated)", fontsize=11)

    _annotated_scatter(axes[1], 0, 2)
    axes[1].set_title(f"{pfx} — PC1 vs PC3 (annotated)", fontsize=11)

    plt.tight_layout()
    plt.savefig(f"outputs/figures/pca/{pfx}_annotated.png", dpi=150)
    plt.close()
    print(f"  Saved pca/{pfx}_annotated.png")


# ---------- Analysis 6: t-SNE Migration Arrows ----------

def plot_tsne_migration(arch, model_name, tag_map, le, top_k=10):
    w_before, w_after = load_checkpoint(arch, model_name)
    pfx = _prefix(arch, model_name)

    delta = w_after - w_before
    shift = np.linalg.norm(delta, axis=1)
    top_idx = np.argsort(shift)[::-1][:top_k]

    # Compute t-SNE on combined embeddings (before + after) for consistent space
    combined = np.vstack([w_before, w_after])
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    all_coords = tsne.fit_transform(combined)
    n = len(le.classes_)
    coords_before = all_coords[:n]
    coords_after = all_coords[n:]

    fig, ax = plt.subplots(figsize=(12, 9))

    # Plot all champions (after) as background
    for idx in range(n):
        tag = tag_map.get(idx, "Unknown")
        color = TAG_COLORS.get(tag, "#95a5a6")
        ax.scatter(coords_after[idx, 0], coords_after[idx, 1],
                   c=color, s=15, alpha=0.3, zorder=1)

    # Draw arrows for top shifted champions
    for idx in top_idx:
        tag = tag_map.get(idx, "Unknown")
        color = TAG_COLORS.get(tag, "#95a5a6")
        dx = coords_after[idx, 0] - coords_before[idx, 0]
        dy = coords_after[idx, 1] - coords_before[idx, 1]
        ax.annotate("",
                    xy=(coords_after[idx, 0], coords_after[idx, 1]),
                    xytext=(coords_before[idx, 0], coords_before[idx, 1]),
                    arrowprops=dict(arrowstyle="->", color=color, lw=1.8, alpha=0.9),
                    zorder=3)
        ax.annotate(le.classes_[idx],
                    (coords_after[idx, 0], coords_after[idx, 1]),
                    fontsize=7, fontweight="bold", alpha=0.9, zorder=4,
                    textcoords="offset points", xytext=(4, 4))
        # Mark before position
        ax.scatter(coords_before[idx, 0], coords_before[idx, 1],
                   c=color, s=50, marker="x", linewidths=1.5, zorder=2)

    for tag, color in TAG_COLORS.items():
        ax.scatter([], [], c=color, label=tag, s=40)
    ax.scatter([], [], c="gray", marker="x", s=50, label="Before (init)")
    ax.legend(fontsize=7, loc="upper right")

    ax.set_title(f"{pfx} — Top {top_k} Embedding Migration (Before → After)", fontsize=12)
    ax.set_xticks([])
    ax.set_yticks([])

    plt.tight_layout()
    plt.savefig(f"outputs/figures/tsne/{pfx}_migration.png", dpi=150)
    plt.close()
    print(f"  Saved tsne/{pfx}_migration.png")


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
        plot_pca_annotated(args.arch, name, tag_map, le)
        plot_tsne_migration(args.arch, name, tag_map, le)
    print("\nAll analyses complete.")


if __name__ == "__main__":
    main()
