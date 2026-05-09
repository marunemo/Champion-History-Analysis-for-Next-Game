import json
import pickle
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from src.model import DraftEmbeddingFFNN

# ---------- helpers ----------

TAG_COLORS = {
    "Fighter": "#e74c3c",
    "Tank": "#3498db",
    "Mage": "#9b59b6",
    "Assassin": "#2ecc71",
    "Marksman": "#f39c12",
    "Support": "#1abc9c",
}


def load_checkpoint(model_name):
    path = f"outputs/models/model_{model_name}.pt"
    ckpt = torch.load(path, weights_only=False, map_location="cpu")
    model = DraftEmbeddingFFNN()
    model.load_state_dict(ckpt["model_state_dict"])
    w_after = model.embed.weight.detach().numpy()
    w_before = ckpt["w_before"].numpy()
    return w_before, w_after


def load_tag_map():
    with open("dataset/champion.json") as f:
        data = json.load(f)["data"]
    with open("weights/label_encoder.pkl", "rb") as f:
        le = pickle.load(f)
    tag_map = {}
    name_to_tags = {v["name"]: v["tags"] for v in data.values()}
    # also try matching by id
    id_to_tags = {v["id"]: v["tags"] for v in data.values()}
    for idx, name in enumerate(le.classes_):
        tags = name_to_tags.get(name) or id_to_tags.get(name)
        tag_map[idx] = tags[0] if tags else "Unknown"
    return tag_map, le


# ---------- Analysis 1: t-SNE ----------

def plot_tsne(model_name, tag_map, le):
    w_before, w_after = load_checkpoint(model_name)

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    for ax, w, title in [
        (axes[0], w_before, f"Model {model_name} — Before (DDragon init)"),
        (axes[1], w_after, f"Model {model_name} — After training"),
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

    # legend
    for tag, color in TAG_COLORS.items():
        axes[1].scatter([], [], c=color, label=tag, s=40)
    axes[1].legend(loc="upper right", fontsize=8)

    plt.tight_layout()
    plt.savefig(f"outputs/figures/tsne_{model_name}.png", dpi=150)
    plt.close()
    print(f"  Saved tsne_{model_name}.png")


# ---------- Analysis 2: Co-occurrence vs Affinity ----------

def plot_heatmap(model_name, tag_map, le):
    _, w_after = load_checkpoint(model_name)
    data_files = {"A": "data/df_pro_enc.csv", "B": "data/df_solo_all_enc.csv", "C": "data/df_solo_high_enc.csv"}
    df = pd.read_csv(data_files[model_name])

    # pick rate top 30
    blue_cols = [f"blue_p{i}" for i in range(1, 6)]
    red_cols = [f"red_p{i}" for i in range(1, 6)]
    all_picks = pd.concat([df[blue_cols].values.flatten(), df[red_cols].values.flatten()], ignore_index=True) if False else None
    pick_counts = pd.Series(
        np.concatenate([df[blue_cols].values.flatten(), df[red_cols].values.flatten()])
    ).value_counts()
    top30 = pick_counts.head(30).index.tolist()

    # co-occurrence (winning blue team only)
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
    # row normalize
    row_sums = co_matrix.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    co_matrix /= row_sums

    # affinity
    emb_top = torch.tensor(w_after[top30], dtype=torch.float32)
    affinity = F.cosine_similarity(emb_top.unsqueeze(0), emb_top.unsqueeze(1), dim=2).numpy()

    labels = [le.inverse_transform([idx])[0][:8] for idx in top30]

    fig, axes = plt.subplots(1, 2, figsize=(18, 7))
    sns.heatmap(co_matrix, ax=axes[0], xticklabels=labels, yticklabels=labels, cmap="YlOrRd", square=True)
    axes[0].set_title(f"Model {model_name} — Co-occurrence (Blue wins)", fontsize=11)
    axes[0].tick_params(labelsize=7)

    sns.heatmap(affinity, ax=axes[1], xticklabels=labels, yticklabels=labels, cmap="coolwarm",
                center=0, square=True, vmin=-1, vmax=1)
    axes[1].set_title(f"Model {model_name} — Embedding Affinity (cosine)", fontsize=11)
    axes[1].tick_params(labelsize=7)

    plt.tight_layout()
    plt.savefig(f"outputs/figures/heatmap_{model_name}.png", dpi=150)
    plt.close()
    print(f"  Saved heatmap_{model_name}.png")


# ---------- Analysis 3: Archetype Clustering ----------

def plot_archetype(model_name, tag_map, le):
    _, w_after = load_checkpoint(model_name)
    data_files = {"A": "data/df_pro_enc.csv", "B": "data/df_solo_all_enc.csv", "C": "data/df_solo_high_enc.csv"}
    df = pd.read_csv(data_files[model_name])

    blue_cols = [f"blue_p{i}" for i in range(1, 6)]
    # mean pool 5 picks
    blue_picks = df[blue_cols].values  # (N, 5)
    team_embs = np.array([w_after[picks].mean(axis=0) for picks in blue_picks])  # (N, 32)

    km = KMeans(n_clusters=7, random_state=42, n_init=10)
    labels_km = km.fit_predict(team_embs)

    # t-SNE on team embeddings
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    coords = tsne.fit_transform(team_embs)

    fig, ax = plt.subplots(figsize=(9, 7))
    for k in range(7):
        mask = labels_km == k
        ax.scatter(coords[mask, 0], coords[mask, 1], s=5, alpha=0.4, label=f"Cluster {k}")
    ax.legend(fontsize=8)
    ax.set_title(f"Model {model_name} — Archetype Clusters (k=7)", fontsize=12)
    ax.set_xticks([])
    ax.set_yticks([])
    plt.tight_layout()
    plt.savefig(f"outputs/figures/archetype_{model_name}.png", dpi=150)
    plt.close()

    # print top5 champions per cluster
    print(f"  Model {model_name} archetype top-5 champions:")
    for k in range(7):
        mask = labels_km == k
        cluster_picks = blue_picks[mask].flatten()
        top5_ids = pd.Series(cluster_picks).value_counts().head(5).index.tolist()
        top5_names = [le.inverse_transform([i])[0] for i in top5_ids]
        print(f"    Cluster {k} ({mask.sum()} games): {top5_names}")

    print(f"  Saved archetype_{model_name}.png")


# ---------- Analysis 4: Delta Weight ----------

def plot_delta(model_name, tag_map, le):
    w_before, w_after = load_checkpoint(model_name)
    delta = w_after - w_before
    shift = np.linalg.norm(delta, axis=1)  # (192,)

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
    ax.set_title(f"Model {model_name} — Top {top_k} Meta-Shifted Champions (DDragon → Learned)", fontsize=12)

    for tag, color in TAG_COLORS.items():
        ax.barh([], [], color=color, label=tag)
    ax.legend(loc="lower right", fontsize=8)

    plt.tight_layout()
    plt.savefig(f"outputs/figures/delta_shift_{model_name}.png", dpi=150)
    plt.close()
    print(f"  Saved delta_shift_{model_name}.png")


# ---------- main ----------

def main():
    tag_map, le = load_tag_map()
    for name in ["A", "B", "C"]:
        print(f"\n=== Model {name} ===")
        plot_tsne(name, tag_map, le)
        plot_heatmap(name, tag_map, le)
        plot_archetype(name, tag_map, le)
        plot_delta(name, tag_map, le)
    print("\nAll analyses complete.")


if __name__ == "__main__":
    main()
