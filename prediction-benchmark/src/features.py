"""Feature representations for the draft -> win-prediction task.

Three complementary champion-draft encodings (plus their concatenation):

  bag      Bag-of-Champions, signed multi-hot of length N_CHAMPIONS.
           +1 for each blue champion, -1 for each red champion. Lets a linear
           model learn a per-champion "strength" coefficient directly.

  meta     Composition features from DDragon metadata (17-dim/champion):
           blue team mean, red team mean, and (blue-mean - red-mean) diff.
           Captures archetype (#tanks, total AD, ...).  -> 51 dims.

  emb      Pretrained embedding_init (32-dim/champion): blue mean, red mean,
           and their difference.  -> 96 dims.

  combo    bag || meta || emb.
"""
from __future__ import annotations
import numpy as np
import torch

from config import PICK_COLS, BLUE_COLS, RED_COLS, N_CHAMPIONS, EMBEDDING_INIT
from champion_meta import build_meta


def bag_of_champions(df) -> np.ndarray:
    X = np.zeros((len(df), N_CHAMPIONS), dtype=np.float32)
    rows = np.arange(len(df))
    for c in BLUE_COLS:
        np.add.at(X, (rows, df[c].values), 1.0)
    for c in RED_COLS:
        np.add.at(X, (rows, df[c].values), -1.0)
    return X


def _team_pool(df, table: np.ndarray):
    """Return (blue_mean, red_mean) where table is [N_CHAMPIONS, d]."""
    blue = np.stack([table[df[c].values] for c in BLUE_COLS], axis=1).mean(1)
    red = np.stack([table[df[c].values] for c in RED_COLS], axis=1).mean(1)
    return blue.astype(np.float32), red.astype(np.float32)


_META_TABLE = None
_EMB_TABLE = None


def _meta_table():
    global _META_TABLE
    if _META_TABLE is None:
        feats, *_ = build_meta(N_CHAMPIONS)
        # standardise columns so means are comparable across heterogeneous scales
        mu, sd = feats.mean(0), feats.std(0) + 1e-6
        _META_TABLE = ((feats - mu) / sd).astype(np.float32)
    return _META_TABLE


def _emb_table():
    global _EMB_TABLE
    if _EMB_TABLE is None:
        emb = torch.load(EMBEDDING_INIT, map_location="cpu").numpy().astype(np.float32)
        if emb.shape[0] < N_CHAMPIONS:   # pad if needed
            pad = np.zeros((N_CHAMPIONS - emb.shape[0], emb.shape[1]), np.float32)
            emb = np.vstack([emb, pad])
        _EMB_TABLE = emb
    return _EMB_TABLE


def meta_features(df) -> np.ndarray:
    b, r = _team_pool(df, _meta_table())
    return np.concatenate([b, r, b - r], axis=1)


def emb_features(df) -> np.ndarray:
    b, r = _team_pool(df, _emb_table())
    return np.concatenate([b, r, b - r], axis=1)


def build_features(df, kind: str) -> np.ndarray:
    if kind == "bag":
        return bag_of_champions(df)
    if kind == "meta":
        return meta_features(df)
    if kind == "emb":
        return emb_features(df)
    if kind == "combo":
        return np.concatenate(
            [bag_of_champions(df), meta_features(df), emb_features(df)], axis=1)
    raise ValueError(f"unknown feature kind: {kind}")


FEATURE_KINDS = ["bag", "meta", "emb", "combo"]


def feature_names(kind: str):
    from champion_meta import build_meta, META_COLS
    _, _, names, _ = build_meta(N_CHAMPIONS)
    if kind == "bag":
        return [f"champ::{names[i]}" for i in range(N_CHAMPIONS)]
    if kind == "meta":
        return ([f"blue_{c}" for c in META_COLS]
                + [f"red_{c}" for c in META_COLS]
                + [f"diff_{c}" for c in META_COLS])
    if kind == "emb":
        d = _emb_table().shape[1]
        return ([f"blue_e{i}" for i in range(d)]
                + [f"red_e{i}" for i in range(d)]
                + [f"diff_e{i}" for i in range(d)])
    if kind == "combo":
        return feature_names("bag") + feature_names("meta") + feature_names("emb")
    raise ValueError(kind)


if __name__ == "__main__":
    from data import load_dataset, make_splits
    df = load_dataset("solo_high")
    sp = make_splits(df)
    for k in FEATURE_KINDS:
        X = build_features(sp["train"], k)
        print(f"{k:6s} -> {X.shape}  range[{X.min():.2f},{X.max():.2f}]  names={len(feature_names(k))}")
