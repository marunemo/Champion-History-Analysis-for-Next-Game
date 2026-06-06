"""Dataset loading and train/val/test splitting.

Each label-encoded CSV has blue_p1..5, red_p1..5 (champion ids), result
(1 = blue win) plus a patch column. Splits are stratified on the target with a
fixed seed so every method (classical, BERT, Qwen) sees the *same* test rows.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from config import DATASETS, PREP_DIR, PICK_COLS, TARGET, SEED, SPLIT


def load_dataset(name: str) -> pd.DataFrame:
    fname, patch_col = DATASETS[name]
    df = pd.read_csv(PREP_DIR / fname)
    df = df.rename(columns={patch_col: "patch"})
    keep = PICK_COLS + [TARGET, "patch"]
    df = df[keep].copy()
    df[PICK_COLS] = df[PICK_COLS].astype(np.int64)
    df[TARGET] = df[TARGET].astype(np.int64)
    return df.reset_index(drop=True)


def make_splits(df: pd.DataFrame, seed: int = SEED):
    """Return dict with train/val/test DataFrames (stratified on TARGET)."""
    tr_frac, va_frac, te_frac = SPLIT
    idx = np.arange(len(df))
    y = df[TARGET].values
    tr, tmp = train_test_split(
        idx, test_size=(va_frac + te_frac), random_state=seed, stratify=y)
    rel = te_frac / (va_frac + te_frac)
    va, te = train_test_split(
        tmp, test_size=rel, random_state=seed, stratify=y[tmp])
    return {
        "train": df.iloc[tr].reset_index(drop=True),
        "val":   df.iloc[va].reset_index(drop=True),
        "test":  df.iloc[te].reset_index(drop=True),
    }


def patch_temporal_split(df: pd.DataFrame, holdout_frac: float = 0.20):
    """Robustness split for solo_all: hold out the latest patches as test.

    Patches are sorted lexically on the cleaned major.minor key so newer
    patches form the test set (no leakage of future drafts into training).
    """
    def patch_key(p):
        parts = str(p).split(".")
        try:
            return (int(parts[0]), int(parts[1]))
        except (ValueError, IndexError):
            return (0, 0)
    order = sorted(df["patch"].unique(), key=patch_key)
    n_hold = max(1, int(round(len(order) * holdout_frac)))
    test_patches = set(order[-n_hold:])
    is_test = df["patch"].isin(test_patches)
    train_val = df[~is_test].reset_index(drop=True)
    sub = make_splits(train_val)
    return {"train": sub["train"], "val": sub["val"],
            "test": df[is_test].reset_index(drop=True),
            "test_patches": sorted(test_patches, key=patch_key)}


if __name__ == "__main__":
    for name in DATASETS:
        df = load_dataset(name)
        sp = make_splits(df)
        print(f"{name:10s} n={len(df):>7d} blue_wr={df[TARGET].mean():.3f} "
              f"| train={len(sp['train'])} val={len(sp['val'])} test={len(sp['test'])} "
              f"| test_wr={sp['test'][TARGET].mean():.3f}")
