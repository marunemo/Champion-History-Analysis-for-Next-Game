"""Champion metadata from DDragon (champion.json) aligned to label-encoder ids.

Produces, indexed by champion id 0..N-1:
  - a 17-dim feature matrix  (6 role tags + 4 info + 7 stats)
  - a primary role label (string) used for colouring UMAP plots
  - the human-readable champion name

Handles the alias mismatch between Oracle's Elixir and the solo-queue source
(e.g. "Aurelion Sol" vs "AurelionSol", "Bel'Veth" vs "Belveth").
"""
from __future__ import annotations
import json
import pickle
import re
import numpy as np

from config import CHAMPION_JSON, LABEL_ENCODER, N_CHAMPIONS

TAG_ORDER = ["Fighter", "Tank", "Mage", "Assassin", "Marksman", "Support"]
INFO_KEYS = ["attack", "defense", "magic", "difficulty"]
STAT_KEYS = ["hp", "armor", "spellblock", "attackdamage",
             "attackspeed", "movespeed", "hpregen"]
META_COLS = (["tag_" + t for t in TAG_ORDER]
             + ["info_" + k for k in INFO_KEYS]
             + STAT_KEYS)
META_DIM = len(META_COLS)   # 17


def _norm(name: str) -> str:
    """Normalise a champion name for fuzzy matching."""
    return re.sub(r"[^a-z0-9]", "", str(name).lower())


def load_label_encoder():
    with open(LABEL_ENCODER, "rb") as f:
        return pickle.load(f)


def _build_ddragon_index() -> dict:
    """normalised name -> raw champion record from champion.json."""
    data = json.loads(CHAMPION_JSON.read_text())["data"]
    idx = {}
    for rec in data.values():
        idx[_norm(rec["name"])] = rec
        idx[_norm(rec["id"])] = rec
    return idx


def _record_to_features(rec: dict) -> np.ndarray:
    tags = set(rec.get("tags", []))
    stats = rec.get("stats", {})
    info = rec.get("info", {})
    vec = []
    for t in TAG_ORDER:
        vec.append(1.0 if t in tags else 0.0)
    for k in INFO_KEYS:
        vec.append(float(info.get(k, 0.0)))
    for k in STAT_KEYS:
        vec.append(float(stats.get(k, 0.0)))
    return np.asarray(vec, dtype=np.float32)


def build_meta(n_champions: int = N_CHAMPIONS):
    """Return (feature_matrix [n,17], roles list[str], names list[str], found_mask).

    feature_matrix rows for unmatched ids are filled with the column mean of
    matched champions (so downstream models see no NaNs).
    """
    le = load_label_encoder()
    names = list(le.classes_)
    ddragon = _build_ddragon_index()

    feats = np.full((n_champions, META_DIM), np.nan, dtype=np.float32)
    roles = ["Unknown"] * n_champions
    out_names = [""] * n_champions
    found = np.zeros(n_champions, dtype=bool)

    for cid in range(min(n_champions, len(names))):
        nm = str(names[cid])
        out_names[cid] = nm
        rec = ddragon.get(_norm(nm))
        if rec is None:
            continue
        feats[cid] = _record_to_features(rec)
        tags = rec.get("tags", [])
        roles[cid] = tags[0] if tags else "Unknown"
        found[cid] = True

    # impute unmatched rows with column means of matched rows
    col_mean = np.nanmean(feats[found], axis=0)
    for cid in range(n_champions):
        if not found[cid]:
            feats[cid] = col_mean
    return feats, roles, out_names, found


if __name__ == "__main__":
    feats, roles, names, found = build_meta()
    print("meta matrix:", feats.shape, "| matched:", int(found.sum()), "/", len(found))
    from collections import Counter
    print("roles:", Counter(r for r, f in zip(roles, found) if f))
    # show a couple
    for cid in [0, 1, 2]:
        print(cid, names[cid], roles[cid], feats[cid].round(2))
