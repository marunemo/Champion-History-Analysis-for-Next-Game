"""Central configuration: paths, constants, split parameters.

All artifacts live under chang/lol_winpred/. Source data is read from the
cloned repo at chang/repo/ (read-only).
"""
from __future__ import annotations
import os
from pathlib import Path

# ---------------------------------------------------------------- paths
PROJECT_ROOT = Path(__file__).resolve().parents[1]          # project root


def _find_dataset_dir() -> Path:
    """Locate the source `dataset/` directory across supported layouts.

    Works whether this project sits next to the cloned repo
    (chang/lol_winpred + chang/repo/dataset) or nested inside it as a folder
    (Champion-History.../chang + Champion-History.../dataset). Falls back to an
    upward search for `dataset/preprocessed results/label_encoder.pkl`.
    """
    sentinel = Path("dataset") / "preprocessed results" / "label_encoder.pkl"
    candidates = [
        PROJECT_ROOT.parent / "repo" / "dataset",   # chang/lol_winpred + chang/repo
        PROJECT_ROOT.parent / "dataset",            # nested: <repo>/chang + <repo>/dataset
        PROJECT_ROOT / "dataset",                   # dataset shipped inside the project
    ]
    for c in candidates:
        if (c / "preprocessed results" / "label_encoder.pkl").exists():
            return c
    for base in [PROJECT_ROOT, *PROJECT_ROOT.parents]:
        if (base / sentinel).exists():
            return base / "dataset"
    return candidates[0]   # original default (clear error downstream if missing)


DATASET_DIR = _find_dataset_dir()
REPO_ROOT = DATASET_DIR.parent
CHANG_ROOT = PROJECT_ROOT.parent
PREP_DIR = DATASET_DIR / "preprocessed results"

CHAMPION_JSON = DATASET_DIR / "champion.json"
LABEL_ENCODER = PREP_DIR / "label_encoder.pkl"
EMBEDDING_INIT = PREP_DIR / "embedding_init.pt"

RESULTS_DIR = PROJECT_ROOT / "results"
METRICS_DIR = RESULTS_DIR / "metrics"
PREDS_DIR = RESULTS_DIR / "preds"
SHAP_DIR = RESULTS_DIR / "shap"
EMB_DIR = RESULTS_DIR / "embeddings"
FIGURES_DIR = PROJECT_ROOT / "figures"
for _d in (METRICS_DIR, PREDS_DIR, SHAP_DIR, EMB_DIR, FIGURES_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------- datasets
# name -> (csv filename, patch column name)
DATASETS = {
    "pro":       ("df_pro_enc.csv",       "patch"),
    "solo_all":  ("df_solo_all_enc.csv",  "Patch"),
    "solo_high": ("df_solo_high_enc.csv", "Patch"),
}

BLUE_COLS = [f"blue_p{i}" for i in range(1, 6)]
RED_COLS = [f"red_p{i}" for i in range(1, 6)]
PICK_COLS = BLUE_COLS + RED_COLS
TARGET = "result"          # 1 = blue side wins

N_CHAMPIONS = 192          # label encoder cardinality (0..191)

# ---------------------------------------------------------------- splits
SEED = 42
SPLIT = (0.70, 0.15, 0.15)   # train / val / test, stratified on TARGET

# ---------------------------------------------------------------- models
HF_HOME = os.environ.get("HF_HOME", "/projects/PSALM/thisaint/.hf_cache")
os.environ.setdefault("HF_HOME", HF_HOME)
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

BERT_MODEL = "bert-base-uncased"
QWEN_MODEL = "Qwen/Qwen3-32B"
