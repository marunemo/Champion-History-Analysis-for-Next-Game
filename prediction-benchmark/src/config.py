"""Central configuration: paths, constants, split parameters.

This project's own outputs live under prediction-benchmark/ (results/, figures/).
Source data and pretrained champion assets are shared with the sibling
`embedding-analysis/` project, which produces them:

    <repo>/embedding-analysis/data/processed/df_*.csv   label-encoded drafts
    <repo>/embedding-analysis/data/raw/champion.json     DDragon metadata
    <repo>/embedding-analysis/weights/label_encoder.pkl  champion id <-> name
    <repo>/embedding-analysis/weights/embedding_init.pt  pretrained [192,32]
"""
from __future__ import annotations
import os
from pathlib import Path

# ---------------------------------------------------------------- paths
PROJECT_ROOT = Path(__file__).resolve().parents[1]          # prediction-benchmark/
REPO_ROOT = PROJECT_ROOT.parent                             # repository root


def _find_data_root() -> Path:
    """Locate the sibling embedding-analysis project that ships the data.

    Primary layout: <repo>/prediction-benchmark + <repo>/embedding-analysis.
    Falls back to an upward search for a directory that holds both
    `data/processed/` and `weights/label_encoder.pkl`.
    """
    sentinel = Path("weights") / "label_encoder.pkl"
    candidates = [
        REPO_ROOT / "embedding-analysis",   # sibling project (canonical)
        REPO_ROOT,                          # data kept at repo root
    ]
    for c in candidates:
        if (c / sentinel).exists() and (c / "data" / "processed").is_dir():
            return c
    for base in [REPO_ROOT, *REPO_ROOT.parents]:
        if (base / "embedding-analysis" / sentinel).exists():
            return base / "embedding-analysis"
    return candidates[0]   # default (clear error downstream if missing)


DATA_ROOT = _find_data_root()
PREP_DIR = DATA_ROOT / "data" / "processed"       # label-encoded CSVs
WEIGHTS_DIR = DATA_ROOT / "weights"

CHAMPION_JSON = DATA_ROOT / "data" / "raw" / "champion.json"
LABEL_ENCODER = WEIGHTS_DIR / "label_encoder.pkl"
EMBEDDING_INIT = WEIGHTS_DIR / "embedding_init.pt"

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
