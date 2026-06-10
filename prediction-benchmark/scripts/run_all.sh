#!/usr/bin/env bash
# End-to-end reproduction of the LoL draft win-prediction study.
# Run from prediction-benchmark/.  Uses the project venv. Source data is read
# from the sibling embedding-analysis/ project (see src/config.py).
set -e
cd "$(dirname "$0")/.."

PY=/projects/PSALM/thisaint/venv/bin/python
export HF_HOME=/projects/PSALM/thisaint/.hf_cache
export TOKENIZERS_PARALLELISM=false
cd src

echo "==> [1/6] sanity: data + features"
$PY data.py
$PY features.py

echo "==> [2/6] classical ML grid + baselines"
$PY train_classical.py

echo "==> [3/6] SHAP champion analysis"
$PY shap_analysis.py

echo "==> [4/6] BERT fine-tune (GPU 0)"
CUDA_VISIBLE_DEVICES=0 $PY train_bert.py --epochs 3 --batch 128

# NOTE: this Blackwell host has broken GPU<->GPU P2P, so Qwen runs 8-bit on a
# SINGLE GPU (see llm_zeroshot.py). Full test split of every dataset.
echo "==> [5/6] Qwen3-32B zero-shot (GPU 0, 8-bit)"
CUDA_VISIBLE_DEVICES=0 $PY llm_zeroshot.py --batch 32

echo "==> [6/6] visualisations (UMAP + comparison)"
$PY viz_umap.py
$PY viz_results.py

echo "DONE. See results/metrics/*.csv and figures/*.png"
