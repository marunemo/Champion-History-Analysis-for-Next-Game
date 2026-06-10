# Can You Predict the Winner from the Draft Alone?
### Traditional Machine Learning vs. Large-Language-Model General Knowledge for League of Legends Win Prediction

**Author:** chang · **Code:** `prediction-benchmark/` · **Data:** Champion-History-Analysis-for-Next-Game

---

## Abstract

We study a deliberately constrained question: given **only the ten drafted champions**
of a League of Legends match — no in-game statistics, no player identities — how well
can the winning side be predicted? We compare two paradigms on identical
train/validation/test splits across three datasets (professional esports, all-tier solo
queue, and Challenger solo queue). The first paradigm is **traditional machine learning**:
sixteen classifiers spanning linear, distance-based, probabilistic, tree-ensemble, and
neural families, over four engineered champion-draft representations, interpreted with
SHAP. The second is **large-language-model knowledge**: a fine-tuned BERT text classifier
and a zero-shot Qwen3-32B that relies purely on its pretrained understanding of champions.
We find that draft-only prediction is **intrinsically low-signal**: the best traditional
model reaches an AUC of only 0.54–0.57, while both LLM approaches sit essentially at chance
(AUC ≈ 0.50) and the zero-shot LLM is severely over-confident (log-loss ≈ 1.1). A
structured bag-of-champions representation consistently beats champion *names as text*. The
strongest exploitable regularity is side bias plus a handful of strong/weak champions, as
revealed by SHAP. UMAP projections show that pretrained champion embeddings cluster cleanly
by role, whereas the win-signal and LLM-name embeddings do not. The negative result is the
finding: champion composition is a weak determinant of match outcome, and general LLM
knowledge does not numerically encode the live meta.

---

## 1. Introduction

In League of Legends, two teams of five each select ("draft") champions before the match.
Conventional wisdom holds that drafts carry meaningful win-rate information — counters,
synergies, scaling. We test how much of the final outcome is actually recoverable from the
draft alone, isolating champion choice from the dominant confounds of individual skill and
in-game execution.

This isolation makes the task a clean probe of two questions:

1. **How much signal lives in the draft?** Answered by traditional ML with explicit
   feature engineering and SHAP attribution.
2. **Does an LLM's general knowledge of champions translate into win prediction?**
   Answered by zero-shot prompting of a 32B model and by fine-tuning BERT on draft text.

Our contribution is an end-to-end, reproducible comparison and an honest **negative
result** with quantified ceilings, plus interpretable and visual evidence for *why* the
ceiling is low.

## 2. Data

Three label-encoded datasets (192 champion ids, 171 actually observed), all sharing one
encoder:

| Dataset | Matches | Blue win-rate | Source |
|---|---|---|---|
| `pro` | 10,008 | 0.533 | Oracle's Elixir 2025 esports, 24 patches |
| `solo_all` | 137,146 | 0.494 | Solo queue, all tiers |
| `solo_high` | 2,566 | 0.513 | Solo queue, Challenger |

Each row is `blue_p1..5`, `red_p1..5`, and `result` (1 = blue win). Exact-draft duplicates
are negligible, so memorisation is impossible and only generalisation is measured. Auxiliary
assets: DDragon champion metadata (`champion.json`: six role tags, four info ratings, seven
base stats) and a pretrained 32-d champion embedding (`embedding_init.pt`). Splits are
**stratified 70/15/15** with a fixed seed so every method is evaluated on identical test rows.

## 3. Methods

### 3.1 Draft representations
- **bag** — Bag-of-Champions: a length-192 signed multi-hot vector, +1 for each blue
  champion and −1 for each red champion. A linear model's coefficient is then a learned
  per-champion "strength". *(primary)*
- **meta** — team-aggregated DDragon metadata (blue mean ∥ red mean ∥ difference; 51-d).
- **emb** — team-mean of pretrained champion embeddings (blue ∥ red ∥ diff; 96-d).
- **combo** — concatenation of all three (339-d).

### 3.2 Traditional classifiers (16)
Linear (Logistic Regression, Ridge, SGD-log, ElasticNet, Linear-SVM); distance (kNN,
RBF-SVM); probabilistic (Gaussian NB); tree ensembles (RandomForest, ExtraTrees,
GradientBoosting, HistGB, AdaBoost, XGBoost, LightGBM); neural (MLP). Two quadratic-cost
models (RBF-SVM, kNN) cap their training set at 12,000 on `solo_all`. Naive baselines:
majority-class and blue-side prior.

### 3.3 SHAP attribution
TreeExplainer on XGBoost + bag-of-champions. Because each feature *is* a champion, a SHAP
value is directly that champion's signed contribution to the blue-win probability.

### 3.4 LLM general knowledge
- **BERT (fine-tuned):** each draft is verbalised as `"Blue team: … . Red team: … ."` and
  `bert-base-uncased` is fine-tuned as a 2-class classifier — a text model that *learns from
  the data*.
- **Qwen3-32B (zero-shot):** no training. The model is shown both drafts and asked which
  side wins; we read the probability mass it places on the next token being "Blue" vs "Red"
  and evaluate on the full test split — purely its *pretrained knowledge*.

> **Engineering note.** The Blackwell host exhibited broken GPU↔GPU peer-to-peer copies (a
> tensor moved across devices returned all zeros), which silently corrupted any
> `device_map="auto"` model split into all-zero hidden states. We diagnosed this to the P2P
> layer and worked around it by loading Qwen3-32B in 8-bit on a single GPU (~35 GB).

## 4. Results

### 4.1 Method comparison (test split)

| Dataset | Method | Acc | AUC | LogLoss | Brier |
|---|---|---|---|---|---|
| pro | **Traditional best (ExtraTrees+bag)** | 0.555 | **0.561** | 0.686 | 0.246 |
| pro | BERT (fine-tuned) | 0.533 | 0.505 | 0.691 | 0.249 |
| pro | Qwen3-32B (zero-shot) | 0.530 | 0.484 | 1.049 | 0.349 |
| pro | Baseline (blue-side prior) | 0.533 | 0.500 | 0.691 | 0.249 |
| solo_all | **Traditional best (LogReg+combo)** | 0.525 | **0.536** | 0.691 | 0.249 |
| solo_all | BERT (fine-tuned) | 0.495 | 0.503 | 0.693 | 0.250 |
| solo_all | Qwen3-32B (zero-shot) | 0.495 | 0.503 | 1.156 | 0.374 |
| solo_all | Baseline (blue-side prior) | 0.505 | 0.500 | 0.693 | 0.250 |
| solo_high | **Traditional best (LogReg+combo)** | 0.558 | **0.570** | 0.722 | 0.261 |
| solo_high | BERT (fine-tuned) | 0.514 | 0.482 | 0.694 | 0.250 |
| solo_high | Qwen3-32B (zero-shot) | 0.522 | 0.550 | 1.121 | 0.357 |
| solo_high | Baseline (blue-side prior) | 0.512 | 0.500 | 0.693 | 0.250 |

### 4.2 Traditional families (solo_all, by test AUC)

| Family | Best model+rep | Acc | AUC |
|---|---|---|---|
| linear | logreg+combo | 0.525 | 0.536 |
| distance | svm_rbf+combo | 0.513 | 0.526 |
| probabilistic | gnb+meta | 0.516 | 0.524 |
| tree | xgb+bag | 0.523 | 0.532 |
| neural | mlp+combo | 0.521 | 0.530 |

All families land within a narrow 0.524–0.536 AUC band — the ceiling is a property of the
*task*, not the model class.

### 4.3 Figures
- `figures/compare_accuracy.png` — best accuracy/AUC per method family × dataset.
- `figures/roc_*.png`, `calibration_*.png`, `confusion_*.png` — diagnostics. Calibration
  curves expose the zero-shot LLM's over-confidence.
- `figures/shap_summary_*.png` — per-champion SHAP. High-impact champions: *pro* — Zac,
  Kha'Zix, Darius, Lux; *solo_all* — Azir, K'Sante, Singed, Veigar.
- `figures/umap_champion_embeddings.png` — pretrained embeddings form crisp role clusters
  (marksman/mage/tank/support); the bag-of-champions win-signal coefficients and Qwen name
  hidden states do **not** organise by role.
- `figures/umap_draft_separability_solo_all.png` — win/loss drafts are almost fully
  overlapping in bag space.

## 5. Discussion

**The draft is weak.** Across 16 classifiers and 4 representations, AUC never exceeds ~0.57
(and only on the small, noisy Challenger set). In the large solo_all set, the best model
beats the blue-prior baseline by ~+0.036 AUC. Match outcomes are dominated by factors
excluded here — player skill, form, in-game decisions.

**Structure beats text.** The bag-of-champions one-hot consistently outperforms feeding
champion *names* to BERT, which barely leaves the majority class (AUC ≈ 0.50). Champion
identity matters, but only when encoded as a discrete entity rather than as a token string.

**LLM general knowledge ≠ meta knowledge.** Qwen3-32B zero-shot ranks at chance and is badly
mis-calibrated (log-loss ≈ 1.1 vs. ≈ 0.69 for a calibrated coin). It "talks the talk" about
champions but does not encode current numeric win-rate relationships; on `pro` it is even
below chance (AUC 0.484), consistent with professional drafts being deliberately balanced.

**Where the little signal lives.** SHAP shows the model leaning on side bias and a short list
of strong/weak champions — exactly the regularities a human would name first.

## 6. Limitations & Future Work

- Champion identity only: no items, runes, summoner spells, bans, or pick order.
- A single prompt template and answer-token scoring for the LLM; few-shot or LoRA
  fine-tuning could lift the zero-shot ceiling (and is a natural next experiment).
- Patch drift is partially controlled by stratification; a strict temporal (newest-patch)
  holdout is provided in `data.py::patch_temporal_split` as a robustness check.
- `solo_high` is small (n_test = 385); its higher AUC carries wide error bars.

## 7. Conclusion

Predicting a League of Legends match from its draft alone is close to a coin flip. Sixteen
traditional models converge on AUC ≈ 0.54–0.57, a fine-tuned text model gains nothing over
the prior, and a 32B language model's general knowledge performs at chance while being
over-confident. The honest takeaway: **champion composition is a minor, interpretable
nudge on win probability — not a predictor — and off-the-shelf LLM knowledge does not
substitute for data-grounded modelling of the live meta.**

---

*Reproduce with `bash scripts/run_all.sh`. Metrics in `results/metrics/`, figures in
`figures/`.*
