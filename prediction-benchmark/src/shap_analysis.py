"""SHAP analysis of the draft -> blue-win model.

We deliberately use XGBoost on the Bag-of-Champions representation: each feature
is one champion (+1 if drafted blue, -1 if red), so a SHAP value is directly the
champion's signed push on the blue-win probability for that game. Aggregating
gives a data-driven champion "strength / blue-favourability" ranking that we can
sanity-check against intuition.

Outputs per dataset:
  results/shap/shap_champion_<ds>.csv   importance + mean signed effect
  figures/shap_summary_<ds>.png         beeswarm of top champions
"""
from __future__ import annotations
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import shap
from xgboost import XGBClassifier

from config import DATASETS, TARGET, SHAP_DIR, FIGURES_DIR, SEED, N_CHAMPIONS
from data import load_dataset, make_splits
from features import bag_of_champions
from champion_meta import build_meta

warnings.filterwarnings("ignore")


def run():
    _, roles, names, _ = build_meta(N_CHAMPIONS)
    for name in DATASETS:
        df = load_dataset(name)
        sp = make_splits(df)
        Xtr = bag_of_champions(sp["train"])
        ytr = sp["train"][TARGET].values
        Xte = bag_of_champions(sp["test"])

        model = XGBClassifier(
            n_estimators=600, max_depth=5, learning_rate=0.03,
            subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0,
            eval_metric="logloss", tree_method="hist", n_jobs=-1,
            random_state=SEED)
        model.fit(Xtr, ytr)

        # SHAP on a sample of the test set (TreeExplainer is exact & fast)
        rng = np.random.RandomState(SEED)
        n = min(2000, len(Xte))
        idx = rng.choice(len(Xte), n, replace=False)
        Xs = Xte[idx]
        explainer = shap.TreeExplainer(model)
        sv = explainer.shap_values(Xs)            # [n, N_CHAMPIONS]

        # per-champion: only count games where the champion was actually drafted
        present = Xs != 0                          # [n, N]
        abs_imp = np.where(present, np.abs(sv), np.nan)
        mean_abs = np.nanmean(abs_imp, axis=0)
        # signed effect on BLUE win prob, normalised so +1=blue,-1=red side:
        # sv * sign(feature) gives the champion's push regardless of its side.
        signed = np.where(present, sv * np.sign(Xs), np.nan)
        mean_signed = np.nanmean(signed, axis=0)
        n_drafted = present.sum(0)

        tbl = pd.DataFrame({
            "champion": [names[i] for i in range(N_CHAMPIONS)],
            "role": [roles[i] for i in range(N_CHAMPIONS)],
            "mean_abs_shap": mean_abs,
            "mean_signed_shap": mean_signed,
            "n_drafted": n_drafted,
        })
        tbl = tbl[tbl.n_drafted >= 5].sort_values(
            "mean_abs_shap", ascending=False).reset_index(drop=True)
        tbl.to_csv(SHAP_DIR / f"shap_champion_{name}.csv", index=False)

        # beeswarm of the top-20 champions by importance
        top = tbl.head(20)["champion"].tolist()
        top_ids = [names.index(c) for c in top]
        expl = shap.Explanation(
            values=sv[:, top_ids], data=Xs[:, top_ids], feature_names=top)
        plt.figure(figsize=(8, 8))
        shap.summary_plot(expl.values, expl.data, feature_names=top,
                          show=False, plot_size=None, max_display=20)
        plt.title(f"SHAP — top champions ({name})  feature=+1 blue / -1 red")
        plt.tight_layout()
        plt.savefig(FIGURES_DIR / f"shap_summary_{name}.png", dpi=130)
        plt.close()

        print(f"[{name}] top blue-favourable:",
              ", ".join(tbl.sort_values('mean_signed_shap', ascending=False)
                        .head(5)['champion']))
        print(f"[{name}] top red-favourable :",
              ", ".join(tbl.sort_values('mean_signed_shap')
                        .head(5)['champion']))
    print("Saved -> results/shap/*.csv ; figures/shap_summary_*.png")


if __name__ == "__main__":
    run()
