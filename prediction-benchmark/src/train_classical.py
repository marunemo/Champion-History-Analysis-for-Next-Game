"""Traditional ML: draft -> blue-win classification.

Grid = {16 classical classifiers across linear / distance / probabilistic /
tree-ensemble / neural families} x {3 datasets} x {4 feature representations}
plus two naive baselines (majority class, blue-side prior). Probabilities on the
test split are saved for the unified comparison/visualisation step; metrics go to
results/metrics/classical.csv.

The best (model, feature) per dataset by validation AUC is persisted so SHAP and
UMAP can reuse the exact fitted estimator.

Two quadratic-cost models (RBF-SVM, kNN) cap their training set on large
datasets (see TRAIN_CAP); the cap actually used is recorded in the output.
"""
from __future__ import annotations
import json
import warnings
import numpy as np
import pandas as pd
from joblib import dump
from scipy.special import expit

from sklearn.linear_model import (LogisticRegression, RidgeClassifier,
                                  SGDClassifier)
from sklearn.ensemble import (RandomForestClassifier, ExtraTreesClassifier,
                              GradientBoostingClassifier,
                              HistGradientBoostingClassifier,
                              AdaBoostClassifier)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC, SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.calibration import CalibratedClassifierCV
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

from config import DATASETS, TARGET, METRICS_DIR, PREDS_DIR, RESULTS_DIR, SEED
from data import load_dataset, make_splits
from features import build_features, FEATURE_KINDS
from utils import compute_metrics, fmt_metrics

warnings.filterwarnings("ignore")
MODELS_DIR = RESULTS_DIR / "models"
MODELS_DIR.mkdir(exist_ok=True)

# models whose training is O(n^2): cap the train set this large on big datasets
TRAIN_CAP = {"svm_rbf": 12000, "knn": 12000}


def make_models():
    """name -> (family, estimator).  Families group the comparison plots."""
    return {
        # ---- linear ----
        "logreg": ("linear", LogisticRegression(C=1.0, max_iter=2000, n_jobs=-1)),
        "ridge": ("linear", CalibratedClassifierCV(
            RidgeClassifier(alpha=1.0), method="sigmoid", cv=3)),
        "sgd_log": ("linear", SGDClassifier(
            loss="log_loss", penalty="l2", alpha=1e-4, max_iter=50,
            random_state=SEED)),
        "elasticnet": ("linear", SGDClassifier(
            loss="log_loss", penalty="elasticnet", l1_ratio=0.15, alpha=1e-4,
            max_iter=50, random_state=SEED)),
        "svm_linear": ("linear", CalibratedClassifierCV(
            LinearSVC(C=0.5, max_iter=5000), method="sigmoid", cv=3)),
        # ---- distance / probabilistic ----
        "knn": ("distance", KNeighborsClassifier(n_neighbors=50, n_jobs=-1)),
        "svm_rbf": ("distance", SVC(C=1.0, kernel="rbf", gamma="scale",
                                    probability=False, random_state=SEED)),
        "gnb": ("probabilistic", GaussianNB()),
        # ---- tree ensembles ----
        "rf": ("tree", RandomForestClassifier(
            n_estimators=400, min_samples_leaf=5, n_jobs=-1, random_state=SEED)),
        "extratrees": ("tree", ExtraTreesClassifier(
            n_estimators=400, min_samples_leaf=5, n_jobs=-1, random_state=SEED)),
        "gradboost": ("tree", GradientBoostingClassifier(
            n_estimators=200, max_depth=3, learning_rate=0.05, subsample=0.5,
            random_state=SEED)),
        "histgb": ("tree", HistGradientBoostingClassifier(
            max_iter=400, learning_rate=0.05, max_depth=4, random_state=SEED)),
        "adaboost": ("tree", AdaBoostClassifier(
            n_estimators=300, learning_rate=0.5, random_state=SEED)),
        "xgb": ("tree", XGBClassifier(
            n_estimators=600, max_depth=5, learning_rate=0.03,
            subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0,
            eval_metric="logloss", tree_method="hist", n_jobs=-1,
            random_state=SEED)),
        "lgbm": ("tree", LGBMClassifier(
            n_estimators=600, num_leaves=63, learning_rate=0.03,
            subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0,
            n_jobs=-1, random_state=SEED, verbose=-1)),
        # ---- neural ----
        "mlp": ("neural", MLPClassifier(
            hidden_layer_sizes=(256, 64), alpha=1e-3, batch_size=256,
            learning_rate_init=1e-3, max_iter=60, early_stopping=True,
            random_state=SEED)),
    }


def get_proba(model, X):
    """P(blue win) for estimators with or without predict_proba."""
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    if hasattr(model, "decision_function"):
        return expit(model.decision_function(X))
    return model.predict(X).astype(float)


def fit_capped(model, mname, X, y, seed=SEED):
    """Fit, sub-sampling the train set for quadratic-cost models on big data."""
    cap = TRAIN_CAP.get(mname)
    if cap and len(X) > cap:
        rng = np.random.RandomState(seed)
        idx = rng.choice(len(X), cap, replace=False)
        model.fit(X[idx], y[idx])
        return cap
    model.fit(X, y)
    return len(X)


def baseline_rows(name, sp):
    """Majority-class and blue-side-prior baselines (probabilities)."""
    ytr = sp["train"][TARGET].values
    yte = sp["test"][TARGET].values
    rows = []
    # majority: predict the more frequent class, p = its train frequency
    p_blue = ytr.mean()
    maj_p = np.full(len(yte), 1.0 if p_blue >= 0.5 else 0.0)
    rows.append(("baseline_majority", "-", compute_metrics(yte, maj_p)))
    # blue-side prior: constant probability = train blue win-rate
    prior_p = np.full(len(yte), p_blue)
    rows.append(("baseline_blueprior", "-", compute_metrics(yte, prior_p)))
    return rows, {"baseline_majority": maj_p, "baseline_blueprior": prior_p}


def run():
    all_rows = []
    best_per_dataset = {}
    for name in DATASETS:
        df = load_dataset(name)
        sp = make_splits(df)
        ytr, yva, yte = (sp[s][TARGET].values for s in ("train", "val", "test"))
        print(f"\n========== {name}  (train={len(ytr)} val={len(yva)} test={len(yte)}) ==========")

        preds_store = {}
        brows, bpreds = baseline_rows(name, sp)
        for mdl, feat, m in brows:
            all_rows.append(dict(dataset=name, model=mdl, feature=feat, split="test", **m))
            print(f"  {mdl:20s}            {fmt_metrics(m)}")
        preds_store.update(bpreds)

        best = None  # (val_auc, model_name, feat, fitted, test_metrics)
        for feat in FEATURE_KINDS:
            Xtr = build_features(sp["train"], feat)
            Xva = build_features(sp["val"], feat)
            Xte = build_features(sp["test"], feat)
            for mname, (family, model) in make_models().items():
                n_used = fit_capped(model, mname, Xtr, ytr)
                pva = get_proba(model, Xva)
                pte = get_proba(model, Xte)
                mva = compute_metrics(yva, pva)
                mte = compute_metrics(yte, pte)
                all_rows.append(dict(dataset=name, model=mname, family=family,
                                     feature=feat, split="test",
                                     val_auc=mva["roc_auc"], n_train_used=n_used,
                                     **mte))
                preds_store[f"{mname}__{feat}"] = pte
                cap_note = f" [cap {n_used}]" if n_used < len(ytr) else ""
                tag = f"{mname}+{feat}"
                print(f"  {tag:22s} val_auc={mva['roc_auc']:.4f} | test {fmt_metrics(mte)}{cap_note}")
                if best is None or mva["roc_auc"] > best[0]:
                    best = (mva["roc_auc"], mname, feat, model, mte)

        # persist best fitted model + its config for SHAP/UMAP reuse
        _, bmodel, bfeat, fitted, bmte = best
        dump(fitted, MODELS_DIR / f"{name}__{bmodel}__{bfeat}.joblib")
        best_per_dataset[name] = dict(model=bmodel, feature=bfeat,
                                      val_auc=best[0], test=bmte)
        print(f"  >> BEST: {bmodel}+{bfeat}  test {fmt_metrics(bmte)}")

        # save test predictions + ground truth for this dataset
        np.savez(PREDS_DIR / f"classical_{name}.npz",
                 y_true=yte, **preds_store)

    out = pd.DataFrame(all_rows)
    out.to_csv(METRICS_DIR / "classical.csv", index=False)
    (METRICS_DIR / "classical_best.json").write_text(json.dumps(best_per_dataset, indent=2))
    print("\nSaved -> results/metrics/classical.csv ; classical_best.json")
    print(json.dumps(best_per_dataset, indent=2))


if __name__ == "__main__":
    run()
