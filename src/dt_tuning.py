#!/usr/bin/env python
"""
Decision-Tree hyperparameter tuning — lightweight IDS on CICIDS2017 (11 classes).
MSc dissertation: Lightweight ML for Network Intrusion Detection.

Companion to src/lr_tuning.py. Uses the IDENTICAL data handling so DT and LR
results are comparable:
  * selects EXACTLY the 68 clean features from results/feature_sets.json['all']
    (no Destination Port, no duplicate/junk columns);
  * cleans the raw 15-class Label to the 11-class target (drop Heartbleed +
    Infiltration, merge the three Web Attack variants).

Differences from the LR script:
  * classifier is a DecisionTree (scale-invariant -> no StandardScaler needed);
  * grid is over tree-shape params (max_depth, min_samples_split, min_samples_leaf);
  * no SMOTE by default -- your no-SMOTE DT was the stronger model (macro-F1 0.972
    vs 0.966), so the final tree is tuned without oversampling.

Saves results/dt_tuning_best.json so the chosen params are on disk permanently.

RUN (type in the terminal, don't use the Run button):
  $env:PYTHONUNBUFFERED="1"; python src/dt_tuning.py
"""

from __future__ import annotations

import json
import logging
import os
import pickle
import platform
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# CONFIG  (edit here, or override via environment variables)
# ---------------------------------------------------------------------------
def _env(name: str, default, cast=str):
    v = os.environ.get(name)
    return cast(v) if v is not None else default


DATA_PKL     = Path(_env("DT_DATA_PKL", "data/merged_clean.pkl"))
FEATURE_JSON = Path(_env("DT_FEATURE_JSON", "results/feature_sets.json"))
FEATURE_KEY  = _env("DT_FEATURE_KEY", "all")     # same 68 features as LR
LABEL_COL    = _env("DT_LABEL_COL", "Label")
RESULTS_DIR  = Path(_env("DT_RESULTS_DIR", "results"))

SUBSAMPLE_N  = _env("DT_SUBSAMPLE_N", 200_000, int)   # trees are cheap; can afford more
CV_FOLDS     = _env("DT_CV_FOLDS", 5, int)            # set 10 to match the notebook CV
RANDOM_STATE = _env("DT_RANDOM_STATE", 42, int)
N_JOBS       = _env("DT_N_JOBS", 1, int)              # in-process: safe on Windows/8 GB

# Label cleaning (methodology: 15 raw classes -> 11)
EXCLUDE_CLASSES   = ["Heartbleed", "Infiltration"]
WEB_ATTACK_MERGED = "Web Attack"
EXPECTED_CLASSES  = 11

# DT grid (matches notebook 04)
PARAM_GRID = {
    "clf__max_depth":         [int(x) for x in _env("DT_MAX_DEPTH", "10,20,30").split(",")],
    "clf__min_samples_split": [int(x) for x in _env("DT_MIN_SPLIT", "2,10").split(",")],
    "clf__min_samples_leaf":  [int(x) for x in _env("DT_MIN_LEAF", "1,5").split(",")],
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def setup_logging() -> logging.Logger:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[logging.FileHandler(RESULTS_DIR / "dt_tuning.log", mode="w"),
                  logging.StreamHandler(sys.stdout)],
        force=True,
    )
    return logging.getLogger("dt_tuning")


def load_data(log: logging.Logger):
    if not DATA_PKL.exists():
        raise FileNotFoundError(f"Data pickle not found: {DATA_PKL.resolve()}")
    if not FEATURE_JSON.exists():
        raise FileNotFoundError(f"Feature JSON not found: {FEATURE_JSON.resolve()}")

    log.info("Loading %s ...", DATA_PKL)
    df = pd.read_pickle(DATA_PKL)
    log.info("Pickle shape: %s columns, %s rows", df.shape[1], df.shape[0])

    if LABEL_COL not in df.columns:
        raise KeyError(f"Label column {LABEL_COL!r} not in pickle.")
    n_before = len(df)
    df = df[~df[LABEL_COL].isin(EXCLUDE_CLASSES)].copy()
    web_mask = df[LABEL_COL].astype(str).str.contains("Web Attack", case=False, na=False)
    df.loc[web_mask, LABEL_COL] = WEB_ATTACK_MERGED
    n_classes = df[LABEL_COL].nunique()
    log.info("Label cleaning: dropped %d rows (%s); merged %d web-attack rows -> %r",
             n_before - len(df), EXCLUDE_CLASSES, int(web_mask.sum()), WEB_ATTACK_MERGED)
    log.info("Classes after cleaning: %d (expected %d)", n_classes, EXPECTED_CLASSES)
    assert n_classes == EXPECTED_CLASSES, (
        f"Expected {EXPECTED_CLASSES} classes, got {n_classes}: {sorted(df[LABEL_COL].unique())}")

    feats = json.loads(FEATURE_JSON.read_text())[FEATURE_KEY]
    log.info("FEATURE SET = %s%d (%d features)", FEATURE_KEY, len(feats), len(feats))
    missing = [c for c in feats if c not in df.columns]
    if missing:
        raise KeyError(f"{len(missing)} feature(s) in JSON missing from pickle: {missing[:10]}")

    X = df[feats].copy()
    y = df[LABEL_COL].copy()
    assert "Destination Port" not in X.columns, "LEAK: Destination Port in X!"
    assert X.shape[1] == len(feats)

    X = X.apply(pd.to_numeric, errors="coerce").astype(np.float32)
    n_nan = int(np.isnan(X.values).sum())
    if n_nan:
        log.warning("%d NaN cells after coercion; filling 0.0", n_nan)
        X = X.fillna(0.0)
    log.info("X ready: %s dtype=%s | y classes=%d", X.shape, X.values.dtype, y.nunique())
    return X, y


def stratified_subsample(X, y, n, log):
    from sklearn.model_selection import train_test_split
    if len(y) <= n:
        log.info("No subsample needed (%d <= %d)", len(y), n)
        return X.values, y.values
    X_sub, _, y_sub, _ = train_test_split(
        X, y, train_size=n, stratify=y, random_state=RANDOM_STATE)
    log.info("Stratified subsample -> %d rows", len(y_sub))
    log.info("Subsample class distribution:\n%s",
             pd.Series(y_sub).value_counts().sort_index().to_string())
    return X_sub.values, y_sub.values


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> int:
    log = setup_logging()
    t0 = time.time()
    log.info("=" * 70)
    log.info("DT TUNING START | %s", datetime.now().isoformat(timespec="seconds"))
    log.info("python=%s on %s", platform.python_version(), platform.platform())
    log.info("subsample=%s cv=%s n_jobs=%s grid=%s",
             SUBSAMPLE_N, CV_FOLDS, N_JOBS, PARAM_GRID)

    from sklearn.tree import DecisionTreeClassifier
    from sklearn.pipeline import Pipeline
    from sklearn.model_selection import GridSearchCV, StratifiedKFold

    X, y = load_data(log)
    Xs, ys = stratified_subsample(X, y, SUBSAMPLE_N, log)
    del X, y

    # Pipeline wrapper keeps 'clf__' grid keys consistent with notebook 04.
    pipe = Pipeline([("clf", DecisionTreeClassifier(random_state=RANDOM_STATE))])
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)

    n_cand = int(np.prod([len(v) for v in PARAM_GRID.values()]))
    log.info("GridSearch: %d candidates x %d folds = %d fits", n_cand, CV_FOLDS, n_cand * CV_FOLDS)

    gs = GridSearchCV(pipe, PARAM_GRID, scoring="f1_macro", cv=cv,
                      n_jobs=N_JOBS, verbose=2, refit=True, error_score=np.nan)
    gs.fit(Xs, ys)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(gs.cv_results_).to_csv(RESULTS_DIR / "dt_tuning_cvresults.csv", index=False)
    with open(RESULTS_DIR / "dt_tuning_gridsearch.pkl", "wb") as f:
        pickle.dump(gs, f)
    best = {
        "model": "DecisionTree",
        "best_params": gs.best_params_,
        "best_f1_macro": float(gs.best_score_),
        "feature_key": FEATURE_KEY,
        "subsample_n": SUBSAMPLE_N,
        "cv_folds": CV_FOLDS,
        "random_state": RANDOM_STATE,
        "elapsed_sec": round(time.time() - t0, 1),
        "finished": datetime.now().isoformat(timespec="seconds"),
    }
    (RESULTS_DIR / "dt_tuning_best.json").write_text(json.dumps(best, indent=2))

    log.info("BEST f1_macro=%.4f  params=%s", gs.best_score_, gs.best_params_)
    log.info("Saved: dt_tuning_{cvresults.csv,gridsearch.pkl,best.json}")
    log.info("DONE in %.1f min", (time.time() - t0) / 60)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception:
        logging.getLogger("dt_tuning").error("FATAL:\n%s", traceback.format_exc())
        sys.exit(1)
