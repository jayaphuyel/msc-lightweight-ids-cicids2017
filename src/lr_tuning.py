#!/usr/bin/env python
"""
LR hyperparameter tuning — lightweight IDS on CICIDS2017 (multiclass, 11 classes).
MSc dissertation: Lightweight ML for Network Intrusion Detection.

WHAT WAS WRONG BEFORE
---------------------
The previous run selected "all columns except Label", so the 79-col pickle
re-admitted 10 columns that must NOT be in X:
  * Destination Port          -> leaks the label (maps ~1:1 to attack type)
  * Fwd Header Length.1        -> duplicate column
  * Bwd PSH/URG Flags, Fwd/Bwd Avg Bytes|Packets|Bulk Rate -> near-constant junk
This script instead selects EXACTLY the 68 clean features listed in
results/feature_sets.json['all'], and asserts Destination Port is absent.

DESIGN (matches project decisions)
----------------------------------
  * Per-fold pipeline order:  StandardScaler -> [imbalance handling] -> LogisticRegression
    (scaler fit on the training split only; SMOTE interpolates in scaled space)
  * Imbalance strategy is a top-level switch so the SAME harness produces your
    notebook-03 comparison as a real evaluation axis:  smote | class_weight | none
  * saga solver (supports l1 & l2, scales to large n), capped max_iter, slightly
    loose tol for overnight speed.
  * 8 GB RAM guards: stratified subsample; float32; n_jobs=1 when SMOTE inflates
    each fold (n_jobs=2 otherwise); SMOTE oversamples minorities to a FLOOR, not
    parity, and skips any class too small to synthesise safely.
  * Crash-proof for an unattended run: logging to file + stdout with timestamps,
    verbose=2, cv_results_ saved to CSV, whole fitted GridSearch pickled, best
    params written to JSON, full traceback on failure, NaN-fold warning.

RUN DETACHED (PowerShell)
-------------------------
  Start-Process python -ArgumentList "src/lr_tuning.py" -NoNewWindow `
    -RedirectStandardOutput "results/lr_run.log" -RedirectStandardError "results/lr_run_err.log"

Config can be overridden by env vars (LR_STRATEGY, LR_SUBSAMPLE_N, LR_CV_FOLDS, ...)
without editing the file -- handy for a quick smoke test vs the full overnight run.
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


DATA_PKL     = Path(_env("LR_DATA_PKL", "data/merged_clean.pkl"))
FEATURE_JSON = Path(_env("LR_FEATURE_JSON", "results/feature_sets.json"))
FEATURE_KEY  = _env("LR_FEATURE_KEY", "all")     # LR needs all 68 (notebook 02)
LABEL_COL    = _env("LR_LABEL_COL", "Label")
RESULTS_DIR  = Path(_env("LR_RESULTS_DIR", "results"))

# ---- Label cleaning (methodology: 15 raw classes -> 11) --------------------
# merged_clean.pkl carries the RAW 15-class Label, so the target must be cleaned
# here to match the project decisions, exactly as the notebooks do:
#   * exclude the two too-rare classes
#   * merge the three web-attack variants into one
EXCLUDE_CLASSES   = ["Heartbleed", "Infiltration"]
WEB_ATTACK_MERGED = "Web Attack"          # <-- confirm this matches your notebook/report naming
EXPECTED_CLASSES  = 11

STRATEGY     = _env("LR_STRATEGY", "smote")      # "smote" | "class_weight" | "none"
SUBSAMPLE_N  = _env("LR_SUBSAMPLE_N", 200_000, int)
CV_FOLDS     = _env("LR_CV_FOLDS", 5, int)       # 5 = tractable; set 10 to match DT run
RANDOM_STATE = _env("LR_RANDOM_STATE", 42, int)

# SMOTE floor: bring each minority up to this count (never to full parity, never
# above the majority), and never a class too small to SMOTE (needs > k samples).
SMOTE_FLOOR       = _env("LR_SMOTE_FLOOR", 20_000, int)
SMOTE_K_NEIGHBORS = _env("LR_SMOTE_K", 5, int)

# Solver: lbfgs is fast + low-memory (L2 only). saga is slower but supports L1
# via l1_ratio. l1_ratio is only a valid parameter for saga, so the grid includes
# it only when SOLVER == 'saga'.
SOLVER   = _env("LR_SOLVER", "lbfgs")
_SAGA    = (SOLVER == "saga")

PARAM_GRID = {
    "clf__C": [float(x) for x in _env("LR_C_GRID", "0.01,0.1,1,10").split(",")],
}
if _SAGA:
    # l1_ratio=0.0 -> pure L2 (ridge)  |  l1_ratio=1.0 -> pure L1 (lasso, sparse)
    PARAM_GRID["clf__l1_ratio"] = [float(x) for x in _env("LR_L1RATIO_GRID", "0.0,1.0").split(",")]

LR_MAX_ITER = _env("LR_MAX_ITER", 1000, int)
LR_TOL      = _env("LR_TOL", 1e-3, float)         # slightly loose for speed; note in methods

# SMOTE inflates each fold's training set -> keep n_jobs=1; otherwise 2 is safe.
N_JOBS = 1 if STRATEGY == "smote" else _env("LR_N_JOBS", 2, int)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def setup_logging(tag: str) -> logging.Logger:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    logfile = RESULTS_DIR / f"lr_tuning_{tag}.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[logging.FileHandler(logfile, mode="w"),
                  logging.StreamHandler(sys.stdout)],
        force=True,
    )
    return logging.getLogger("lr_tuning")


class SmoteFloorStrategy:
    """Picklable callable for imblearn SMOTE.sampling_strategy, evaluated per fold.

    Oversamples each class that is (a) below `floor` and (b) has more than
    k_neighbors samples (SMOTE needs > k to interpolate), up to `floor` but
    never above the current majority. Classes at/above the floor, the majority,
    and any class too small to synthesise are left untouched.

    Implemented as a top-level class (not a closure) so the fitted GridSearch
    -- which holds a reference to it inside best_estimator_ -- pickles cleanly.
    """

    def __init__(self, floor: int, k_neighbors: int):
        self.floor = floor
        self.k_neighbors = k_neighbors

    def __call__(self, y):
        y = np.asarray(y)
        classes, counts = np.unique(y, return_counts=True)
        majority = int(counts.max())
        target = {}
        for c, n in zip(classes, counts):
            n = int(n)
            if n < self.floor and n > self.k_neighbors:
                target[c] = min(self.floor, majority)   # > n by construction
        return target


def build_pipeline(strategy: str, log: logging.Logger):
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression

    if strategy == "smote":
        from imblearn.pipeline import Pipeline          # sampler-aware pipeline
        from imblearn.over_sampling import SMOTE
        steps = [
            ("scaler", StandardScaler()),
            ("smote", SMOTE(
                sampling_strategy=SmoteFloorStrategy(SMOTE_FLOOR, SMOTE_K_NEIGHBORS),
                k_neighbors=SMOTE_K_NEIGHBORS,
                random_state=RANDOM_STATE,
            )),
            ("clf", LogisticRegression(
                solver=SOLVER, max_iter=LR_MAX_ITER, tol=LR_TOL,
                random_state=RANDOM_STATE,
            )),
        ]
    elif strategy in ("class_weight", "none"):
        from sklearn.pipeline import Pipeline
        cw = "balanced" if strategy == "class_weight" else None
        steps = [
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(
                solver=SOLVER, max_iter=LR_MAX_ITER, tol=LR_TOL,
                class_weight=cw, random_state=RANDOM_STATE,
            )),
        ]
    else:
        raise ValueError(f"Unknown STRATEGY={strategy!r} (use smote|class_weight|none)")

    log.info("Pipeline steps: %s", " -> ".join(name for name, _ in steps))
    return Pipeline(steps)


def load_data(log: logging.Logger):
    if not DATA_PKL.exists():
        raise FileNotFoundError(f"Data pickle not found: {DATA_PKL.resolve()}")
    if not FEATURE_JSON.exists():
        raise FileNotFoundError(f"Feature JSON not found: {FEATURE_JSON.resolve()}")

    log.info("Loading %s ...", DATA_PKL)
    df = pd.read_pickle(DATA_PKL)
    log.info("Pickle shape: %s columns, %s rows", df.shape[1], df.shape[0])

    # ---- LABEL CLEANING: raw 15-class Label -> 11-class target ----
    if LABEL_COL not in df.columns:
        raise KeyError(f"Label column {LABEL_COL!r} not in pickle. Columns: {list(df.columns)[:10]}...")
    n_before = len(df)
    df = df[~df[LABEL_COL].isin(EXCLUDE_CLASSES)].copy()
    web_mask = df[LABEL_COL].astype(str).str.contains("Web Attack", case=False, na=False)
    df.loc[web_mask, LABEL_COL] = WEB_ATTACK_MERGED
    n_classes = df[LABEL_COL].nunique()
    log.info("Label cleaning: dropped %d rows (%s); merged %d web-attack rows -> %r",
             n_before - len(df), EXCLUDE_CLASSES, int(web_mask.sum()), WEB_ATTACK_MERGED)
    log.info("Classes after cleaning: %d (expected %d)", n_classes, EXPECTED_CLASSES)
    assert n_classes == EXPECTED_CLASSES, (
        f"Expected {EXPECTED_CLASSES} classes after cleaning, got {n_classes}: "
        f"{sorted(df[LABEL_COL].unique())}"
    )

    feats = json.loads(FEATURE_JSON.read_text())[FEATURE_KEY]
    log.info("FEATURE SET = %s%d (%d features)", FEATURE_KEY, len(feats), len(feats))

    missing = [c for c in feats if c not in df.columns]
    if missing:
        raise KeyError(f"{len(missing)} feature(s) in JSON missing from pickle: {missing[:10]}")

    # ---- THE LEAK FIX: take EXACTLY the JSON features, not "all but Label" ----
    X = df[feats].copy()
    y = df[LABEL_COL].copy()

    # Loud guards so a leak can never slip through silently again.
    assert "Destination Port" not in X.columns, "LEAK: Destination Port is in X!"
    assert LABEL_COL not in X.columns, "LEAK: Label is in X!"
    assert X.shape[1] == len(feats), f"Expected {len(feats)} cols, got {X.shape[1]}"

    # Numeric + float32 for RAM headroom on 8 GB.
    X = X.apply(pd.to_numeric, errors="coerce").astype(np.float32)
    n_nan = int(np.isnan(X.values).sum())
    if n_nan:
        log.warning("%d NaN cells after numeric coercion; filling with 0.0", n_nan)
        X = X.fillna(0.0)

    log.info("X ready: %s  dtype=%s  | y classes=%d",
             X.shape, X.values.dtype, y.nunique())
    return X, y


def stratified_subsample(X, y, n, log):
    from sklearn.model_selection import train_test_split
    if len(y) <= n:
        log.info("No subsample needed (%d <= %d)", len(y), n)
        return X.values, y.values
    X_sub, _, y_sub, _ = train_test_split(
        X, y, train_size=n, stratify=y, random_state=RANDOM_STATE
    )
    log.info("Stratified subsample -> %d rows", len(y_sub))
    dist = pd.Series(y_sub).value_counts().sort_index()
    log.info("Subsample class distribution:\n%s", dist.to_string())
    return X_sub.values, y_sub.values


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> int:
    tag = STRATEGY
    log = setup_logging(tag)
    t0 = time.time()

    log.info("=" * 70)
    log.info("LR TUNING START | strategy=%s | %s", tag, datetime.now().isoformat(timespec="seconds"))
    log.info("python=%s sklearn+imblearn on %s", platform.python_version(), platform.platform())
    log.info("subsample=%s cv=%s n_jobs=%s solver=%s grid=%s max_iter=%s tol=%s",
             SUBSAMPLE_N, CV_FOLDS, N_JOBS, SOLVER, PARAM_GRID, LR_MAX_ITER, LR_TOL)
    if tag == "smote":
        log.info("SMOTE floor=%s k_neighbors=%s (minorities to floor, not parity)",
                 SMOTE_FLOOR, SMOTE_K_NEIGHBORS)

    from sklearn.model_selection import GridSearchCV, StratifiedKFold

    X, y = load_data(log)
    Xs, ys = stratified_subsample(X, y, SUBSAMPLE_N, log)
    del X, y  # free the full frame before the heavy fits

    pipe = build_pipeline(tag, log)
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)

    n_cand = int(np.prod([len(v) for v in PARAM_GRID.values()]))
    log.info("GridSearch: %d candidates x %d folds = %d fits", n_cand, CV_FOLDS, n_cand * CV_FOLDS)

    gs = GridSearchCV(
        pipe, PARAM_GRID, scoring="f1_macro", cv=cv,
        n_jobs=N_JOBS, verbose=2, refit=True,
        error_score=np.nan,   # one failed fold -> NaN, don't kill the whole run
    )
    gs.fit(Xs, ys)

    # ---- persist everything an unattended run might need in the morning ----
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    cvres = pd.DataFrame(gs.cv_results_)
    cvres.to_csv(RESULTS_DIR / f"lr_tuning_{tag}_cvresults.csv", index=False)

    with open(RESULTS_DIR / f"lr_tuning_{tag}_gridsearch.pkl", "wb") as f:
        pickle.dump(gs, f)

    best_l1 = gs.best_params_.get("clf__l1_ratio")
    reg = ("L1" if best_l1 == 1.0 else "L2" if best_l1 == 0.0
           else f"elasticnet({best_l1})" if best_l1 is not None
           else "L2")   # lbfgs/newton solvers are L2-only
    best = {
        "strategy": tag,
        "best_params": gs.best_params_,
        "regularization": reg,
        "best_f1_macro": float(gs.best_score_),
        "feature_key": FEATURE_KEY,
        "subsample_n": SUBSAMPLE_N,
        "cv_folds": CV_FOLDS,
        "elapsed_sec": round(time.time() - t0, 1),
        "finished": datetime.now().isoformat(timespec="seconds"),
    }
    (RESULTS_DIR / f"lr_tuning_{tag}_best.json").write_text(json.dumps(best, indent=2))

    n_nan = int(cvres["mean_test_score"].isna().sum())
    if n_nan:
        log.warning("%d/%d candidate(s) scored NaN -- a fold failed for those params.",
                    n_nan, len(cvres))

    log.info("BEST f1_macro=%.4f  params=%s", gs.best_score_, gs.best_params_)
    log.info("Saved: lr_tuning_%s_{cvresults.csv,gridsearch.pkl,best.json}", tag)
    log.info("DONE in %.1f min", (time.time() - t0) / 60)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception:
        logging.getLogger("lr_tuning").error("FATAL:\n%s", traceback.format_exc())
        sys.exit(1)
