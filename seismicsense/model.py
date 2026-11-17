"""
model.py - the "Pattern Classifier" tool used by the agents.

A gradient-boosted tree ensemble learns to tell deep moonquakes, meteoroid impacts and
shallow moonquakes apart from catalog features. Artificial impacts are not learned
(only 9 exist) - the Mission Historian agent identifies them from Apollo mission records.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                             f1_score, precision_score, recall_score)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split

from . import catalog as C

MODEL_PATH = C.ROOT / "models" / "seismic_classifier.joblib"
METRICS_PATH = C.ROOT / "reports" / "metrics.json"
SEED = 42

# plain-English names for feature groups (used in the report card + LLM evidence)
FEATURE_GROUPS = {
    "Tidal timing (Earth's pull on the Moon)": ("earth_moon_km", "distance_rate", "anomalistic", "synodic", "draconic"),
    "Repeating 'fingerprint' (same source firing again)": ("recur_", "fp_"),
    "Shaking strength at each station": ("amp_", "logamp", "strength_x"),
    "Strength pattern between stations": ("ratio_",),
    "How long the shaking lasted": ("duration",),
    "Which sensors picked it up": ("seen_", "n_stations", "n_operating", "operating_", "ch_", "traces", "P_", "Q_"),
    "Signal quality grade": ("grade",),
    "How busy the Moon was that day": ("neighbors_",),
    "Time of day / season": ("hour_", "doy_"),
}


def feature_group(col: str) -> str:
    for g, keys in FEATURE_GROUPS.items():
        if any(col.startswith(k) or k in col for k in keys):
            return g
    return "Other"


def make_model() -> HistGradientBoostingClassifier:
    return HistGradientBoostingClassifier(
        max_iter=500, learning_rate=0.05, l2_regularization=1.0,
        class_weight="balanced", random_state=0,
    )


def train(verbose: bool = True) -> dict:
    t0 = time.time()
    cat = C.load_catalog()
    X = C.build_features(cat)
    mask = cat["nasa_label"].isin(C.NATURAL)
    Xn, yn = X[mask], cat.loc[mask, "nasa_label"].to_numpy()
    ids = cat.loc[mask, "event_id"].to_numpy()
    Xtr, Xte, ytr, yte, idtr, idte = train_test_split(Xn, yn, ids, test_size=0.2, stratify=yn, random_state=SEED)
    if verbose:
        print(f"Events: {len(cat):,} total | {mask.sum():,} labelled natural events | train {len(Xtr):,} / test {len(Xte):,}")

    cv = cross_val_score(make_model(), Xtr, ytr, cv=StratifiedKFold(5, shuffle=True, random_state=SEED),
                         scoring="f1_weighted", n_jobs=-1)
    model = make_model().fit(Xtr, ytr)
    pred = model.predict(Xte)
    classes = list(model.classes_)

    # ---- full agent pipeline incl. Mission Historian on the 9 artificial impacts
    art = cat["nasa_label"] == "artificial"
    hist = mission_historian(cat.loc[art, "time"])
    y_pipe = np.concatenate([yte, np.full(art.sum(), "artificial")])
    p_pipe = np.concatenate([pred, np.where(hist.notna(), "artificial", "unknown")])

    imp = permutation_importance(model, Xte, yte, n_repeats=3, random_state=0, scoring="f1_weighted", n_jobs=-1)
    gimp = pd.Series(imp.importances_mean, index=Xte.columns).groupby(feature_group).sum()
    gimp = (gimp.clip(lower=0) / gimp.clip(lower=0).sum()).sort_values(ascending=False)

    rep = classification_report(yte, pred, output_dict=True, zero_division=0)
    metrics = {
        "dataset": "NASA PDS Apollo Seismic Event Catalog (levent.1008weber, Nakamura et al.)",
        "n_events_total": int(len(cat)),
        "n_train": int(len(Xtr)), "n_test": int(len(Xte)),
        "classes": classes,
        "test_accuracy": accuracy_score(yte, pred),
        "test_f1_weighted": f1_score(yte, pred, average="weighted"),
        "test_f1_macro": f1_score(yte, pred, average="macro"),
        "test_precision_weighted": precision_score(yte, pred, average="weighted", zero_division=0),
        "test_recall_weighted": recall_score(yte, pred, average="weighted", zero_division=0),
        "cv5_f1_weighted_mean": float(cv.mean()), "cv5_f1_weighted_std": float(cv.std()),
        "per_class": {c: {k: rep[c][k] for k in ("precision", "recall", "f1-score", "support")} for c in classes},
        "confusion_matrix": confusion_matrix(yte, pred, labels=classes).tolist(),
        "pipeline_4class": {
            "accuracy": accuracy_score(y_pipe, p_pipe),
            "f1_weighted": f1_score(y_pipe, p_pipe, average="weighted"),
            "artificial_found": int(hist.notna().sum()), "artificial_total": int(art.sum()),
        },
        "feature_group_importance": gimp.round(4).to_dict(),
        "train_seconds": round(time.time() - t0, 1),
    }
    joblib.dump({"model": model, "features": list(X.columns), "classes": classes,
                 "test_ids": list(idte), "train_ids": list(idtr)}, MODEL_PATH, compress=3)
    METRICS_PATH.parent.mkdir(exist_ok=True)
    METRICS_PATH.write_text(json.dumps(metrics, indent=2))
    if verbose:
        print(classification_report(yte, pred, digits=3))
        print(f"Test accuracy {metrics['test_accuracy']:.3f} | weighted F1 {metrics['test_f1_weighted']:.3f} | "
              f"5-fold CV weighted F1 {cv.mean():.3f} ± {cv.std():.3f}")
        print(f"Full pipeline (4 classes, with Mission Historian): accuracy {metrics['pipeline_4class']['accuracy']:.3f}")
        print(f"Saved {MODEL_PATH.name} and metrics.json in {metrics['train_seconds']} s")
    return metrics


def load_model() -> dict:
    return joblib.load(MODEL_PATH)


# ---------------------------------------------------------------- Mission Historian tool
def mission_historian(times: pd.Series) -> pd.Series:
    """Return the name of the Apollo hardware impact matching each event time, else NaN.
    Seismic waves arrive seconds-minutes after impact; the catalog lists start minute."""
    ai = C.load_locations()["artificial"]
    out = pd.Series(np.nan, index=times.index, dtype=object)
    for _, r in ai.iterrows():
        dt = (times - r["time"]) / pd.Timedelta("1min")
        out[(dt >= -5) & (dt <= 30)] = r["name"]
    return out
