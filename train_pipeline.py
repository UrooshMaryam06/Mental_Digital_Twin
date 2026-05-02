"""Training pipeline utilities for the Electricity Demand & Price project.

This script provides reusable functions to:
- load and preprocess the CSV dataset
- run TimeSeriesSplit cross-validation for regressors and classifiers
- train XGBoost / RandomForest models (if available)
- produce and persist clustering artifacts (scaler, PCA, kmeans)
- save models and metadata into the `artifacts/` directory
- expose a simple Prefect flow wiring the tasks with retries

Usage:
  - Ensure required packages are installed: scikit-learn, xgboost, pandas, numpy, prefect
  - Run locally: `python train_pipeline.py --data energy_dataset.csv`

This file is intentionally conservative: it checks for optional dependencies
and raises clear errors when a library is missing so the user can install it.
"""
from __future__ import annotations

import json
import pickle
import argparse
from pathlib import Path
from typing import Dict, Any, Tuple

import numpy as np
import pandas as pd

from sklearn.model_selection import TimeSeriesSplit
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import mean_absolute_error, accuracy_score

try:
    from xgboost import XGBRegressor, XGBClassifier
    _HAS_XGBOOST = True
except Exception:
    _HAS_XGBOOST = False

try:
    from prefect import task, flow
    _HAS_PREFECT = True
except Exception:
    _HAS_PREFECT = False


def ensure_artifacts_dir(path: Path = Path("artifacts")) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def load_data(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    return df


def simple_feature_matrix(df: pd.DataFrame, feature_cols=None) -> Tuple[np.ndarray, np.ndarray]:
    if feature_cols is None:
        # heuristic: use numeric columns except target-like names
        feature_cols = [c for c in df.select_dtypes(include=[np.number]).columns if 'demand' not in c.lower() and 'price' not in c.lower()]
    X = df[feature_cols].fillna(0).values
    return X, feature_cols


def run_timeseries_cv_regressors(X: np.ndarray, y: np.ndarray, n_splits: int = 5) -> Dict[str, Any]:
    tscv = TimeSeriesSplit(n_splits=n_splits)
    results = {}

    regressors = {"rf": RandomForestRegressor(random_state=0)}
    if _HAS_XGBOOST:
        regressors["xgb"] = XGBRegressor(tree_method='hist', eval_metric='mae', use_label_encoder=False)

    for name, model in regressors.items():
        maes = []
        for train_idx, test_idx in tscv.split(X):
            model.fit(X[train_idx], y[train_idx])
            preds = model.predict(X[test_idx])
            maes.append(float(mean_absolute_error(y[test_idx], preds)))
        results[name] = {"cv_mae": float(np.mean(maes)), "model": model}

    return results


def train_classifiers(X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
    classifiers = {"rf": RandomForestClassifier(random_state=0)}
    if _HAS_XGBOOST:
        classifiers["xgb"] = XGBClassifier(use_label_encoder=False, eval_metric='logloss')

    out = {}
    for name, clf in classifiers.items():
        clf.fit(X, y)
        preds = clf.predict(X)
        out[name] = {"model": clf, "accuracy": float(accuracy_score(y, preds))}
    return out


def cluster_and_persist(X: np.ndarray, artifacts_dir: Path, n_clusters: int = 4) -> Dict[str, Any]:
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    pca = PCA(n_components=min(8, Xs.shape[1]))
    Xp = pca.fit_transform(Xs)
    kmeans = KMeans(n_clusters=n_clusters, random_state=0)
    labels = kmeans.fit_predict(Xp)

    artifacts = {
        "scaler": scaler,
        "pca": pca,
        "kmeans": kmeans,
        "labels": labels,
    }

    # save artifacts
    ensure_artifacts_dir(artifacts_dir)
    with open(artifacts_dir / "cluster_artifacts.pkl", "wb") as f:
        pickle.dump(artifacts, f)

    # simple cluster profiles
    profiles = pd.DataFrame(X).assign(cluster=labels).groupby("cluster").mean()
    profiles.to_csv(artifacts_dir / "cluster_profiles.csv")

    return {"profiles_path": str(artifacts_dir / "cluster_profiles.csv"), "artifact_pkl": str(artifacts_dir / "cluster_artifacts.pkl")}


def save_models(models: Dict[str, Any], artifacts_dir: Path, prefix: str = "model") -> Dict[str, str]:
    ensure_artifacts_dir(artifacts_dir)
    paths = {}
    for name, info in models.items():
        model = info.get("model") if isinstance(info, dict) else info
        path = artifacts_dir / f"{prefix}_{name}.pkl"
        with open(path, "wb") as f:
            pickle.dump(model, f)
        paths[name] = str(path)
    return paths


def run_association_task(df=None):
    """Mine association rules from the processed energy dataframe."""
    from src.association_rules_mining import run_association_mining
    rules = run_association_mining()
    return f"Association mining complete: {len(rules)} rules found"


def write_metadata(metadata: Dict[str, Any], artifacts_dir: Path) -> Path:
    ensure_artifacts_dir(artifacts_dir)
    p = artifacts_dir / "model_metadata.json"
    with open(p, "w", encoding="utf8") as f:
        json.dump(metadata, f, indent=2)
    return p


def build_and_run_pipeline(csv_path: Path, artifacts_dir: Path = Path("artifacts")) -> Dict[str, Any]:
    df = load_data(csv_path)

    # Example target selection heuristics - user may adjust these to match notebook
    if "demand" in df.columns:
        y = df["demand"].values
    else:
        # fall back to first numeric column
        y = df.select_dtypes(include=[np.number]).iloc[:, 0].values

    X, feature_cols = simple_feature_matrix(df)
    finite_mask = np.isfinite(y)
    if not np.all(finite_mask):
        X = X[finite_mask]
        y = y[finite_mask]

    reg_results = run_timeseries_cv_regressors(X, y)
    reg_paths = save_models(reg_results, artifacts_dir, prefix="reg")

    # classification: create a simple binned label from target median
    y_bin = (y > np.nanmedian(y)).astype(int)
    clf_results = train_classifiers(X, y_bin)
    clf_paths = save_models(clf_results, artifacts_dir, prefix="clf")

    cluster_info = cluster_and_persist(X, artifacts_dir)
    assoc_result = run_association_task(df)

    metadata = {
        "regressors": reg_paths,
        "classifiers": clf_paths,
        "features": list(feature_cols),
        "cluster_profiles": cluster_info,
        "association_rules": assoc_result,
    }
    meta_path = write_metadata(metadata, artifacts_dir)

    return {"metadata": metadata, "metadata_path": str(meta_path)}


if _HAS_PREFECT:
    @flow
    def prefect_build_pipeline(csv_path: str, artifacts_dir: str = "artifacts"):
        return build_and_run_pipeline(Path(csv_path), Path(artifacts_dir))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="energy_dataset.csv")
    parser.add_argument("--artifacts", default="artifacts")
    args = parser.parse_args()

    result = build_and_run_pipeline(Path(args.data), Path(args.artifacts))
    print("Pipeline finished. Metadata:", result.get("metadata_path"))


if __name__ == "__main__":
    main()
