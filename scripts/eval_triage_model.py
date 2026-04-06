#!/usr/bin/env python3
"""Evaluate a triage-severity model (``model.pkl``) on a parquet dataset.

Prints macro F1, per-class recall, confusion matrix, and feature importance when available.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, f1_score, recall_score
from sklearn.pipeline import Pipeline

DEFAULT_LABEL_ORDER = [0, 1, 2]


def _load_artifact(path: Path) -> dict[str, Any]:
    obj = joblib.load(path)
    if not isinstance(obj, dict) or "model" not in obj:
        raise SystemExit(
            f"Expected joblib dict with 'model' key (from train_triage_model.py), got {type(obj)}"
        )
    return obj


def _label_names(artifact: dict[str, Any]) -> list[str]:
    names = artifact.get("label_names")
    if isinstance(names, list) and len(names) >= 3:
        return [str(x) for x in names[:3]]
    mapping = artifact.get("label_mapping") or {}
    inv = {int(v): str(k) for k, v in mapping.items() if str(k).strip()}
    return [inv.get(i, str(i)) for i in DEFAULT_LABEL_ORDER]


def _prepare_xy(
    df: pd.DataFrame,
    feature_names: list[str],
) -> tuple[np.ndarray, np.ndarray]:
    missing = [c for c in feature_names if c not in df.columns]
    if missing:
        raise SystemExit(f"Dataset missing columns: {missing}")
    if "label_int" not in df.columns:
        raise SystemExit("Dataset must contain label_int column.")
    X = df[feature_names].to_numpy(dtype=np.float64)
    y = df["label_int"].to_numpy(dtype=np.int64)
    return X, y


def _feature_importance(
    estimator: Any,
    feature_names: list[str],
    *,
    top_k: int,
) -> list[tuple[str, float]] | None:
    """Return (name, score) sorted descending, or None if unavailable."""

    def _from_matrix(coef: np.ndarray) -> np.ndarray:
        # (n_classes, n_features) → importance per feature
        return np.mean(np.abs(coef), axis=0)

    est = estimator
    if isinstance(est, Pipeline):
        if "clf" in est.named_steps:
            clf = est.named_steps["clf"]
            coef = getattr(clf, "coef_", None)
            if coef is not None:
                imp = _from_matrix(np.asarray(coef))
                pairs = sorted(
                    zip(feature_names, imp.tolist()),
                    key=lambda t: t[1],
                    reverse=True,
                )
                return pairs[:top_k]
        return None

    fi = getattr(est, "feature_importances_", None)
    if fi is not None:
        fi = np.asarray(fi, dtype=np.float64)
        pairs = sorted(
            zip(feature_names, fi.tolist()),
            key=lambda t: t[1],
            reverse=True,
        )
        return pairs[:top_k]

    return None


def _print_report(
    *,
    dataset_path: Path,
    model_path: Path,
    model_type: str,
    n_rows: int,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    label_order: list[int],
    label_names: list[str],
    importance: list[tuple[str, float]] | None,
    top_k: int,
) -> None:
    macro_f1 = float(
        f1_score(y_true, y_pred, average="macro", labels=label_order, zero_division=0)
    )
    recalls = recall_score(
        y_true, y_pred, average=None, labels=label_order, zero_division=0
    )
    cm = confusion_matrix(y_true, y_pred, labels=label_order)

    width = 56
    line = "=" * width

    print(line, flush=True)
    print("TRIAGE MODEL EVALUATION", flush=True)
    print(line, flush=True)
    print(f"  Model:      {model_path}", flush=True)
    print(f"  Model type: {model_type}", flush=True)
    print(f"  Dataset:    {dataset_path}", flush=True)
    print(f"  Rows:       {n_rows}", flush=True)
    print(flush=True)

    print("METRICS", flush=True)
    print("-" * width, flush=True)
    print(f"  Macro F1:   {macro_f1:.4f}", flush=True)
    print(flush=True)
    print("  Per-class recall (support-weighted over full set):", flush=True)
    for i, lab in enumerate(label_order):
        name = label_names[i] if i < len(label_names) else str(lab)
        r = float(recalls[i]) if i < len(recalls) else 0.0
        print(f"    {name:12} (y={lab}): {r:.4f}", flush=True)
    print(flush=True)

    print("  Confusion matrix [rows = true, cols = pred]:", flush=True)
    header = "              " + "".join(
        f"{label_names[j]:>12}" for j in range(min(len(label_names), len(label_order)))
    )
    print(header, flush=True)
    for i, lab in enumerate(label_order):
        name = label_names[i] if i < len(label_names) else str(lab)
        row = cm[i] if i < cm.shape[0] else np.zeros(len(label_order), dtype=int)
        cells = "".join(f"{int(row[j]):>12}" for j in range(len(label_order)))
        print(f"    true {name:8}{cells}", flush=True)
    print(line, flush=True)

    print("FEATURE IMPORTANCE", flush=True)
    print("-" * width, flush=True)
    if importance:
        print(f"  Top {min(top_k, len(importance))} features:", flush=True)
        for rank, (fname, score) in enumerate(importance, start=1):
            print(f"    {rank:2}. {fname:24} {score:.6f}", flush=True)
    else:
        print(
            "  (Not available for this estimator — logistic pipelines and "
            "XGBoost expose coefficients / gain-based importance.)",
            flush=True,
        )
    print(line, flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        type=Path,
        default=Path("models/triage_severity/model.pkl"),
        help="Joblib artifact from train_triage_model.py",
    )
    parser.add_argument(
        "dataset",
        type=Path,
        nargs="?",
        default=Path("data/processed/triage_train.parquet"),
        help="Parquet with same features as training + label_int",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=12,
        help="How many features to show when importance is available (default: 12)",
    )
    args = parser.parse_args()
    model_path: Path = args.model
    data_path: Path = args.dataset
    top_k = max(1, int(args.top_k))

    if not model_path.is_file():
        raise SystemExit(f"Model not found: {model_path}")
    if not data_path.is_file():
        raise SystemExit(f"Dataset not found: {data_path}")

    artifact = _load_artifact(model_path)
    estimator = artifact["model"]
    feature_names = artifact.get("feature_names")
    if not isinstance(feature_names, list) or not feature_names:
        raise SystemExit("Artifact missing feature_names list.")

    df = pd.read_parquet(data_path)
    X, y = _prepare_xy(df, feature_names)
    label_names = _label_names(artifact)
    model_type = str(artifact.get("model_type", "unknown"))

    try:
        y_pred = estimator.predict(X)
    except Exception as exc:
        raise SystemExit(f"predict() failed: {exc}") from exc

    importance = _feature_importance(estimator, feature_names, top_k=top_k)

    _print_report(
        dataset_path=data_path,
        model_path=model_path,
        model_type=model_type,
        n_rows=len(y),
        y_true=y,
        y_pred=y_pred,
        label_order=DEFAULT_LABEL_ORDER,
        label_names=label_names,
        importance=importance,
        top_k=top_k,
    )


if __name__ == "__main__":
    main()
