"""Ensure triage model directory (``model.pkl`` + ``feature_spec.json``) exists before tests import ``app``."""

from __future__ import annotations

import json
import os
from pathlib import Path

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

_FIXTURE_DIR = Path(__file__).resolve().parent / "fixtures"

_FEATURE_SPEC: dict = {
    "version": 1,
    "target_column": "label_int",
    "target_class_map": {"low": 0, "medium": 1, "high": 2},
    "inverse_label_map": {"0": "low", "1": "medium", "2": "high"},
    "feature_columns": [
        "age",
        "sex_enc",
        "num_symptoms",
        "num_risk_factors",
        "has_chest_pain",
        "has_dyspnea",
        "has_neuro_deficit",
        "smoking",
        "hypertension",
        "diabetes",
    ],
    "binary_flag_columns": [
        "has_chest_pain",
        "has_dyspnea",
        "has_neuro_deficit",
        "smoking",
        "hypertension",
        "diabetes",
    ],
    "optional_columns_included": False,
    "notes": "Test fixture aligned with train_triage_model / build_training_table.",
}

_MODEL = _FIXTURE_DIR / "model.pkl"
_SPEC = _FIXTURE_DIR / "feature_spec.json"


def _ensure_triage_test_bundle() -> None:
    _FIXTURE_DIR.mkdir(parents=True, exist_ok=True)
    need_train = not _MODEL.is_file() or _MODEL.stat().st_size <= 80
    if need_train:
        X = np.array(
            [
                [-1, -1, 0, 0, 0, 0, 0, 0, 0, 0],
                [40, 1, 1, 0, 0, 0, 0, 0, 0, 0],
                [55, 0, 2, 1, 1, 0, 0, 0, 1, 0],
                [75, 1, 3, 3, 1, 1, 1, 1, 1, 1],
            ],
            dtype=np.float64,
        )
        y = np.array([0, 0, 1, 2], dtype=np.int64)
        pipe = Pipeline(
            [
                ("scaler", StandardScaler()),
                ("clf", LogisticRegression(max_iter=2000, random_state=0)),
            ]
        )
        pipe.fit(X, y)
        joblib.dump(
            {
                "model": pipe,
                "feature_names": list(_FEATURE_SPEC["feature_columns"]),
                "label_mapping": {"low": 0, "medium": 1, "high": 2},
                "label_names": ["low", "medium", "high"],
                "model_type": "logreg",
                "random_state": 0,
            },
            _MODEL,
        )
    if need_train or not _SPEC.is_file():
        _SPEC.write_text(json.dumps(_FEATURE_SPEC, indent=2), encoding="utf-8")


_ensure_triage_test_bundle()
os.environ.pop("SCORING_TRIAGE_MODEL_PATH", None)
os.environ["SCORING_TRIAGE_MODEL_DIR"] = str(_FIXTURE_DIR)
