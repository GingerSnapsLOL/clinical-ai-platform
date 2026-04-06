"""
Triage severity via on-disk ML bundle (``models/triage_severity/``).

Uses :mod:`app.models.loader` for ``model.pkl`` + ``feature_spec.json`` (+ optional ``metrics.json``).
"""

from __future__ import annotations

from typing import Any, Literal, cast

import numpy as np
from services.shared.schemas_v1 import FeatureContribution

from app.features import ExtractedFeatures
from app.models.loader import feature_vector_from_row, load_target_model_cached
from app.targets.base import TargetPrediction

SYMPTOM_SIGNAL_KEYS: frozenset[str] = frozenset(
    {
        "symptom_chest_pain",
        "symptom_dyspnea",
        "symptom_syncope",
        "neuro_focal_deficit",
        "neuro_speech",
        "infection_sepsis",
    }
)

RISK_SIGNAL_KEYS: frozenset[str] = frozenset(
    {
        "disease_hypertension",
        "disease_diabetes",
        "disease_copd",
        "disease_heart_failure",
        "disease_cad",
        "disease_stroke",
        "disease_ckd",
        "bp_keyword_elevated",
        "bp_systolic_elevated",
        "bp_diastolic_elevated",
        "age_older_adult",
        "bmi_obesity",
        "smoking_current",
        "anticoagulant",
    }
)

NEURO_DEFICIT_KEYS: frozenset[str] = frozenset(
    {"neuro_focal_deficit", "neuro_speech", "disease_stroke"}
)

TRIAGE_TARGET_ID = "triage_severity"


def _coerce_int(value: Any, default: int = -1) -> int:
    if value is None or isinstance(value, bool):
        return default
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    try:
        return int(str(value).strip())
    except (TypeError, ValueError):
        return default


def _encode_sex(value: Any) -> int:
    if value is None:
        return -1
    s = str(value).strip().lower()
    if s in ("male", "m", "man"):
        return 0
    if s in ("female", "f", "woman"):
        return 1
    if "non-binary" in s or s in ("nb", "nonbinary"):
        return 2
    return -1


def _sig(features: ExtractedFeatures, key: str) -> float:
    v = features.signals.get(key)
    if v is None:
        return 0.0
    return float(v)


def _bin_sig(features: ExtractedFeatures, key: str) -> int:
    return 1 if _sig(features, key) > 0.0 else 0


def _build_feature_map(features: ExtractedFeatures) -> dict[str, float]:
    s = features.signals
    st = features.structured_features

    age = _coerce_int(st.get("age"), default=-1)

    sex_enc = _encode_sex(st.get("sex"))

    num_symptoms = sum(1 for k in SYMPTOM_SIGNAL_KEYS if s.get(k, 0.0) and float(s[k]) > 0.0)
    num_risk_factors = sum(1 for k in RISK_SIGNAL_KEYS if s.get(k, 0.0) and float(s[k]) > 0.0)

    has_chest_pain = _bin_sig(features, "symptom_chest_pain")
    has_dyspnea = _bin_sig(features, "symptom_dyspnea")
    has_neuro_deficit = int(
        any(float(s.get(k, 0.0) or 0.0) > 0.0 for k in NEURO_DEFICIT_KEYS)
    )
    smoking = _bin_sig(features, "smoking_current")
    hypertension = int(
        _bin_sig(features, "disease_hypertension")
        or _bin_sig(features, "bp_systolic_elevated")
        or _bin_sig(features, "bp_diastolic_elevated")
        or _bin_sig(features, "bp_keyword_elevated")
    )
    diabetes = _bin_sig(features, "disease_diabetes")

    out: dict[str, float] = {
        "age": float(age),
        "sex_enc": float(sex_enc),
        "num_symptoms": float(num_symptoms),
        "num_risk_factors": float(num_risk_factors),
        "has_chest_pain": float(has_chest_pain),
        "has_dyspnea": float(has_dyspnea),
        "has_neuro_deficit": float(has_neuro_deficit),
        "smoking": float(smoking),
        "hypertension": float(hypertension),
        "diabetes": float(diabetes),
    }

    note_len = st.get("note_text_length")
    if note_len is not None:
        out["text_length"] = float(_coerce_int(note_len, default=0))
    else:
        out["text_length"] = 0.0

    out["entity_count"] = float(features.entity_count)
    return out


def _severity_score_from_proba(estimator: Any, proba_row: np.ndarray) -> float:
    classes = getattr(estimator, "classes_", None)
    if classes is None and hasattr(estimator, "named_steps"):
        clf = estimator.named_steps.get("clf")
        if clf is not None:
            classes = getattr(clf, "classes_", None)
    if classes is None:
        p = proba_row.reshape(-1)
        if p.size == 3:
            return float(0.5 * p[1] + 1.0 * p[2])
        return float(np.max(p))

    idx: dict[int, int] = {}
    for i, c in enumerate(classes):
        try:
            idx[int(c)] = i
        except (TypeError, ValueError):
            continue
    p = proba_row.reshape(-1)
    p1 = p[idx[1]] if 1 in idx else 0.0
    p2 = p[idx[2]] if 2 in idx else 0.0
    return float(0.5 * p1 + p2)


def _label_from_class(y_hat: int) -> str:
    names = ("low", "medium", "high")
    yi = int(y_hat)
    if 0 <= yi < len(names):
        return names[yi]
    return "low"


def _approximate_explanation(
    features: ExtractedFeatures,
    ml_score: float,
) -> list[FeatureContribution]:
    fired = {k: float(v) for k, v in features.signals.items() if float(v) > 0.0}
    if not fired or ml_score <= 0.0:
        return []
    raw = sum(fired.values())
    if raw <= 0.0:
        return []
    scale = ml_score / raw
    rows = [
        FeatureContribution(
            feature=f"triage_input:{k}",
            contribution=round(float(v) * scale, 6),
        )
        for k, v in sorted(fired.items(), key=lambda kv: (-kv[1], kv[0]))
    ]
    total = sum(r.contribution for r in rows)
    drift = round(ml_score - total, 6)
    if rows and abs(drift) >= 1e-6:
        rows[0] = FeatureContribution(
            feature=rows[0].feature,
            contribution=round(rows[0].contribution + drift, 6),
        )
    return rows


class TriageSeverityTarget:
    target_id = TRIAGE_TARGET_ID

    def predict(self, features: ExtractedFeatures) -> TargetPrediction:
        pack = load_target_model_cached(TRIAGE_TARGET_ID)
        estimator = pack.estimator
        row = _build_feature_map(features)
        X = feature_vector_from_row(row, pack.feature_columns, target_id=TRIAGE_TARGET_ID)

        y_hat = int(estimator.predict(X)[0])
        label = cast(
            Literal["low", "medium", "high"],
            _label_from_class(y_hat),
        )

        proba = estimator.predict_proba(X)[0]
        score = round(_severity_score_from_proba(estimator, proba), 6)
        score = float(min(1.0, max(0.0, score)))

        explanation = _approximate_explanation(features, score)

        return TargetPrediction(
            score=score,
            label=label,
            explanation=explanation,
            ready=True,
            detail=None,
        )
