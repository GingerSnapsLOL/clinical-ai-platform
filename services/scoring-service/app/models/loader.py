"""
Generic loading for per-target model directories under ``models/<target_id>/``.

Expected layout::

    models/triage_severity/
        model.pkl          # joblib dict with at least ``model`` (estimator) and ``feature_names``
        feature_spec.json  # training feature contract (versioned)
        metrics.json       # optional evaluation snapshot

Future targets (``stroke_risk``, ``diabetes_risk``) use the same layout and env overrides.
"""

from __future__ import annotations

import json
import logging
import math
import os
from dataclasses import dataclass
from pathlib import Path
from threading import Lock
from typing import Any, Mapping, Sequence

import joblib
import numpy as np

logger = logging.getLogger(__name__)

SUPPORTED_FEATURE_SPEC_VERSIONS: frozenset[int] = frozenset({1})

# Per-target env var for an explicit model directory (absolute or relative).
TARGET_MODEL_DIR_ENV: dict[str, str] = {
    "triage_severity": "SCORING_TRIAGE_MODEL_DIR",
    "stroke_risk": "SCORING_STROKE_RISK_MODEL_DIR",
    "diabetes_risk": "SCORING_DIABETES_RISK_MODEL_DIR",
}

# Legacy: single file path; parent directory is used for spec + metrics.
LEGACY_TRIAGE_MODEL_FILE_ENV = "SCORING_TRIAGE_MODEL_PATH"

MODELS_ROOT_ENV = "SCORING_MODELS_ROOT"
DEFAULT_MODELS_ROOT = Path("models")

MODEL_FILENAME = "model.pkl"
FEATURE_SPEC_FILENAME = "feature_spec.json"
METRICS_FILENAME = "metrics.json"


class ModelLoadError(RuntimeError):
    """Raised when artifacts are missing, incompatible, or malformed."""


class ModelValidationError(ValueError):
    """Raised when an input feature row fails spec validation."""


@dataclass(frozen=True)
class LoadedModelPackage:
    """In-memory view of a target's on-disk bundle."""

    target_id: str
    root_dir: Path
    spec_version: int
    feature_columns: tuple[str, ...]
    estimator: Any
    label_mapping: dict[str, int]
    label_names: tuple[str, ...]
    model_type: str
    feature_spec: dict[str, Any]
    metrics: dict[str, Any] | None
    raw_bundle: dict[str, Any]


_cache_lock = Lock()
_package_cache: dict[str, LoadedModelPackage] = {}


def resolve_model_directory(target_id: str) -> Path:
    """
    Resolve the directory containing ``model.pkl`` for ``target_id``.

    Precedence:
    1. Target-specific env (e.g. ``SCORING_TRIAGE_MODEL_DIR``).
    2. For ``triage_severity`` only: ``SCORING_TRIAGE_MODEL_PATH`` if it points to a file
       (parent directory is used).
    3. ``{SCORING_MODELS_ROOT}/{target_id}`` (default root ``models``).
    """
    env_dir = TARGET_MODEL_DIR_ENV.get(target_id)
    if env_dir:
        raw = os.getenv(env_dir, "").strip()
        if raw:
            return Path(raw).expanduser().resolve()

    if target_id == "triage_severity":
        legacy = os.getenv(LEGACY_TRIAGE_MODEL_FILE_ENV, "").strip()
        if legacy:
            p = Path(legacy).expanduser().resolve()
            if p.is_file():
                return p.parent
            if p.is_dir():
                return p

    root = Path(os.getenv(MODELS_ROOT_ENV, str(DEFAULT_MODELS_ROOT))).expanduser().resolve()
    return (root / target_id).resolve()


def _read_json(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ModelLoadError(f"{path} must contain a JSON object")
    return data


def load_feature_spec(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise ModelLoadError(f"feature spec not found: {path}")
    spec = _read_json(path)
    version = spec.get("version", 1)
    if not isinstance(version, int):
        raise ModelLoadError(f"{path}: invalid spec version type")
    if version not in SUPPORTED_FEATURE_SPEC_VERSIONS:
        raise ModelLoadError(
            f"{path}: unsupported feature_spec version {version!r}; "
            f"supported={sorted(SUPPORTED_FEATURE_SPEC_VERSIONS)}"
        )
    cols = spec.get("feature_columns")
    if not isinstance(cols, list) or not cols:
        raise ModelLoadError(f"{path}: feature_columns must be a non-empty list")
    return spec


def _synthetic_spec_from_bundle(bundle: dict[str, Any]) -> dict[str, Any]:
    names = bundle.get("feature_names")
    if not isinstance(names, list) or not names:
        raise ModelLoadError("joblib bundle missing feature_names; cannot infer spec")
    return {
        "version": 1,
        "feature_columns": list(names),
        "target_column": "label_int",
        "notes": "Synthetic spec: feature_spec.json was missing; inferred from model.pkl",
    }


def _load_joblib_bundle(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise ModelLoadError(f"model file not found: {path}")
    bundle = joblib.load(path)
    if not isinstance(bundle, dict) or "model" not in bundle:
        raise ModelLoadError(f"{path}: expected joblib dict with 'model' key")
    return bundle


def _reconcile_columns(
    bundle_names: Sequence[str],
    spec_columns: Sequence[str],
    *,
    target_id: str,
) -> tuple[str, ...]:
    a = list(bundle_names)
    b = list(spec_columns)
    if a == b:
        return tuple(a)
    raise ModelLoadError(
        f"{target_id}: feature_names in model.pkl do not match feature_spec.json "
        f"feature_columns.\n  model.pkl: {a}\n  spec:      {b}"
    )


def validate_feature_row(
    values: Mapping[str, float],
    feature_columns: Sequence[str],
    *,
    target_id: str = "",
) -> None:
    """Ensure every required column is present and finite."""
    prefix = f"{target_id}: " if target_id else ""
    for col in feature_columns:
        if col not in values:
            raise ModelValidationError(f"{prefix}missing feature {col!r}")
        v = values[col]
        if isinstance(v, bool) or not isinstance(v, (int, float)):
            raise ModelValidationError(f"{prefix}feature {col!r} must be numeric")
        f = float(v)
        if not math.isfinite(f):
            raise ModelValidationError(f"{prefix}feature {col!r} must be finite")
    # Warn-only for unexpected keys could go here


def feature_vector_from_row(
    values: Mapping[str, float],
    feature_columns: Sequence[str],
    *,
    target_id: str = "",
) -> np.ndarray:
    validate_feature_row(values, feature_columns, target_id=target_id)
    vec = [float(values[c]) for c in feature_columns]
    return np.asarray(vec, dtype=np.float64).reshape(1, -1)


def load_target_model(target_id: str) -> LoadedModelPackage:
    """
    Load ``model.pkl``, ``feature_spec.json`` (or infer from bundle), and optional ``metrics.json``.
    """
    root = resolve_model_directory(target_id)
    model_path = root / MODEL_FILENAME
    spec_path = root / FEATURE_SPEC_FILENAME
    metrics_path = root / METRICS_FILENAME

    bundle = _load_joblib_bundle(model_path)
    bundle_names = bundle.get("feature_names")
    if not isinstance(bundle_names, list) or not bundle_names:
        raise ModelLoadError(f"{model_path}: bundle missing feature_names")

    if spec_path.is_file():
        feature_spec = load_feature_spec(spec_path)
        spec_columns = [str(x) for x in feature_spec["feature_columns"]]
        columns = _reconcile_columns(bundle_names, spec_columns, target_id=target_id)
    else:
        logger.warning(
            "feature_spec_missing_using_bundle",
            extra={"target_id": target_id, "dir": str(root)},
        )
        feature_spec = _synthetic_spec_from_bundle(bundle)
        columns = _reconcile_columns(bundle_names, feature_spec["feature_columns"], target_id=target_id)

    metrics: dict[str, Any] | None = None
    if metrics_path.is_file():
        try:
            metrics = _read_json(metrics_path)
        except (OSError, json.JSONDecodeError, ModelLoadError) as exc:
            logger.warning(
                "metrics_load_failed",
                extra={"target_id": target_id, "path": str(metrics_path), "error": str(exc)},
            )

    label_mapping_raw = bundle.get("label_mapping")
    label_mapping: dict[str, int] = (
        {str(k): int(v) for k, v in label_mapping_raw.items()}
        if isinstance(label_mapping_raw, dict)
        else {}
    )
    ln = bundle.get("label_names")
    if isinstance(ln, list) and len(ln) >= 3:
        label_names: tuple[str, ...] = tuple(str(x) for x in ln[:3])
    else:
        label_names = ("low", "medium", "high")

    model_type = str(bundle.get("model_type", "unknown"))
    estimator = bundle["model"]

    pkg = LoadedModelPackage(
        target_id=target_id,
        root_dir=root,
        spec_version=int(feature_spec["version"]),
        feature_columns=columns,
        estimator=estimator,
        label_mapping=label_mapping,
        label_names=label_names if label_names else ("low", "medium", "high"),
        model_type=model_type,
        feature_spec=feature_spec,
        metrics=metrics,
        raw_bundle=bundle,
    )
    logger.info(
        "target_model_loaded",
        extra={
            "target_id": target_id,
            "dir": str(root),
            "spec_version": pkg.spec_version,
            "n_features": len(columns),
            "model_type": model_type,
        },
    )
    return pkg


def load_target_model_cached(target_id: str) -> LoadedModelPackage:
    with _cache_lock:
        if target_id not in _package_cache:
            _package_cache[target_id] = load_target_model(target_id)
        return _package_cache[target_id]


def ensure_target_loaded(target_id: str) -> None:
    """Eager load (e.g. FastAPI lifespan)."""
    load_target_model_cached(target_id)


def clear_model_cache_for_tests() -> None:
    """Drop cached packages (tests only)."""
    with _cache_lock:
        _package_cache.clear()
