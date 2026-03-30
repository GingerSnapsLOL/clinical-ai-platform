"""Per-target scoring models."""

from app.targets.registry import (
    DEFAULT_PRIMARY_TARGET,
    TARGET_REGISTRY,
    get_target,
    valid_target_ids,
)

__all__ = [
    "DEFAULT_PRIMARY_TARGET",
    "TARGET_REGISTRY",
    "get_target",
    "valid_target_ids",
]
