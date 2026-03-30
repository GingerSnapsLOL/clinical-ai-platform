"""Service settings (target thresholds live on each target module)."""

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class Settings:
    scoring_service_port: int

    @classmethod
    def from_env(cls) -> "Settings":
        return cls(
            scoring_service_port=int(os.getenv("SCORING_SERVICE_PORT", "8050")),
        )


settings = Settings.from_env()
