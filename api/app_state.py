"""
Application runtime state container.

Encapsulates ML model, datasets, and metadata in a type-safe dataclass.
Accessed via FastAPI dependency injection for clean endpoint signatures.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pandas as pd
import torch

if TYPE_CHECKING:
    from api.inference.model import RBM

logger = logging.getLogger(__name__)


@dataclass
class AppState:
    """Runtime state container for ML model and datasets."""

    device: torch.device = field(
        default_factory=lambda: torch.device("cuda" if torch.cuda.is_available() else "cpu")
    )
    anime_metadata: dict[str, Any] = field(default_factory=dict)
    ratings_df: pd.DataFrame | None = None
    anime_df: pd.DataFrame | None = None
    anime_ids: list[int] = field(default_factory=list)
    rbm: "RBM | None" = None
    is_initialized: bool = False

    def get_metadata(self, anime_id: int) -> dict[str, Any]:
        """Lookup metadata by anime_id (handles both str and int keys)."""
        return (
            self.anime_metadata.get(str(anime_id))
            or self.anime_metadata.get(anime_id, {})
        )

    @property
    def model_loaded(self) -> bool:
        return self.rbm is not None

    @property
    def dataset_loaded(self) -> bool:
        return self.anime_df is not None and len(self.anime_df) > 0

    @property
    def metadata_loaded(self) -> bool:
        return bool(self.anime_metadata)

    def load_metadata(self, metadata_path: Path) -> None:
        """Load anime metadata from JSON file."""
        if metadata_path.exists():
            with metadata_path.open("r", encoding="utf-8") as f:
                self.anime_metadata = json.load(f)
            logger.info(f"Loaded metadata for {len(self.anime_metadata)} anime")
        else:
            logger.warning(f"Metadata file not found: {metadata_path}")

    def load_anime_csv(self, csv_path: Path) -> None:
        """Load anime data from CSV file."""
        if csv_path.exists():
            self.anime_df = pd.read_csv(str(csv_path))
            logger.info(f"Loaded anime data: {len(self.anime_df)} entries")
        else:
            logger.warning(f"Anime CSV not found: {csv_path}")

    def get_health_status(self) -> dict[str, Any]:
        """Generate health check response for all components."""
        status: dict[str, Any] = {
            "status": "ok",
            "version": "1.0.0",
            "services": {}
        }

        if self.model_loaded:
            status["services"]["model"] = {"status": "ok", "type": "RBM"}
        else:
            status["services"]["model"] = {"status": "error", "error": "Model not loaded"}
            status["status"] = "degraded"

        if self.dataset_loaded:
            status["services"]["dataset"] = {
                "status": "ok",
                "anime_count": len(self.anime_df),
                "rating_count": len(self.ratings_df) if self.ratings_df is not None else 0
            }
        else:
            status["services"]["dataset"] = {"status": "error", "error": "Dataset not loaded"}
            status["status"] = "degraded"

        if self.metadata_loaded:
            status["services"]["metadata"] = {
                "status": "ok",
                "entry_count": len(self.anime_metadata)
            }
        else:
            status["services"]["metadata"] = {"status": "error", "error": "Metadata not loaded"}
            status["status"] = "degraded"

        return status


_app_state: AppState | None = None


def get_app_state() -> AppState:
    """Get global AppState instance."""
    if _app_state is None:
        raise RuntimeError("AppState not initialized")
    return _app_state


def init_app_state() -> AppState:
    """Initialize global AppState instance at startup."""
    global _app_state
    _app_state = AppState()
    return _app_state
