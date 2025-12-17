"""
Application state container.

This module provides a type-safe container for all runtime application state,
including the ML model, datasets, and metadata. Using a class instead of
global variables provides:

1. Type safety - All attributes are typed and validated
2. Encapsulation - State is grouped logically and protected
3. Testability - Easy to create mock state for testing
4. Thread safety - Immutable after initialization (in production)
5. Clear lifecycle - State is created once at startup

Usage:
    # In lifespan function:
    state = AppState()
    state.load_all(settings)
    app.state.app_state = state

    # In endpoints (via dependency):
    def get_app_state(request: Request) -> AppState:
        return request.app.state.app_state
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
    from settings import Settings

logger = logging.getLogger(__name__)


@dataclass
class AppState:
    """
    Container for all application runtime state.

    Attributes:
        device: PyTorch device (CPU or CUDA) for model inference
        anime_metadata: Dictionary mapping anime_id to metadata dict
        ratings_df: DataFrame of user-anime ratings
        anime_df: DataFrame of anime information
        anime_ids: List of anime IDs in the filtered dataset
        rbm: The trained Restricted Boltzmann Machine model
        is_initialized: Whether all components have been loaded
    """

    # PyTorch device for model inference
    device: torch.device = field(
        default_factory=lambda: torch.device("cuda" if torch.cuda.is_available() else "cpu")
    )

    # Anime metadata from JSON file
    anime_metadata: dict[str, Any] = field(default_factory=dict)

    # DataFrames for ratings and anime info
    ratings_df: pd.DataFrame | None = None
    anime_df: pd.DataFrame | None = None

    # Filtered anime IDs used by the model
    anime_ids: list[int] = field(default_factory=list)

    # The RBM model instance
    rbm: "RBM | None" = None

    # Initialization flag
    is_initialized: bool = False

    def get_metadata(self, anime_id: int) -> dict[str, Any]:
        """
        Lookup metadata by anime_id, trying both str and int keys.

        Args:
            anime_id: The anime ID to look up

        Returns:
            Metadata dictionary, or empty dict if not found
        """
        return (
            self.anime_metadata.get(str(anime_id))
            or self.anime_metadata.get(anime_id, {})
        )

    @property
    def model_loaded(self) -> bool:
        """Check if the RBM model is loaded and ready."""
        return self.rbm is not None

    @property
    def dataset_loaded(self) -> bool:
        """Check if anime dataset is loaded."""
        return self.anime_df is not None and len(self.anime_df) > 0

    @property
    def metadata_loaded(self) -> bool:
        """Check if anime metadata is loaded."""
        return bool(self.anime_metadata)

    def load_metadata(self, metadata_path: Path) -> None:
        """
        Load anime metadata from a JSON file.

        Args:
            metadata_path: Path to the metadata JSON file
        """
        if metadata_path.exists():
            with metadata_path.open("r", encoding="utf-8") as f:
                self.anime_metadata = json.load(f)
            logger.info(f"Loaded metadata for {len(self.anime_metadata)} anime")
        else:
            logger.warning(f"Metadata file not found: {metadata_path}")

    def load_anime_csv(self, csv_path: Path) -> None:
        """
        Load anime data from a CSV file.

        Args:
            csv_path: Path to the anime CSV file
        """
        if csv_path.exists():
            self.anime_df = pd.read_csv(str(csv_path))
            logger.info(f"Loaded anime data: {len(self.anime_df)} entries")
        else:
            logger.warning(f"Anime CSV not found: {csv_path}")

    def get_health_status(self) -> dict[str, Any]:
        """
        Generate health check status for all components.

        Returns:
            Dictionary with status of model, dataset, and metadata
        """
        status = {
            "status": "ok",
            "version": "1.0.0",
            "services": {}
        }

        # Check model
        if self.model_loaded:
            status["services"]["model"] = {"status": "ok", "type": "RBM"}
        else:
            status["services"]["model"] = {"status": "error", "error": "Model not loaded"}
            status["status"] = "degraded"

        # Check dataset
        if self.dataset_loaded:
            status["services"]["dataset"] = {
                "status": "ok",
                "anime_count": len(self.anime_df),
                "rating_count": len(self.ratings_df) if self.ratings_df is not None else 0
            }
        else:
            status["services"]["dataset"] = {"status": "error", "error": "Dataset not loaded"}
            status["status"] = "degraded"

        # Check metadata
        if self.metadata_loaded:
            status["services"]["metadata"] = {
                "status": "ok",
                "entry_count": len(self.anime_metadata)
            }
        else:
            status["services"]["metadata"] = {"status": "error", "error": "Metadata not loaded"}
            status["status"] = "degraded"

        return status


# Singleton instance for direct imports (use sparingly - prefer dependency injection)
_app_state: AppState | None = None


def get_app_state() -> AppState:
    """
    Get the global AppState instance.

    This is a convenience function for cases where dependency injection
    isn't possible. Prefer using the FastAPI dependency pattern instead.

    Returns:
        The global AppState instance

    Raises:
        RuntimeError: If AppState hasn't been initialized
    """
    if _app_state is None:
        raise RuntimeError("AppState not initialized. Call init_app_state() first.")
    return _app_state


def init_app_state() -> AppState:
    """
    Initialize the global AppState instance.

    Should be called once during application startup.

    Returns:
        The newly created AppState instance
    """
    global _app_state
    _app_state = AppState()
    return _app_state
