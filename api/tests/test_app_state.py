"""Tests for AppState container."""
import json
import tempfile
from pathlib import Path

import pandas as pd
import pytest

from api.app_state import AppState, init_app_state, get_app_state


class TestAppState:
    """Tests for AppState dataclass."""

    def test_default_initialization(self):
        """AppState initializes with defaults."""
        state = AppState()
        assert state.anime_metadata == {}
        assert state.ratings_df is None
        assert state.anime_df is None
        assert state.anime_ids == []
        assert state.rbm is None
        assert state.is_initialized is False

    def test_model_loaded_property(self):
        """model_loaded reflects RBM presence."""
        state = AppState()
        assert state.model_loaded is False

        from api.inference.model import RBM
        state.rbm = RBM(n_visible=10, n_hidden=5)
        assert state.model_loaded is True

    def test_dataset_loaded_property(self):
        """dataset_loaded reflects anime_df presence."""
        state = AppState()
        assert state.dataset_loaded is False

        state.anime_df = pd.DataFrame({'anime_id': [1, 2, 3]})
        assert state.dataset_loaded is True

    def test_dataset_loaded_empty_df(self):
        """Empty DataFrame returns False for dataset_loaded."""
        state = AppState()
        state.anime_df = pd.DataFrame()
        assert state.dataset_loaded is False

    def test_metadata_loaded_property(self):
        """metadata_loaded reflects metadata presence."""
        state = AppState()
        assert state.metadata_loaded is False

        state.anime_metadata = {"1": {"name": "Test"}}
        assert state.metadata_loaded is True

    def test_get_metadata_string_key(self):
        """get_metadata works with string keys."""
        state = AppState()
        state.anime_metadata = {"123": {"image_url": "test.jpg"}}
        result = state.get_metadata(123)
        assert result["image_url"] == "test.jpg"

    def test_get_metadata_int_key(self):
        """get_metadata works with integer keys."""
        state = AppState()
        state.anime_metadata = {123: {"image_url": "test.jpg"}}
        result = state.get_metadata(123)
        assert result["image_url"] == "test.jpg"

    def test_get_metadata_missing(self):
        """get_metadata returns empty dict for missing ID."""
        state = AppState()
        result = state.get_metadata(999)
        assert result == {}

    def test_get_anime_info_from_metadata(self):
        """get_anime_info returns info from metadata."""
        state = AppState()
        state.anime_metadata = {"123": {"title": "Test Anime", "title_english": "Test EN", "image_url": "test.jpg"}}
        result = state.get_anime_info(123)
        assert result["name"] == "Test Anime"
        assert result["title_english"] == "Test EN"
        assert result["image_url"] == "test.jpg"

    def test_get_anime_info_fallback_to_dataframe(self):
        """get_anime_info falls back to DataFrame when metadata missing."""
        state = AppState()
        state.anime_df = pd.DataFrame({"anime_id": [1], "name": ["DF Anime"], "title_english": ["DF EN"]})
        result = state.get_anime_info(1)
        assert result["name"] == "DF Anime"
        assert result["title_english"] == "DF EN"

    def test_get_anime_info_missing(self):
        """get_anime_info returns empty dict for missing ID."""
        state = AppState()
        result = state.get_anime_info(999)
        assert result == {}

    def test_load_metadata_from_file(self):
        """load_metadata reads JSON file correctly."""
        state = AppState()
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump({"1": {"name": "Anime 1"}, "2": {"name": "Anime 2"}}, f)
            temp_path = Path(f.name)

        try:
            state.load_metadata(temp_path)
            assert len(state.anime_metadata) == 2
            assert state.anime_metadata["1"]["name"] == "Anime 1"
        finally:
            temp_path.unlink()

    def test_load_metadata_missing_file(self):
        """load_metadata handles missing file gracefully."""
        state = AppState()
        state.load_metadata(Path("/nonexistent/path.json"))
        assert state.anime_metadata == {}

    def test_load_anime_csv(self):
        """load_anime_csv reads CSV file correctly."""
        state = AppState()
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            f.write("anime_id,name\n1,Test Anime\n2,Another Anime\n")
            temp_path = Path(f.name)

        try:
            state.load_anime_csv(temp_path)
            assert len(state.anime_df) == 2
            assert state.anime_df.iloc[0]['name'] == "Test Anime"
        finally:
            temp_path.unlink()

    def test_load_anime_csv_missing_file(self):
        """load_anime_csv handles missing file gracefully."""
        state = AppState()
        state.load_anime_csv(Path("/nonexistent/path.csv"))
        assert state.anime_df is None


class TestHealthStatus:
    """Tests for health check status generation."""

    def test_all_services_ok(self):
        """Health status OK when all services loaded."""
        state = AppState()
        state.anime_df = pd.DataFrame({'anime_id': [1, 2, 3]})
        state.ratings_df = pd.DataFrame({'user_id': [1, 2], 'score': [8, 9]})
        state.anime_metadata = {"1": {}}
        from api.inference.model import RBM
        state.rbm = RBM(n_visible=10, n_hidden=5)

        status = state.get_health_status()
        assert status["status"] == "ok"
        assert status["services"]["model"]["status"] == "ok"
        assert status["services"]["dataset"]["status"] == "ok"
        assert status["services"]["metadata"]["status"] == "ok"

    def test_degraded_without_model(self):
        """Health status degraded without model."""
        state = AppState()
        state.anime_df = pd.DataFrame({'anime_id': [1]})
        state.anime_metadata = {"1": {}}

        status = state.get_health_status()
        assert status["status"] == "degraded"
        assert status["services"]["model"]["status"] == "error"

    def test_degraded_without_dataset(self):
        """Health status degraded without dataset."""
        state = AppState()
        state.anime_metadata = {"1": {}}
        from api.inference.model import RBM
        state.rbm = RBM(n_visible=10, n_hidden=5)

        status = state.get_health_status()
        assert status["status"] == "degraded"
        assert status["services"]["dataset"]["status"] == "error"

    def test_degraded_without_metadata(self):
        """Health status degraded without metadata."""
        state = AppState()
        state.anime_df = pd.DataFrame({'anime_id': [1]})
        from api.inference.model import RBM
        state.rbm = RBM(n_visible=10, n_hidden=5)

        status = state.get_health_status()
        assert status["status"] == "degraded"
        assert status["services"]["metadata"]["status"] == "error"

    def test_includes_counts(self):
        """Health status includes data counts."""
        state = AppState()
        state.anime_df = pd.DataFrame({'anime_id': [1, 2, 3]})
        state.ratings_df = pd.DataFrame({'user_id': [1, 2], 'score': [8, 9]})
        state.anime_metadata = {"1": {}, "2": {}}

        status = state.get_health_status()
        assert status["services"]["dataset"]["anime_count"] == 3
        assert status["services"]["dataset"]["rating_count"] == 2
        assert status["services"]["metadata"]["entry_count"] == 2


class TestGlobalAppState:
    """Tests for global AppState management."""

    def test_init_app_state(self):
        """init_app_state creates global instance."""
        state = init_app_state()
        assert isinstance(state, AppState)

    def test_get_app_state_after_init(self):
        """get_app_state returns initialized instance."""
        init_app_state()
        state = get_app_state()
        assert isinstance(state, AppState)
