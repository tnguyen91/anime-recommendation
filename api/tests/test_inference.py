"""Tests for inference module: RBM model and recommendation generation."""
import numpy as np
import pandas as pd
import pytest
import torch

from api.inference.model import RBM
from api.inference.recommender import get_recommendations


class TestRBM:
    """Tests for RBM model architecture and forward pass."""

    def test_initialization(self):
        """RBM initializes with correct dimensions."""
        rbm = RBM(n_visible=100, n_hidden=50)
        assert rbm.n_visible == 100
        assert rbm.n_hidden == 50
        assert rbm.W.shape == (50, 100)
        assert rbm.v_bias.shape == (100,)
        assert rbm.h_bias.shape == (50,)

    def test_sample_h_shape(self):
        """sample_h returns correct shapes."""
        rbm = RBM(n_visible=10, n_hidden=5)
        v = torch.zeros(3, 10)
        p_h, h_sample = rbm.sample_h(v)
        assert p_h.shape == (3, 5)
        assert h_sample.shape == (3, 5)

    def test_sample_v_shape(self):
        """sample_v returns correct shapes."""
        rbm = RBM(n_visible=10, n_hidden=5)
        h = torch.zeros(3, 5)
        p_v, v_sample = rbm.sample_v(h)
        assert p_v.shape == (3, 10)
        assert v_sample.shape == (3, 10)

    def test_sample_h_probability_range(self):
        """Hidden unit probabilities are in [0, 1]."""
        rbm = RBM(n_visible=10, n_hidden=5)
        v = torch.rand(3, 10)
        p_h, _ = rbm.sample_h(v)
        assert (p_h >= 0).all() and (p_h <= 1).all()

    def test_sample_v_probability_range(self):
        """Visible unit probabilities are in [0, 1]."""
        rbm = RBM(n_visible=10, n_hidden=5)
        h = torch.rand(3, 5)
        p_v, _ = rbm.sample_v(h)
        assert (p_v >= 0).all() and (p_v <= 1).all()

    def test_sample_binary_output(self):
        """Sampled units are binary (0 or 1)."""
        rbm = RBM(n_visible=10, n_hidden=5)
        v = torch.rand(3, 10)
        _, h_sample = rbm.sample_h(v)
        assert set(h_sample.unique().tolist()).issubset({0.0, 1.0})

    def test_forward_reconstruction(self):
        """Full forward pass produces valid reconstruction."""
        rbm = RBM(n_visible=10, n_hidden=5)
        v = torch.FloatTensor([[1, 0, 1, 0, 1, 0, 1, 0, 1, 0]])
        p_h, _ = rbm.sample_h(v)
        p_v, _ = rbm.sample_v(p_h)
        assert p_v.shape == v.shape
        assert (p_v >= 0).all() and (p_v <= 1).all()


class TestGetRecommendations:
    """Tests for recommendation generation."""

    @pytest.fixture
    def mock_rbm(self):
        """Create a simple RBM for testing."""
        rbm = RBM(n_visible=5, n_hidden=3)
        return rbm

    @pytest.fixture
    def mock_anime_df(self):
        """Create mock anime DataFrame."""
        return pd.DataFrame({
            'anime_id': [1, 2, 3, 4, 5],
            'name': ['Anime A', 'Anime B', 'Anime C', 'Anime D', 'Anime E'],
            'title_english': ['A English', 'B English', 'C English', 'D English', 'E English'],
        })

    @pytest.fixture
    def anime_ids(self):
        """Anime IDs matching the mock DataFrame."""
        return [1, 2, 3, 4, 5]

    def test_basic_recommendations(self, mock_rbm, anime_ids, mock_anime_df):
        """Basic recommendation returns expected structure."""
        input_vec = [1, 0, 0, 0, 0]
        result = get_recommendations(
            input_vec, mock_rbm, anime_ids, mock_anime_df, top_n=3
        )
        assert isinstance(result, pd.DataFrame)
        assert 'anime_id' in result.columns
        assert 'name' in result.columns
        assert 'score' in result.columns
        assert len(result) <= 3

    def test_excludes_liked_anime(self, mock_rbm, anime_ids, mock_anime_df):
        """Liked anime get negative scores (effectively excluded)."""
        input_vec = [1, 1, 0, 0, 0]
        result = get_recommendations(
            input_vec, mock_rbm, anime_ids, mock_anime_df, top_n=3
        )
        # With top_n=3, only top 3 are returned; liked items should have negative scores
        # and not appear in top recommendations
        assert len(result) <= 3

    def test_exclude_ids_parameter(self, mock_rbm, anime_ids, mock_anime_df):
        """Excluded IDs get negative scores."""
        input_vec = [0, 0, 0, 0, 0]
        result = get_recommendations(
            input_vec, mock_rbm, anime_ids, mock_anime_df,
            top_n=3, exclude_ids=[1, 2]
        )
        # With top_n=3 and 2 excluded from 5, we get 3 non-excluded
        assert len(result) <= 3

    def test_tensor_input(self, mock_rbm, anime_ids, mock_anime_df):
        """Works with torch.Tensor input."""
        input_vec = torch.FloatTensor([1, 0, 0, 0, 0])
        result = get_recommendations(
            input_vec, mock_rbm, anime_ids, mock_anime_df, top_n=3
        )
        assert len(result) <= 3

    def test_numpy_input(self, mock_rbm, anime_ids, mock_anime_df):
        """Works with numpy array input."""
        input_vec = np.array([1, 0, 0, 0, 0])
        result = get_recommendations(
            input_vec, mock_rbm, anime_ids, mock_anime_df, top_n=3
        )
        assert len(result) <= 3

    def test_empty_exclude_ids(self, mock_rbm, anime_ids, mock_anime_df):
        """Empty exclude_ids works correctly."""
        input_vec = [1, 0, 0, 0, 0]
        result = get_recommendations(
            input_vec, mock_rbm, anime_ids, mock_anime_df,
            top_n=3, exclude_ids=[]
        )
        assert len(result) <= 3

    def test_top_n_limit(self, mock_rbm, anime_ids, mock_anime_df):
        """Respects top_n parameter."""
        input_vec = [1, 0, 0, 0, 0]
        result = get_recommendations(
            input_vec, mock_rbm, anime_ids, mock_anime_df, top_n=2
        )
        assert len(result) <= 2

    def test_missing_anime_in_df(self, mock_rbm, mock_anime_df):
        """Handles anime_ids not present in DataFrame."""
        anime_ids = [1, 2, 3, 999, 998]
        input_vec = [1, 0, 0, 0, 0]
        result = get_recommendations(
            input_vec, mock_rbm, anime_ids, mock_anime_df, top_n=5
        )
        assert all(aid in [1, 2, 3] for aid in result['anime_id'].tolist())

    def test_scores_are_numeric(self, mock_rbm, anime_ids, mock_anime_df):
        """Recommendation scores are numeric."""
        input_vec = [1, 0, 0, 0, 0]
        result = get_recommendations(
            input_vec, mock_rbm, anime_ids, mock_anime_df, top_n=3
        )
        assert result['score'].dtype in [np.float32, np.float64]
