"""Tests for data preprocessing functions."""
import pandas as pd
import pytest

from api.inference.preprocess import filter_hentai, filter_data
from api.config import RATING_THRESHOLD


class TestFilterHentai:
    """Tests for adult content filtering."""

    def test_removes_hentai_anime(self):
        """Filters anime with Hentai in genre."""
        anime = pd.DataFrame({
            'anime_id': [1, 2, 3],
            'name': ['Safe Anime', 'Adult Content', 'Another Safe'],
            'genre': ['Action', 'Hentai', 'Comedy']
        })
        ratings = pd.DataFrame({
            'anime_id': [1, 2, 3],
            'user_id': [100, 100, 100],
            'score': [8, 9, 7]
        })

        filtered_ratings, filtered_anime = filter_hentai(ratings, anime)

        assert 2 not in filtered_anime['anime_id'].values
        assert 2 not in filtered_ratings['anime_id'].values
        assert len(filtered_anime) == 2
        assert len(filtered_ratings) == 2

    def test_case_insensitive_filter(self):
        """Filters regardless of case."""
        anime = pd.DataFrame({
            'anime_id': [1, 2],
            'name': ['Safe', 'Unsafe'],
            'genre': ['Action', 'HENTAI']
        })
        ratings = pd.DataFrame({
            'anime_id': [1, 2],
            'user_id': [100, 100],
            'score': [8, 9]
        })

        _, filtered_anime = filter_hentai(ratings, anime)
        assert len(filtered_anime) == 1

    def test_keeps_all_safe_anime(self):
        """Keeps all anime when none contain Hentai."""
        anime = pd.DataFrame({
            'anime_id': [1, 2, 3],
            'name': ['Anime A', 'Anime B', 'Anime C'],
            'genre': ['Action', 'Comedy', 'Drama']
        })
        ratings = pd.DataFrame({
            'anime_id': [1, 2, 3],
            'user_id': [100, 100, 100],
            'score': [8, 9, 7]
        })

        filtered_ratings, filtered_anime = filter_hentai(ratings, anime)

        assert len(filtered_anime) == 3
        assert len(filtered_ratings) == 3

    def test_checks_all_columns(self):
        """Filters if Hentai appears in any column."""
        anime = pd.DataFrame({
            'anime_id': [1, 2],
            'name': ['Hentai Prince', 'Normal Anime'],
            'genre': ['Comedy', 'Action']
        })
        ratings = pd.DataFrame({
            'anime_id': [1, 2],
            'user_id': [100, 100],
            'score': [8, 9]
        })

        _, filtered_anime = filter_hentai(ratings, anime)
        assert 1 not in filtered_anime['anime_id'].values


class TestFilterData:
    """Tests for data filtering based on activity thresholds."""

    @pytest.fixture
    def sample_ratings(self):
        """Create sample ratings DataFrame."""
        return pd.DataFrame({
            'user_id': [1, 1, 1, 2, 2, 3, 3, 3, 3],
            'anime_id': [10, 20, 30, 10, 20, 10, 20, 30, 40],
            'score': [8, 9, 6, 8, 7, 9, 8, 5, 10],
            'status': ['Completed'] * 9
        })

    def test_filters_by_min_likes_anime(self, sample_ratings):
        """Filters anime with insufficient likes."""
        active_anime, _ = filter_data(
            sample_ratings,
            min_likes_user=1,
            min_likes_anime=3
        )
        assert 10 in active_anime
        assert 20 in active_anime
        assert 40 not in active_anime

    def test_filters_by_min_likes_user(self):
        """Filters users with insufficient likes."""
        ratings = pd.DataFrame({
            'user_id': [1, 1, 1, 2, 3, 3, 3],
            'anime_id': [10, 20, 30, 10, 10, 20, 30],
            'score': [8, 9, 8, 6, 9, 8, 9],  # user 2 has score 6 (< 7), no likes
            'status': ['Completed'] * 7
        })
        _, filtered_df = filter_data(
            ratings,
            min_likes_user=2,
            min_likes_anime=1
        )
        # User 2 has 0 likes (score 6 < threshold 7), should be filtered
        assert 2 not in filtered_df['user_id'].values

    def test_only_completed_anime(self):
        """Only considers completed anime."""
        ratings = pd.DataFrame({
            'user_id': [1, 1, 1],
            'anime_id': [10, 10, 10],
            'score': [8, 9, 10],
            'status': ['Completed', 'Watching', 'Dropped']
        })
        active_anime, filtered = filter_data(ratings, min_likes_user=1, min_likes_anime=1)
        assert len(filtered) == 1

    def test_liked_threshold(self):
        """Respects RATING_THRESHOLD for likes."""
        ratings = pd.DataFrame({
            'user_id': [1, 1, 1],
            'anime_id': [10, 10, 10],
            'score': [RATING_THRESHOLD, RATING_THRESHOLD - 1, RATING_THRESHOLD + 1],
            'status': ['Completed', 'Completed', 'Completed']
        })
        _, filtered = filter_data(ratings, min_likes_user=1, min_likes_anime=1)
        liked_count = filtered['liked'].sum()
        assert liked_count == 2

    def test_returns_integer_anime_ids(self, sample_ratings):
        """Returns anime IDs as integers."""
        active_anime, _ = filter_data(
            sample_ratings,
            min_likes_user=1,
            min_likes_anime=1
        )
        assert all(isinstance(aid, int) for aid in active_anime)

    def test_empty_after_filtering(self):
        """Handles case where all data is filtered out."""
        ratings = pd.DataFrame({
            'user_id': [1],
            'anime_id': [10],
            'score': [5],
            'status': ['Completed']
        })
        active_anime, filtered = filter_data(
            ratings,
            min_likes_user=100,
            min_likes_anime=100
        )
        assert len(filtered) == 0
