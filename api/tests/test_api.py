import os
import sys
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest
from fastapi.testclient import TestClient

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, PROJECT_ROOT)

import api.main as api

@pytest.fixture()
def client():
    with (
        patch('api.main.load_anime_dataset') as mock_load_data,
        patch('api.main.filter_data') as mock_filter,
        patch('api.main.RBM') as mock_rbm,
    ):
        mock_ratings = pd.DataFrame({'user_id': [], 'anime_id': [], 'score': []})
        mock_anime_df = pd.DataFrame({
            'anime_id': [1, 2],
            'name': ['Sample Anime', 'Another Title'],
            'title_english': ['Sample Anime', 'Another Title'],
            'title_japanese': ['サンプル', '別タイトル']
        })
        mock_user_anime = pd.DataFrame(columns=[1, 2])

        mock_load_data.return_value = (mock_ratings, mock_anime_df)
        mock_filter.return_value = (list(mock_user_anime.columns), mock_ratings)

        rbm_instance = MagicMock()
        rbm_instance.n_visible = len(mock_user_anime.columns)
        rbm_instance.n_hidden = 4
        rbm_instance.to.return_value = rbm_instance
        mock_rbm.return_value = rbm_instance

        with TestClient(api.app) as test_client:
            api.anime_df = mock_anime_df.copy()
            api.ratings_df = mock_ratings.copy()
            api.user_anime = mock_user_anime.copy()
            api.anime_ids = list(mock_user_anime.columns)
            api.rbm = rbm_instance
            api.anime_metadata = {
                "1": {
                    "image_url": "https://example.com/sample.jpg",
                    "genres": ["Action"],
                    "synopsis": "Sample synopsis"
                },
                1: {
                    "image_url": "https://example.com/sample.jpg",
                    "genres": ["Action"],
                    "synopsis": "Sample synopsis"
                }
            }
            yield test_client

def test_health_endpoint(client):
    """Test the health check endpoint"""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] in {"ok", "degraded"}
    assert "model" in data["services"]
    assert "dataset" in data["services"]

def test_search_anime_empty_query(client):
    """Empty query string returns no results"""
    response = client.get("/api/v1/search-anime?query=")
    assert response.status_code == 200
    payload = response.json()
    assert payload["results"] == []
    assert payload["total"] == 0
    assert payload["limit"] == 20
    assert payload["offset"] == 0

def test_search_anime_missing_query(client):
    """Omitted query parameter defaults to empty string."""
    response = client.get("/api/v1/search-anime")
    assert response.status_code == 200
    payload = response.json()
    assert payload["results"] == []
    assert payload["total"] == 0

def test_search_anime_valid_query(client):
    """Query returns enriched metadata with pagination."""
    response = client.get("/api/v1/search-anime?query=sample")
    assert response.status_code == 200
    payload = response.json()
    assert payload["results"]
    assert payload["total"] >= 1
    assert "limit" in payload
    assert "offset" in payload
    first = payload["results"][0]
    assert first["anime_id"] == 1
    assert first["image_url"] == "https://example.com/sample.jpg"
    assert first["genre"] == ["Action"]

def test_recommend_endpoint_no_liked_anime(client):
    """Test recommend endpoint with empty liked_anime list"""
    response = client.post("/api/v1/recommend", json={"liked_anime": []})
    assert response.status_code == 400
    assert "liked_anime must be a non-empty list" in response.json()["detail"]

def test_recommend_endpoint_invalid_anime(client):
    """Test recommend endpoint with anime that doesn't exist"""
    response = client.post("/api/v1/recommend", json={"liked_anime": ["Nonexistent Anime"]})
    assert response.status_code == 400
    assert "No matching anime found" in response.json()["detail"]

def test_search_anime_query_too_long(client):
    """Test search endpoint with query that's too long"""
    long_query = "a" * 101
    response = client.get(f"/api/v1/search-anime?query={long_query}")
    assert response.status_code == 400
    assert "Query too long" in response.json()["detail"]

def test_search_anime_invalid_chars(client):
    """Dangerous characters should be rejected."""
    response = client.get("/api/v1/search-anime?query=<script>")
    assert response.status_code == 400
    assert "Invalid characters" in response.json()["detail"]

def test_search_anime_pagination(client):
    """Test pagination parameters work correctly."""
    response = client.get("/api/v1/search-anime?query=sample&limit=1")
    assert response.status_code == 200
    payload = response.json()
    assert payload["limit"] == 1
    assert len(payload["results"]) <= 1
    
    response = client.get("/api/v1/search-anime?query=sample&limit=200")
    assert response.status_code == 400
    assert "Limit must be between 1 and 100" in response.json()["detail"]
    
    response = client.get("/api/v1/search-anime?query=sample&offset=-1")
    assert response.status_code == 400
    assert "Offset must be non-negative" in response.json()["detail"]

def test_recommend_endpoint_success(client):
    """Valid liked anime returns recommendation payload."""
    with patch('api.main.get_recommendations') as mock_get_recommendations:
        mock_get_recommendations.return_value = pd.DataFrame([
            {"anime_id": 2, "name": "Another Title", "score": 0.9}
        ])

        response = client.post("/api/v1/recommend", json={"liked_anime": ["Sample Anime"]})
        assert response.status_code == 200
        payload = response.json()
        assert payload["recommendations"][0]["anime_id"] == 2
        mock_get_recommendations.assert_called_once()

def test_recommend_endpoint_with_exclude_ids(client):
    """Exclude IDs are passed to the recommendation function."""
    with patch('api.main.get_recommendations') as mock_get_recommendations:
        mock_get_recommendations.return_value = pd.DataFrame([
            {"anime_id": 2, "name": "Another Title", "score": 0.9}
        ])

        response = client.post("/api/v1/recommend", json={
            "liked_anime": ["Sample Anime"],
            "exclude_ids": [1, 3, 5]
        })
        assert response.status_code == 200

        call_kwargs = mock_get_recommendations.call_args
        assert call_kwargs.kwargs["exclude_ids"] == [1, 3, 5]

def test_recommend_endpoint_exclude_ids_limit(client):
    """Exclude IDs are limited to 200 items."""
    with patch('api.main.get_recommendations') as mock_get_recommendations:
        mock_get_recommendations.return_value = pd.DataFrame([
            {"anime_id": 2, "name": "Another Title", "score": 0.9}
        ])

        large_exclude_list = list(range(300))
        response = client.post("/api/v1/recommend", json={
            "liked_anime": ["Sample Anime"],
            "exclude_ids": large_exclude_list
        })
        assert response.status_code == 200

        call_kwargs = mock_get_recommendations.call_args
        assert len(call_kwargs.kwargs["exclude_ids"]) == 200

def test_recommend_endpoint_empty_exclude_ids(client):
    """Empty exclude_ids list works correctly."""
    with patch('api.main.get_recommendations') as mock_get_recommendations:
        mock_get_recommendations.return_value = pd.DataFrame([
            {"anime_id": 2, "name": "Another Title", "score": 0.9}
        ])

        response = client.post("/api/v1/recommend", json={
            "liked_anime": ["Sample Anime"],
            "exclude_ids": []
        })
        assert response.status_code == 200

        call_kwargs = mock_get_recommendations.call_args
        assert call_kwargs.kwargs["exclude_ids"] == []

def test_recommend_endpoint_top_n_limit(client):
    """Top N is limited to 50."""
    with patch('api.main.get_recommendations') as mock_get_recommendations:
        mock_get_recommendations.return_value = pd.DataFrame([
            {"anime_id": 2, "name": "Another Title", "score": 0.9}
        ])

        response = client.post("/api/v1/recommend", json={
            "liked_anime": ["Sample Anime"],
            "top_n": 100
        })
        assert response.status_code == 200

        call_kwargs = mock_get_recommendations.call_args
        assert call_kwargs.kwargs["top_n"] == 50
