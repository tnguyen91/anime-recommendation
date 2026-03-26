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
            app_state = api.app.state.app_state
            app_state.anime_df = mock_anime_df.copy()
            app_state.rbm = rbm_instance
            app_state.anime_ids = list(mock_user_anime.columns)
            app_state.anime_metadata = {
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
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] in {"ok", "degraded"}
    assert "model" in data["services"]
    assert "dataset" in data["services"]


def test_search_anime_empty_query(client):
    response = client.get("/api/v1/search-anime?query=")
    assert response.status_code == 200
    payload = response.json()
    assert payload["results"] == []
    assert payload["total"] == 0
    assert payload["limit"] == 20
    assert payload["offset"] == 0


def test_search_anime_missing_query(client):
    response = client.get("/api/v1/search-anime")
    assert response.status_code == 200
    payload = response.json()
    assert payload["results"] == []
    assert payload["total"] == 0


def test_search_anime_valid_query(client):
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
    response = client.post("/api/v1/recommend", json={"liked_anime": []})
    assert response.status_code == 422


def test_recommend_endpoint_invalid_anime(client):
    response = client.post("/api/v1/recommend", json={"liked_anime": ["Nonexistent Anime"]})
    assert response.status_code == 400
    assert "No matching anime" in response.json()["error"]["message"]


def test_search_anime_query_too_long(client):
    long_query = "a" * 101
    response = client.get(f"/api/v1/search-anime?query={long_query}")
    assert response.status_code == 422


def test_search_anime_pagination(client):
    response = client.get("/api/v1/search-anime?query=sample&limit=1")
    assert response.status_code == 200
    payload = response.json()
    assert payload["limit"] == 1
    assert len(payload["results"]) <= 1

    response = client.get("/api/v1/search-anime?query=sample&limit=200")
    assert response.status_code == 422

    response = client.get("/api/v1/search-anime?query=sample&offset=-1")
    assert response.status_code == 422


def test_recommend_endpoint_success(client):
    with patch('api.recommendations.service.get_recommendations') as mock_get_recs:
        mock_get_recs.return_value = pd.DataFrame([
            {"anime_id": 2, "name": "Another Title", "score": 0.9}
        ])

        response = client.post("/api/v1/recommend", json={"liked_anime": ["Sample Anime"]})
        assert response.status_code == 200
        payload = response.json()
        assert payload["recommendations"][0]["anime_id"] == 2
        mock_get_recs.assert_called_once()


def test_recommend_endpoint_with_exclude_ids(client):
    with patch('api.recommendations.service.get_recommendations') as mock_get_recs:
        mock_get_recs.return_value = pd.DataFrame([
            {"anime_id": 2, "name": "Another Title", "score": 0.9}
        ])

        response = client.post("/api/v1/recommend", json={
            "liked_anime": ["Sample Anime"],
            "exclude_ids": [1, 3, 5]
        })
        assert response.status_code == 200

        call_kwargs = mock_get_recs.call_args
        assert call_kwargs.kwargs["exclude_ids"] == [1, 3, 5]


def test_recommend_endpoint_exclude_ids_over_limit(client):
    response = client.post("/api/v1/recommend", json={
        "liked_anime": ["Sample Anime"],
        "exclude_ids": list(range(300))
    })
    assert response.status_code == 422


def test_recommend_endpoint_empty_exclude_ids(client):
    with patch('api.recommendations.service.get_recommendations') as mock_get_recs:
        mock_get_recs.return_value = pd.DataFrame([
            {"anime_id": 2, "name": "Another Title", "score": 0.9}
        ])

        response = client.post("/api/v1/recommend", json={
            "liked_anime": ["Sample Anime"],
            "exclude_ids": []
        })
        assert response.status_code == 200

        call_kwargs = mock_get_recs.call_args
        assert call_kwargs.kwargs["exclude_ids"] == []


def test_recommend_endpoint_top_n_over_limit(client):
    response = client.post("/api/v1/recommend", json={
        "liked_anime": ["Sample Anime"],
        "top_n": 100
    })
    assert response.status_code == 422


def test_recommend_response_includes_request_id(client):
    with patch('api.recommendations.service.get_recommendations') as mock_get_recs:
        mock_get_recs.return_value = pd.DataFrame([
            {"anime_id": 2, "name": "Another Title", "score": 0.9}
        ])

        response = client.post("/api/v1/recommend", json={"liked_anime": ["Sample Anime"]})
        assert response.status_code == 200
        assert "request_id" in response.json()
        assert "X-Request-ID" in response.headers
