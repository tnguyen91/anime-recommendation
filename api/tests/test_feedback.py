"""Tests for feedback endpoints."""
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, PROJECT_ROOT)

from api.database import Base, get_db
from api.models import User, RecommendationFeedback
from api.auth.security import get_password_hash, create_access_token


# Create in-memory SQLite database for testing
SQLALCHEMY_DATABASE_URL = "sqlite:///:memory:"

engine = create_engine(
    SQLALCHEMY_DATABASE_URL,
    connect_args={"check_same_thread": False},
    poolclass=StaticPool,
)
TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def override_get_db():
    """Override database dependency for testing."""
    db = TestingSessionLocal()
    try:
        yield db
    finally:
        db.close()


@pytest.fixture(scope="function")
def client():
    """Create test client with mocked dependencies."""
    # Create tables
    Base.metadata.create_all(bind=engine)
    
    # Import app after setting up database
    from api.main import app
    
    # Override dependencies
    app.dependency_overrides[get_db] = override_get_db
    
    # Mock the ML-related startup to avoid loading models
    with (
        patch('api.main.load_anime_dataset') as mock_load_data,
        patch('api.main.filter_data') as mock_filter,
        patch('api.main.RBM') as mock_rbm,
        patch('api.main.download_to_cache') as mock_download,
    ):
        import pandas as pd
        mock_ratings = pd.DataFrame({'user_id': [], 'anime_id': [], 'score': []})
        mock_anime_df = pd.DataFrame({
            'anime_id': [1, 2, 3],
            'name': ['Naruto', 'One Piece', 'Attack on Titan'],
            'title_english': ['Naruto', 'One Piece', 'Attack on Titan'],
            'title_japanese': ['ナルト', 'ワンピース', '進撃の巨人']
        })
        
        mock_load_data.return_value = (mock_ratings, mock_anime_df)
        mock_filter.return_value = ([1, 2, 3], mock_ratings)

        mock_path = MagicMock(spec=Path)
        mock_path.exists.return_value = False
        mock_download.return_value = mock_path

        rbm_instance = MagicMock()
        rbm_instance.to.return_value = rbm_instance
        mock_rbm.return_value = rbm_instance

        with TestClient(app) as test_client:
            yield test_client
    
    # Clean up
    Base.metadata.drop_all(bind=engine)
    app.dependency_overrides.clear()


@pytest.fixture
def test_user(client):
    """Create a test user in the database."""
    db = TestingSessionLocal()
    user = User(
        email="test@example.com",
        username="testuser",
        hashed_password=get_password_hash("testpassword123"),
        is_active=True
    )
    db.add(user)
    db.commit()
    db.refresh(user)
    db.close()
    return user


@pytest.fixture
def auth_headers(test_user):
    """Get authentication headers for the test user."""
    token = create_access_token(data={"sub": str(test_user.id), "email": test_user.email})
    return {"Authorization": f"Bearer {token}"}


@pytest.fixture
def test_feedback(client, test_user):
    """Create test feedback entries in the database."""
    db = TestingSessionLocal()
    feedback_items = [
        RecommendationFeedback(
            user_id=test_user.id,
            anime_id=1,
            action="favorited",
            recommendation_request_id="abc123",
            recorded_at=datetime.now(timezone.utc)
        ),
        RecommendationFeedback(
            user_id=test_user.id,
            anime_id=2,
            action="dismissed",
            recorded_at=datetime.now(timezone.utc)
        ),
        RecommendationFeedback(
            user_id=test_user.id,
            anime_id=3,
            action="watched",
            recorded_at=datetime.now(timezone.utc)
        ),
    ]
    db.add_all(feedback_items)
    db.commit()
    db.close()
    return feedback_items


class TestSubmitFeedback:
    """Tests for POST /api/v1/feedback"""
    
    def test_submit_feedback_authenticated(self, client, auth_headers):
        """Submit feedback as authenticated user."""
        response = client.post(
            "/api/v1/feedback",
            json={
                "anime_id": 1,
                "action": "favorited",
                "recommendation_request_id": "req123"
            },
            headers=auth_headers
        )
        assert response.status_code == 201
        data = response.json()
        assert data["anime_id"] == 1
        assert data["action"] == "favorited"
        assert "recorded_at" in data
        assert data["message"] == "Feedback recorded successfully"
    
    def test_submit_feedback_anonymous(self, client):
        """Submit feedback without authentication (anonymous)."""
        response = client.post(
            "/api/v1/feedback",
            json={
                "anime_id": 1,
                "action": "favorited"
            }
        )
        assert response.status_code == 201
        data = response.json()
        assert data["anime_id"] == 1
        assert data["action"] == "favorited"
    
    def test_submit_feedback_all_actions(self, client, auth_headers):
        """Test all valid feedback action types."""
        actions = ["favorited", "dismissed", "watched"]
        
        for i, action in enumerate(actions):
            response = client.post(
                "/api/v1/feedback",
                json={
                    "anime_id": i + 1,
                    "action": action
                },
                headers=auth_headers
            )
            assert response.status_code == 201
            assert response.json()["action"] == action
    
    def test_submit_feedback_invalid_action(self, client, auth_headers):
        """Submit feedback with invalid action type."""
        response = client.post(
            "/api/v1/feedback",
            json={
                "anime_id": 1,
                "action": "invalid_action"
            },
            headers=auth_headers
        )
        assert response.status_code == 422
    
    def test_submit_feedback_invalid_anime_id(self, client, auth_headers):
        """Submit feedback with invalid anime_id."""
        response = client.post(
            "/api/v1/feedback",
            json={
                "anime_id": -1,
                "action": "favorited"
            },
            headers=auth_headers
        )
        assert response.status_code == 422
    
    def test_submit_feedback_missing_fields(self, client, auth_headers):
        """Submit feedback with missing required fields."""
        response = client.post(
            "/api/v1/feedback",
            json={"anime_id": 1},  # missing action
            headers=auth_headers
        )
        assert response.status_code == 422


class TestBulkFeedback:
    """Tests for POST /api/v1/feedback/bulk"""
    
    def test_submit_bulk_feedback(self, client, auth_headers):
        """Submit multiple feedback items at once."""
        response = client.post(
            "/api/v1/feedback/bulk",
            json={
                "feedback_items": [
                    {"anime_id": 1, "action": "favorited"},
                    {"anime_id": 2, "action": "dismissed"},
                    {"anime_id": 3, "action": "watched"}
                ]
            },
            headers=auth_headers
        )
        assert response.status_code == 201
        data = response.json()
        assert data["recorded_count"] == 3
        assert "Successfully recorded 3 feedback items" in data["message"]
    
    def test_submit_bulk_feedback_anonymous(self, client):
        """Submit bulk feedback without authentication."""
        response = client.post(
            "/api/v1/feedback/bulk",
            json={
                "feedback_items": [
                    {"anime_id": 1, "action": "favorited"},
                    {"anime_id": 2, "action": "dismissed"}
                ]
            }
        )
        assert response.status_code == 201
        assert response.json()["recorded_count"] == 2
    
    def test_submit_bulk_feedback_empty_list(self, client, auth_headers):
        """Submit bulk feedback with empty list."""
        response = client.post(
            "/api/v1/feedback/bulk",
            json={"feedback_items": []},
            headers=auth_headers
        )
        assert response.status_code == 422


class TestFeedbackStats:
    """Tests for GET /api/v1/feedback/stats"""
    
    def test_get_feedback_stats(self, client, auth_headers, test_feedback):
        """Get feedback statistics for authenticated user."""
        response = client.get("/api/v1/feedback/stats", headers=auth_headers)
        assert response.status_code == 200
        data = response.json()
        assert data["total_feedback"] == 3
        assert "feedback_by_action" in data
        assert "top_favorited_anime" in data
        assert "top_dismissed_anime" in data
    
    def test_get_feedback_stats_empty(self, client, auth_headers):
        """Get stats when user has no feedback."""
        response = client.get("/api/v1/feedback/stats", headers=auth_headers)
        assert response.status_code == 200
        data = response.json()
        assert data["total_feedback"] == 0
    
    def test_get_feedback_stats_with_days_filter(self, client, auth_headers, test_feedback):
        """Get stats with custom day range."""
        response = client.get(
            "/api/v1/feedback/stats?days=7",
            headers=auth_headers
        )
        assert response.status_code == 200
    
    def test_get_feedback_stats_requires_auth(self, client):
        """Stats endpoint requires authentication."""
        response = client.get("/api/v1/feedback/stats")
        assert response.status_code == 401


class TestFeedbackHistory:
    """Tests for GET /api/v1/feedback/history"""
    
    def test_get_feedback_history(self, client, auth_headers, test_feedback):
        """Get feedback history for authenticated user."""
        response = client.get("/api/v1/feedback/history", headers=auth_headers)
        assert response.status_code == 200
        data = response.json()
        assert len(data["items"]) == 3
        assert data["total"] == 3
    
    def test_get_feedback_history_with_pagination(self, client, auth_headers, test_feedback):
        """Get feedback history with pagination."""
        response = client.get(
            "/api/v1/feedback/history?limit=2&offset=0",
            headers=auth_headers
        )
        assert response.status_code == 200
        data = response.json()
        assert len(data["items"]) == 2
        assert data["total"] == 3
        assert data["limit"] == 2
        assert data["offset"] == 0
    
    def test_get_feedback_history_filter_by_action(self, client, auth_headers, test_feedback):
        """Get feedback history filtered by action type."""
        response = client.get(
            "/api/v1/feedback/history?action=favorited",
            headers=auth_headers
        )
        assert response.status_code == 200
        data = response.json()
        assert len(data["items"]) == 1
        assert data["items"][0]["action"] == "favorited"
    
    def test_get_feedback_history_requires_auth(self, client):
        """History endpoint requires authentication."""
        response = client.get("/api/v1/feedback/history")
        assert response.status_code == 401
