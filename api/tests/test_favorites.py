"""Tests for favorites endpoints."""
import os
import sys
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
from api.models import User, UserFavorite
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
def test_favorite(client, test_user):
    """Create a test favorite in the database."""
    db = TestingSessionLocal()
    favorite = UserFavorite(
        user_id=test_user.id,
        anime_id=1  # Naruto
    )
    db.add(favorite)
    db.commit()
    db.refresh(favorite)
    db.close()
    return favorite


class TestListFavorites:
    """Tests for GET /api/v1/favorites"""
    
    def test_list_favorites_empty(self, client, auth_headers):
        """List favorites when user has none."""
        response = client.get("/api/v1/favorites", headers=auth_headers)
        assert response.status_code == 200
        data = response.json()
        assert data["favorites"] == []
        assert data["total"] == 0
    
    def test_list_favorites_with_items(self, client, auth_headers, test_favorite):
        """List favorites when user has some."""
        response = client.get("/api/v1/favorites", headers=auth_headers)
        assert response.status_code == 200
        data = response.json()
        assert data["total"] == 1
        assert len(data["favorites"]) == 1
        assert data["favorites"][0]["anime_id"] == 1
    
    def test_list_favorites_unauthorized(self, client):
        """List favorites without authentication."""
        response = client.get("/api/v1/favorites")
        assert response.status_code == 401


class TestAddFavorite:
    """Tests for POST /api/v1/favorites"""
    
    def test_add_favorite_success(self, client, auth_headers):
        """Successfully add an anime to favorites."""
        response = client.post(
            "/api/v1/favorites",
            json={"anime_id": 2},
            headers=auth_headers
        )
        assert response.status_code == 201
        data = response.json()
        assert data["anime_id"] == 2
        assert "id" in data
        assert "added_at" in data
    
    def test_add_favorite_duplicate(self, client, auth_headers, test_favorite):
        """Adding same anime twice should fail."""
        response = client.post(
            "/api/v1/favorites",
            json={"anime_id": 1},  # Same as test_favorite
            headers=auth_headers
        )
        assert response.status_code == 400
        assert "already in favorites" in response.json()["detail"]
    
    def test_add_favorite_invalid_anime_id(self, client, auth_headers):
        """Adding with invalid anime_id should fail."""
        response = client.post(
            "/api/v1/favorites",
            json={"anime_id": -1},
            headers=auth_headers
        )
        assert response.status_code == 422  # Validation error
    
    def test_add_favorite_unauthorized(self, client):
        """Adding favorite without authentication."""
        response = client.post(
            "/api/v1/favorites",
            json={"anime_id": 1}
        )
        assert response.status_code == 401


class TestRemoveFavorite:
    """Tests for DELETE /api/v1/favorites/{favorite_id}"""
    
    def test_remove_favorite_success(self, client, auth_headers, test_favorite):
        """Successfully remove a favorite by ID."""
        response = client.delete(
            f"/api/v1/favorites/{test_favorite.id}",
            headers=auth_headers
        )
        assert response.status_code == 200
        assert "removed successfully" in response.json()["message"]
        
        # Verify it's gone
        list_response = client.get("/api/v1/favorites", headers=auth_headers)
        assert list_response.json()["total"] == 0
    
    def test_remove_favorite_not_found(self, client, auth_headers):
        """Remove non-existent favorite."""
        response = client.delete(
            "/api/v1/favorites/99999",
            headers=auth_headers
        )
        assert response.status_code == 404
    
    def test_remove_favorite_unauthorized(self, client, test_favorite):
        """Remove favorite without authentication."""
        response = client.delete(f"/api/v1/favorites/{test_favorite.id}")
        assert response.status_code == 401


class TestRemoveFavoriteByAnime:
    """Tests for DELETE /api/v1/favorites/anime/{anime_id}"""
    
    def test_remove_by_anime_success(self, client, auth_headers, test_favorite):
        """Successfully remove a favorite by anime ID."""
        response = client.delete(
            "/api/v1/favorites/anime/1",
            headers=auth_headers
        )
        assert response.status_code == 200
        assert "removed successfully" in response.json()["message"]
    
    def test_remove_by_anime_not_found(self, client, auth_headers):
        """Remove anime not in favorites."""
        response = client.delete(
            "/api/v1/favorites/anime/99999",
            headers=auth_headers
        )
        assert response.status_code == 404


class TestCheckFavorite:
    """Tests for GET /api/v1/favorites/check/{anime_id}"""
    
    def test_check_favorite_true(self, client, auth_headers, test_favorite):
        """Check anime that is in favorites."""
        response = client.get(
            "/api/v1/favorites/check/1",
            headers=auth_headers
        )
        assert response.status_code == 200
        data = response.json()
        assert data["is_favorite"] is True
        assert data["anime_id"] == 1
    
    def test_check_favorite_false(self, client, auth_headers):
        """Check anime that is not in favorites."""
        response = client.get(
            "/api/v1/favorites/check/99999",
            headers=auth_headers
        )
        assert response.status_code == 200
        data = response.json()
        assert data["is_favorite"] is False
    
    def test_check_favorite_unauthorized(self, client):
        """Check favorite without authentication."""
        response = client.get("/api/v1/favorites/check/1")
        assert response.status_code == 401


class TestFavoriteIsolation:
    """Test that users can only see their own favorites."""
    
    def test_user_cannot_see_other_favorites(self, client, test_favorite):
        """User cannot see another user's favorites."""
        # Create second user
        db = TestingSessionLocal()
        user2 = User(
            email="user2@example.com",
            username="user2",
            hashed_password=get_password_hash("password123"),
            is_active=True
        )
        db.add(user2)
        db.commit()
        db.refresh(user2)
        db.close()
        
        # Get token for user2
        token = create_access_token(data={"sub": str(user2.id), "email": user2.email})
        headers = {"Authorization": f"Bearer {token}"}
        
        # User2 should see empty favorites (test_favorite belongs to test_user)
        response = client.get("/api/v1/favorites", headers=headers)
        assert response.status_code == 200
        assert response.json()["total"] == 0
    
    def test_user_cannot_delete_other_favorites(self, client, test_favorite):
        """User cannot delete another user's favorites."""
        # Create second user
        db = TestingSessionLocal()
        user2 = User(
            email="user3@example.com",
            username="user3",
            hashed_password=get_password_hash("password123"),
            is_active=True
        )
        db.add(user2)
        db.commit()
        db.refresh(user2)
        db.close()
        
        # Get token for user2
        token = create_access_token(data={"sub": str(user2.id), "email": user2.email})
        headers = {"Authorization": f"Bearer {token}"}
        
        # User2 trying to delete test_user's favorite should get 404
        response = client.delete(
            f"/api/v1/favorites/{test_favorite.id}",
            headers=headers
        )
        assert response.status_code == 404
