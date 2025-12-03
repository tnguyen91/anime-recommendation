"""Tests for authentication endpoints."""
import os
import sys
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, PROJECT_ROOT)

from api.database import Base, get_db
from api.models import User
from api.auth.security import get_password_hash


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
            'anime_id': [1, 2],
            'name': ['Sample Anime', 'Another Title'],
            'title_english': ['Sample Anime', 'Another Title'],
            'title_japanese': ['サンプル', '別タイトル']
        })
        
        mock_load_data.return_value = (mock_ratings, mock_anime_df)
        mock_filter.return_value = ([1, 2], mock_ratings)
        
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


class TestAuthRegister:
    """Tests for POST /api/v1/auth/register"""
    
    def test_register_success(self, client):
        """Successfully register a new user."""
        response = client.post(
            "/api/v1/auth/register",
            json={
                "email": "newuser@example.com",
                "password": "securepassword123",
                "username": "newuser"
            }
        )
        assert response.status_code == 201
        data = response.json()
        assert data["email"] == "newuser@example.com"
        assert data["username"] == "newuser"
        assert "id" in data
        assert "password" not in data  # Password should not be returned
    
    def test_register_duplicate_email(self, client, test_user):
        """Registering with existing email should fail."""
        response = client.post(
            "/api/v1/auth/register",
            json={
                "email": "test@example.com",  # Already exists
                "password": "anotherpassword123"
            }
        )
        assert response.status_code == 400
        assert "already registered" in response.json()["detail"]
    
    def test_register_short_password(self, client):
        """Password less than 8 characters should fail."""
        response = client.post(
            "/api/v1/auth/register",
            json={
                "email": "short@example.com",
                "password": "short"  # Too short
            }
        )
        assert response.status_code == 422  # Validation error
    
    def test_register_invalid_email(self, client):
        """Invalid email format should fail."""
        response = client.post(
            "/api/v1/auth/register",
            json={
                "email": "not-an-email",
                "password": "validpassword123"
            }
        )
        assert response.status_code == 422  # Validation error


class TestAuthLogin:
    """Tests for POST /api/v1/auth/login"""
    
    def test_login_success(self, client, test_user):
        """Successfully login with correct credentials."""
        response = client.post(
            "/api/v1/auth/login",
            data={  # OAuth2 uses form data, not JSON
                "username": "test@example.com",
                "password": "testpassword123"
            }
        )
        assert response.status_code == 200
        data = response.json()
        assert "access_token" in data
        assert data["token_type"] == "bearer"
    
    def test_login_wrong_password(self, client, test_user):
        """Login with wrong password should fail."""
        response = client.post(
            "/api/v1/auth/login",
            data={
                "username": "test@example.com",
                "password": "wrongpassword"
            }
        )
        assert response.status_code == 401
        assert "Incorrect email or password" in response.json()["detail"]
    
    def test_login_nonexistent_user(self, client):
        """Login with non-existent email should fail."""
        response = client.post(
            "/api/v1/auth/login",
            data={
                "username": "nobody@example.com",
                "password": "anypassword123"
            }
        )
        assert response.status_code == 401


class TestAuthMe:
    """Tests for GET /api/v1/auth/me"""
    
    def test_get_me_authenticated(self, client, test_user):
        """Get current user info with valid token."""
        # First login to get token
        login_response = client.post(
            "/api/v1/auth/login",
            data={
                "username": "test@example.com",
                "password": "testpassword123"
            }
        )
        token = login_response.json()["access_token"]
        
        # Use token to get user info
        response = client.get(
            "/api/v1/auth/me",
            headers={"Authorization": f"Bearer {token}"}
        )
        assert response.status_code == 200
        data = response.json()
        assert data["email"] == "test@example.com"
        assert data["username"] == "testuser"
    
    def test_get_me_no_token(self, client):
        """Accessing /me without token should fail."""
        response = client.get("/api/v1/auth/me")
        assert response.status_code == 401
    
    def test_get_me_invalid_token(self, client):
        """Accessing /me with invalid token should fail."""
        response = client.get(
            "/api/v1/auth/me",
            headers={"Authorization": "Bearer invalid-token"}
        )
        assert response.status_code == 401
