"""Tests for settings validation."""
import os
import sys
import pytest
from pathlib import Path
from functools import lru_cache

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class _TestableSettings(BaseSettings):
    """Settings class for testing without loading from env."""

    model_config = SettingsConfigDict(
        env_file=None,
        case_sensitive=False,
        extra="ignore",
    )

    database_url: str | None = Field(default=None)
    postgres_user: str = Field(default="anime_user")
    postgres_password: str = Field(default="anime_password")
    postgres_host: str = Field(default="localhost")
    postgres_port: int = Field(default=5432)
    postgres_db: str = Field(default="anime_recommendation")

    jwt_secret_key: str = Field(..., min_length=16)
    jwt_algorithm: str = Field(default="HS256")
    jwt_access_token_expire_minutes: int = Field(default=30, ge=1, le=43200)

    cache_dir: Path = Field(default=Path("/tmp/cache"))
    allowed_origins: str = Field(default="http://localhost:8080,http://localhost:3000")
    log_level: str = Field(default="INFO")

    @field_validator("cache_dir", mode="before")
    @classmethod
    def resolve_cache_dir(cls, v: str | Path) -> Path:
        return Path(v).resolve() if isinstance(v, str) else v.resolve()

    def get_allowed_origins(self) -> list[str]:
        """Parse origins from JSON array or comma-separated string."""
        v = self.allowed_origins.strip()
        # Handle JSON array format
        if v.startswith("["):
            import json
            try:
                parsed = json.loads(v)
                if isinstance(parsed, list):
                    return [str(origin).strip() for origin in parsed if origin]
            except json.JSONDecodeError:
                pass
        # Fall back to comma-separated
        return [origin.strip() for origin in v.split(",") if origin.strip()]

    @field_validator("log_level")
    @classmethod
    def validate_log_level(cls, v: str) -> str:
        valid_levels = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        upper = v.upper()
        if upper not in valid_levels:
            raise ValueError(f"log_level must be one of {valid_levels}")
        return upper

    def get_database_url(self) -> str | None:
        if self.database_url:
            return self.database_url
        return (
            f"postgresql://{self.postgres_user}:{self.postgres_password}"
            f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
        )


Settings = _TestableSettings


class TestSettings:
    """Tests for Pydantic settings validation."""

    def test_jwt_secret_key_required(self):
        """JWT secret key is required."""
        with pytest.raises(Exception):
            Settings(jwt_secret_key=None)

    def test_jwt_secret_key_min_length(self):
        """JWT secret key must be at least 16 characters."""
        with pytest.raises(Exception):
            Settings(jwt_secret_key="short")

    def test_jwt_secret_key_valid(self):
        """Valid JWT secret key is accepted."""
        s = Settings(jwt_secret_key="this-is-a-valid-secret-key-123")
        assert len(s.jwt_secret_key) >= 16

    def test_default_jwt_algorithm(self):
        """Default JWT algorithm is HS256."""
        s = Settings(jwt_secret_key="this-is-a-valid-secret-key-123")
        assert s.jwt_algorithm == "HS256"

    def test_default_jwt_expire_minutes(self):
        """Default JWT expiration is 30 minutes."""
        s = Settings(jwt_secret_key="this-is-a-valid-secret-key-123")
        assert s.jwt_access_token_expire_minutes == 30

    def test_jwt_expire_minutes_range(self):
        """JWT expiration must be between 1 and 43200 minutes."""
        with pytest.raises(Exception):
            Settings(
                jwt_secret_key="this-is-a-valid-secret-key-123",
                jwt_access_token_expire_minutes=0
            )

    def test_allowed_origins_from_string(self):
        """Parses comma-separated origins string."""
        s = Settings(
            jwt_secret_key="this-is-a-valid-secret-key-123",
            allowed_origins="http://localhost:3000,http://localhost:8080"
        )
        origins = s.get_allowed_origins()
        assert "http://localhost:3000" in origins
        assert "http://localhost:8080" in origins

    def test_allowed_origins_from_json_string(self):
        """Parses JSON array string for origins."""
        s = Settings(
            jwt_secret_key="this-is-a-valid-secret-key-123",
            allowed_origins='["http://localhost:3000", "http://example.com"]'
        )
        origins = s.get_allowed_origins()
        assert "http://localhost:3000" in origins
        assert "http://example.com" in origins

    def test_log_level_validation(self):
        """Log level must be valid."""
        with pytest.raises(Exception):
            Settings(
                jwt_secret_key="this-is-a-valid-secret-key-123",
                log_level="INVALID"
            )

    def test_log_level_case_insensitive(self):
        """Log level is case-insensitive."""
        s = Settings(
            jwt_secret_key="this-is-a-valid-secret-key-123",
            log_level="debug"
        )
        assert s.log_level == "DEBUG"

    def test_cache_dir_resolved(self):
        """Cache dir is resolved to absolute path."""
        s = Settings(
            jwt_secret_key="this-is-a-valid-secret-key-123",
            cache_dir="./cache"
        )
        assert s.cache_dir.is_absolute()

    def test_database_url_from_components(self):
        """Database URL constructed from components when not provided."""
        s = Settings(
            jwt_secret_key="this-is-a-valid-secret-key-123",
            postgres_user="testuser",
            postgres_password="testpass",
            postgres_host="testhost",
            postgres_port=5432,
            postgres_db="testdb"
        )
        url = s.get_database_url()
        assert "testuser" in url
        assert "testpass" in url
        assert "testhost" in url
        assert "testdb" in url

    def test_database_url_override(self):
        """Direct database_url takes precedence."""
        s = Settings(
            jwt_secret_key="this-is-a-valid-secret-key-123",
            database_url="postgresql://override:pass@host/db"
        )
        assert s.get_database_url() == "postgresql://override:pass@host/db"

    def test_default_postgres_values(self):
        """Default Postgres values are set."""
        s = Settings(jwt_secret_key="this-is-a-valid-secret-key-123")
        assert s.postgres_user == "anime_user"
        assert s.postgres_password == "anime_password"
        assert s.postgres_host == "localhost"
        assert s.postgres_port == 5432
        assert s.postgres_db == "anime_recommendation"
