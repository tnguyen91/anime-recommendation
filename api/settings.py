"""
Centralized application settings using Pydantic Settings.

This module provides type-safe, validated configuration loaded from environment
variables. All settings are validated at application startup, ensuring early
failure if required configuration is missing.

Usage:
    from settings import settings

    # Access settings as typed attributes
    cache_dir = settings.cache_dir
    jwt_secret = settings.jwt_secret_key

    # Settings are validated - if JWT_SECRET_KEY is missing, app won't start

Environment Variables:
    Required:
        - JWT_SECRET_KEY: Secret key for JWT token signing (min 32 chars in prod)

    Optional:
        - DATABASE_URL: PostgreSQL connection string
        - CACHE_DIR: Directory for model/data cache (default: /tmp/cache)
        - MODEL_URI: URL to download RBM model weights
        - METADATA_URI: URL to download anime metadata JSON
        - ANIME_CSV_URI: URL to download anime CSV
        - USER_REVIEW_CSV_URI: URL to download user reviews CSV
        - JWT_ALGORITHM: JWT signing algorithm (default: HS256)
        - JWT_ACCESS_TOKEN_EXPIRE_MINUTES: Token expiry (default: 30)
        - ALLOWED_ORIGINS: Comma-separated CORS origins
        - LOG_LEVEL: Logging level (default: INFO)
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Annotated

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings with validation and type coercion."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",  # Ignore extra env vars not defined here
    )

    # =========================================================================
    # Database Configuration
    # =========================================================================
    database_url: str | None = Field(
        default=None,
        description="PostgreSQL connection string (e.g., postgresql://user:pass@host:5432/db)"
    )

    # Individual DB components (used if DATABASE_URL not provided)
    postgres_user: str = Field(default="anime_user")
    postgres_password: str = Field(default="anime_password")
    postgres_host: str = Field(default="localhost")
    postgres_port: int = Field(default=5432)
    postgres_db: str = Field(default="anime_recommendation")

    # =========================================================================
    # JWT Authentication
    # =========================================================================
    jwt_secret_key: str = Field(
        ...,  # Required field
        min_length=16,
        description="Secret key for JWT signing. Must be at least 16 characters."
    )
    jwt_algorithm: str = Field(
        default="HS256",
        description="Algorithm for JWT token signing"
    )
    jwt_access_token_expire_minutes: int = Field(
        default=30,
        ge=1,
        le=43200,  # Max 30 days
        description="JWT token expiration time in minutes"
    )

    # =========================================================================
    # Cache & Data Directories
    # =========================================================================
    cache_dir: Path = Field(
        default=Path("/tmp/cache"),
        description="Directory for caching downloaded models and data"
    )

    # =========================================================================
    # Model & Data URIs
    # =========================================================================
    model_uri: str | None = Field(
        default=None,
        description="URL to download RBM model weights (.pth file)"
    )
    metadata_uri: str | None = Field(
        default=None,
        description="URL to download anime metadata JSON"
    )
    anime_csv_uri: str | None = Field(
        default=None,
        description="URL to download Anime.csv dataset"
    )
    user_review_csv_uri: str | None = Field(
        default=None,
        description="URL to download User-AnimeReview.csv dataset"
    )

    # =========================================================================
    # CORS Configuration
    # =========================================================================
    allowed_origins: list[str] = Field(
        default_factory=lambda: ["http://localhost:8080", "http://localhost:3000"],
        description="List of allowed CORS origins"
    )

    # =========================================================================
    # Logging
    # =========================================================================
    log_level: str = Field(
        default="INFO",
        description="Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)"
    )

    # =========================================================================
    # Validators
    # =========================================================================
    @field_validator("cache_dir", mode="before")
    @classmethod
    def resolve_cache_dir(cls, v: str | Path) -> Path:
        """Convert string to Path and resolve to absolute path."""
        path = Path(v) if isinstance(v, str) else v
        return path.resolve()

    @field_validator("allowed_origins", mode="before")
    @classmethod
    def parse_allowed_origins(cls, v: str | list[str]) -> list[str]:
        """Parse comma-separated string into list of origins."""
        if isinstance(v, str):
            return [origin.strip() for origin in v.split(",") if origin.strip()]
        return v

    @field_validator("log_level")
    @classmethod
    def validate_log_level(cls, v: str) -> str:
        """Ensure log level is valid."""
        valid_levels = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        upper = v.upper()
        if upper not in valid_levels:
            raise ValueError(f"log_level must be one of {valid_levels}")
        return upper

    # =========================================================================
    # Computed Properties
    # =========================================================================
    def get_database_url(self) -> str | None:
        """
        Get the database URL, constructing from components if not provided directly.

        Returns:
            Database connection string or None if not configured.
        """
        if self.database_url:
            return self.database_url

        # Construct from individual components
        return (
            f"postgresql://{self.postgres_user}:{self.postgres_password}"
            f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
        )


@lru_cache
def get_settings() -> Settings:
    """
    Get cached settings instance.

    Uses lru_cache to ensure settings are only loaded once.
    This also validates all settings at first access.

    Returns:
        Validated Settings instance.

    Raises:
        ValidationError: If required settings are missing or invalid.
    """
    return Settings()


# Convenience singleton for direct imports
# Usage: from settings import settings
settings = get_settings()
