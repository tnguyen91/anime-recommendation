"""
Centralized application settings with Pydantic validation.

Loads configuration from environment variables with type coercion and validation.
Required settings are validated at startup to ensure fast failure.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Type-safe application configuration."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # Database
    database_url: str | None = Field(default=None)
    postgres_user: str = Field(default="anime_user")
    postgres_password: str = Field(default="anime_password")
    postgres_host: str = Field(default="localhost")
    postgres_port: int = Field(default=5432)
    postgres_db: str = Field(default="anime_recommendation")

    # JWT Authentication
    jwt_secret_key: str = Field(..., min_length=16)
    jwt_algorithm: str = Field(default="HS256")
    jwt_access_token_expire_minutes: int = Field(default=30, ge=1, le=43200)

    # Cache & Data
    cache_dir: Path = Field(default=Path("/tmp/cache"))
    model_uri: str | None = Field(default=None)
    metadata_uri: str | None = Field(default=None)

    # CORS - Use str type to prevent Pydantic's auto-JSON parsing of env vars
    allowed_origins: str = Field(default="http://localhost:8080,http://localhost:3000")

    # Logging
    log_level: str = Field(default="INFO")

    @field_validator("cache_dir", mode="before")
    @classmethod
    def resolve_cache_dir(cls, v: str | Path) -> Path:
        """Resolve to absolute path."""
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
        """Validate log level."""
        valid_levels = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        upper = v.upper()
        if upper not in valid_levels:
            raise ValueError(f"log_level must be one of {valid_levels}")
        return upper

    def get_database_url(self) -> str | None:
        """Get database URL, constructing from components if not provided."""
        if self.database_url:
            return self.database_url
        return (
            f"postgresql://{self.postgres_user}:{self.postgres_password}"
            f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
        )


@lru_cache
def get_settings() -> Settings:
    """Get cached settings instance (singleton pattern)."""
    return Settings()


settings = get_settings()
