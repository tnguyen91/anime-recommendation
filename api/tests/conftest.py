"""
Pytest configuration - runs before any test imports.

This file is automatically loaded by pytest. We use it to set up
environment variables that must be present before importing app modules.

Note: Tests use in-memory SQLite databases (created in each test file),
not the production PostgreSQL database.
"""
import os

# Set required environment variables for testing BEFORE importing settings
# JWT_SECRET_KEY must be at least 16 characters (validated by Pydantic)
os.environ.setdefault("JWT_SECRET_KEY", "test-secret-key-for-testing-only-min-16-chars")