"""Pytest configuration for test environment setup."""
import os

os.environ.setdefault("JWT_SECRET_KEY", "test-secret-key-for-testing-only-min-16-chars")