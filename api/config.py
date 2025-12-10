"""
Application configuration constants.

These values control the behavior of the recommendation system.
Centralizing them here makes the codebase easier to tune and maintain.
"""
from __future__ import annotations

# HTTP status codes
HTTP_BAD_REQUEST = 400
HTTP_INTERNAL_ERROR = 500

# Recommendation settings
DEFAULT_TOP_N = 10          # Number of recommendations to return
RATING_THRESHOLD = 7        # Minimum rating to consider "liked"

# Data filtering thresholds
MIN_LIKES_USER = 100        # Minimum reviews a user must have
MIN_LIKES_ANIME = 50        # Minimum reviews an anime must have

# RBM model architecture
N_HIDDEN = 256              # Number of hidden units in RBM