"""
Anime metadata cache for enriching favorites and recommendations.

This module provides a shared cache of anime metadata (titles, images, etc.)
that is populated at application startup and used to enrich API responses.

The cache avoids circular imports between main.py and favorites/router.py
by providing a clean interface for accessing metadata.

Usage:
    # At startup (in main.py lifespan):
    set_anime_cache(anime_metadata, anime_df)
    
    # When enriching responses:
    info = get_anime_info(anime_id)  # Returns {"name": ..., "image_url": ...}
"""
from typing import Any

_anime_metadata: dict[str, Any] = {}
_anime_df = None


def set_anime_cache(metadata: dict, df) -> None:
    """
    Initialize the anime cache with metadata and dataframe.
    
    Called once during application startup.
    """
    global _anime_metadata, _anime_df
    _anime_metadata = metadata
    _anime_df = df


def get_anime_info(anime_id: int) -> dict:
    """
    Get anime info by ID from the cached metadata.
    
    Tries metadata dict first, falls back to dataframe if available.
    
    Returns:
        Dict with name, title_english, image_url (any may be None)
    """
    info = {}

    metadata = _anime_metadata.get(str(anime_id)) or _anime_metadata.get(anime_id, {})
    if metadata:
        info["name"] = metadata.get("title") or metadata.get("name")
        info["title_english"] = metadata.get("title_english")
        info["image_url"] = metadata.get("image_url")

    if not info.get("name") and _anime_df is not None:
        match = _anime_df[_anime_df["anime_id"] == anime_id]
        if not match.empty:
            row = match.iloc[0]
            info["name"] = row.get("name", "")
            info["title_english"] = row.get("title_english")

    return info