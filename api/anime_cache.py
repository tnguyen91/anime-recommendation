"""Anime metadata cache for enriching API responses."""
from typing import Any

_anime_metadata: dict[str, Any] = {}
_anime_df = None


def set_anime_cache(metadata: dict, df) -> None:
    """Initialize the anime cache at startup."""
    global _anime_metadata, _anime_df
    _anime_metadata = metadata
    _anime_df = df


def get_anime_info(anime_id: int) -> dict:
    """Retrieve anime metadata by ID."""
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