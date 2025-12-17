"""Pydantic schemas for favorites API."""
from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field


class FavoriteCreate(BaseModel):
    """Request body for adding an anime to favorites."""
    anime_id: int = Field(..., gt=0, description="The anime ID to add to favorites")


class FavoriteResponse(BaseModel):
    """A single favorite with enriched anime metadata."""
    id: int
    anime_id: int
    added_at: datetime
    name: str | None = None
    title_english: str | None = None
    image_url: str | None = None

    model_config = ConfigDict(from_attributes=True)


class FavoriteListResponse(BaseModel):
    """Paginated list of favorites."""
    favorites: list[FavoriteResponse]
    total: int


class FavoriteCheckResponse(BaseModel):
    """Response for checking if anime is in favorites."""
    is_favorite: bool
    anime_id: int


class MessageResponse(BaseModel):
    """Generic success message response."""
    message: str