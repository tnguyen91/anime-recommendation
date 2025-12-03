"""Pydantic schemas for favorites requests and responses."""
from datetime import datetime
from pydantic import BaseModel, ConfigDict, Field


class FavoriteCreate(BaseModel):
    """Schema for adding an anime to favorites."""
    anime_id: int = Field(..., gt=0, description="The anime ID to add to favorites")


class FavoriteResponse(BaseModel):
    """Schema for a single favorite response."""
    id: int
    anime_id: int
    added_at: datetime
    
    name: str | None = None
    title_english: str | None = None
    image_url: str | None = None
    
    model_config = ConfigDict(from_attributes=True)


class FavoriteListResponse(BaseModel):
    """Schema for list of favorites response."""
    favorites: list[FavoriteResponse]
    total: int


class MessageResponse(BaseModel):
    """Schema for simple message responses."""
    message: str
