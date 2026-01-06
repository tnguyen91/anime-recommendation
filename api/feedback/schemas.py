"""Pydantic schemas for feedback API."""
from datetime import datetime
from enum import Enum
from typing import Optional

from pydantic import BaseModel, ConfigDict, Field


class FeedbackAction(str, Enum):
    """Valid feedback actions for recommendations."""
    FAVORITED = "favorited"
    DISMISSED = "dismissed"
    WATCHED = "watched"


class FeedbackCreate(BaseModel):
    """Request body for submitting recommendation feedback."""
    anime_id: int = Field(..., gt=0, description="The anime ID the feedback is for")
    action: FeedbackAction = Field(..., description="The type of feedback action")
    recommendation_request_id: Optional[str] = Field(
        None,
        max_length=50,
        description="Optional request ID from the recommendation that generated this anime"
    )


class FeedbackResponse(BaseModel):
    """Response after recording feedback."""
    id: int
    anime_id: int
    action: FeedbackAction
    recorded_at: datetime
    message: str = "Feedback recorded successfully"

    model_config = ConfigDict(from_attributes=True)


class FeedbackStatsResponse(BaseModel):
    """Aggregated feedback statistics."""
    total_feedback: int
    feedback_by_action: dict[str, int]
    top_favorited_anime: list[dict]
    top_dismissed_anime: list[dict]


class BulkFeedbackCreate(BaseModel):
    """Request body for submitting multiple feedback items."""
    feedback_items: list[FeedbackCreate] = Field(
        ...,
        min_length=1,
        max_length=50,
        description="List of feedback items to record"
    )


class BulkFeedbackResponse(BaseModel):
    """Response after recording bulk feedback."""
    recorded_count: int
    message: str = "Feedback recorded successfully"
