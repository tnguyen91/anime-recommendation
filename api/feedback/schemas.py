from datetime import datetime
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field


class FeedbackAction(StrEnum):
    FAVORITED = "favorited"
    DISMISSED = "dismissed"
    WATCHED = "watched"


class FeedbackCreate(BaseModel):
    anime_id: int = Field(..., gt=0, description="The anime ID the feedback is for")
    action: FeedbackAction = Field(..., description="The type of feedback action")
    recommendation_request_id: str | None = Field(
        None,
        max_length=50,
        description="Optional request ID from the recommendation that generated this anime",
    )


class FeedbackResponse(BaseModel):
    id: int
    anime_id: int
    action: FeedbackAction
    recorded_at: datetime
    message: str = "Feedback recorded successfully"

    model_config = ConfigDict(from_attributes=True)


class FeedbackStatsResponse(BaseModel):
    total_feedback: int
    feedback_by_action: dict[str, int]
    top_favorited_anime: list[dict]
    top_dismissed_anime: list[dict]


class BulkFeedbackCreate(BaseModel):
    feedback_items: list[FeedbackCreate] = Field(
        ...,
        min_length=1,
        max_length=50,
        description="List of feedback items to record",
    )


class BulkFeedbackResponse(BaseModel):
    recorded_count: int
    message: str = "Feedback recorded successfully"


class FeedbackHistoryItem(BaseModel):
    id: int
    anime_id: int
    action: str
    recorded_at: datetime
    recommendation_request_id: str | None = None

    model_config = ConfigDict(from_attributes=True)


class FeedbackHistoryResponse(BaseModel):
    items: list[FeedbackHistoryItem]
    total: int
    limit: int
    offset: int
