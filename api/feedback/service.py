import logging
from datetime import datetime, timedelta, timezone

from sqlalchemy.orm import Session

from api.feedback.repository import FeedbackRepository
from api.feedback.schemas import (
    BulkFeedbackCreate,
    BulkFeedbackResponse,
    FeedbackAction,
    FeedbackCreate,
    FeedbackHistoryItem,
    FeedbackHistoryResponse,
    FeedbackResponse,
    FeedbackStatsResponse,
)

logger = logging.getLogger(__name__)


class FeedbackService:
    def __init__(self, db: Session):
        self.repo = FeedbackRepository(db)

    def submit_feedback(
        self, feedback_data: FeedbackCreate, user_id: int | None = None
    ) -> FeedbackResponse:
        record = self.repo.create(
            anime_id=feedback_data.anime_id,
            action=feedback_data.action.value,
            user_id=user_id,
            recommendation_request_id=feedback_data.recommendation_request_id,
        )
        logger.info(
            "Feedback recorded: user=%s anime=%d action=%s",
            user_id,
            feedback_data.anime_id,
            feedback_data.action.value,
        )
        return FeedbackResponse(
            id=record.id,
            anime_id=record.anime_id,
            action=FeedbackAction(record.action),
            recorded_at=record.recorded_at,
        )

    def submit_bulk(
        self, bulk_data: BulkFeedbackCreate, user_id: int | None = None
    ) -> BulkFeedbackResponse:
        items = [
            {
                "user_id": user_id,
                "anime_id": item.anime_id,
                "action": item.action.value,
                "recommendation_request_id": item.recommendation_request_id,
            }
            for item in bulk_data.feedback_items
        ]
        count = self.repo.create_bulk(items)
        logger.info("Bulk feedback recorded: user=%s count=%d", user_id, count)
        return BulkFeedbackResponse(
            recorded_count=count,
            message=f"Successfully recorded {count} feedback items",
        )

    def get_stats(self, user_id: int, days: int = 30) -> FeedbackStatsResponse:
        cutoff = datetime.now(timezone.utc) - timedelta(days=days)

        action_counts = self.repo.get_action_counts(user_id, cutoff)
        top_favorited = self.repo.get_top_anime_by_action(user_id, "favorited", cutoff)
        top_dismissed = self.repo.get_top_anime_by_action(user_id, "dismissed", cutoff)

        return FeedbackStatsResponse(
            total_feedback=sum(action_counts.values()),
            feedback_by_action=action_counts,
            top_favorited_anime=top_favorited,
            top_dismissed_anime=top_dismissed,
        )

    def get_history(
        self,
        user_id: int,
        limit: int = 50,
        offset: int = 0,
        action: FeedbackAction | None = None,
    ) -> FeedbackHistoryResponse:
        action_value = action.value if action else None
        items, total = self.repo.get_history(user_id, action_value, limit, offset)

        return FeedbackHistoryResponse(
            items=[
                FeedbackHistoryItem(
                    id=f.id,
                    anime_id=f.anime_id,
                    action=f.action,
                    recorded_at=f.recorded_at,
                    recommendation_request_id=f.recommendation_request_id,
                )
                for f in items
            ],
            total=total,
            limit=limit,
            offset=offset,
        )
