"""Feedback query utilities for recommendation filtering."""
from typing import Optional

from sqlalchemy.orm import Session

from api.models import RecommendationFeedback


def get_excluded_anime_ids_for_user(
    db: Session,
    user_id: int,
    actions: Optional[list[str]] = None
) -> list[int]:
    """Get anime IDs to exclude from recommendations based on user feedback."""
    if actions is None:
        actions = ["dismissed", "watched"]
    
    feedback_records = (
        db.query(RecommendationFeedback.anime_id)
        .filter(
            RecommendationFeedback.user_id == user_id,
            RecommendationFeedback.action.in_(actions)
        )
        .distinct()
        .all()
    )
    
    return [record.anime_id for record in feedback_records]


def get_positive_feedback_anime_ids(
    db: Session,
    user_id: int
) -> list[int]:
    """Get anime IDs the user has favorited."""
    feedback_records = (
        db.query(RecommendationFeedback.anime_id)
        .filter(
            RecommendationFeedback.user_id == user_id,
            RecommendationFeedback.action == "favorited"
        )
        .distinct()
        .all()
    )
    
    return [record.anime_id for record in feedback_records]
