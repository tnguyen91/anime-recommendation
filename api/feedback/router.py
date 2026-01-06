"""Feedback API endpoints for tracking user interactions with recommendations."""
import logging
from collections import Counter
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy import func
from sqlalchemy.orm import Session

from api.auth.dependencies import get_current_user, get_optional_current_user
from api.database import get_db
from api.feedback.schemas import (
    BulkFeedbackCreate,
    BulkFeedbackResponse,
    FeedbackAction,
    FeedbackCreate,
    FeedbackResponse,
    FeedbackStatsResponse,
)
from api.models import RecommendationFeedback, User

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/feedback", tags=["Feedback"])


@router.post("", response_model=FeedbackResponse, status_code=status.HTTP_201_CREATED)
async def submit_feedback(
    feedback_data: FeedbackCreate,
    db: Session = Depends(get_db),
    current_user: Optional[User] = Depends(get_optional_current_user)
):
    """Submit feedback on a recommendation. Authentication optional."""
    user_id = current_user.id if current_user else None
    
    new_feedback = RecommendationFeedback(
        user_id=user_id,
        anime_id=feedback_data.anime_id,
        action=feedback_data.action.value,
        recommendation_request_id=feedback_data.recommendation_request_id
    )
    
    db.add(new_feedback)
    db.commit()
    db.refresh(new_feedback)
    
    logger.info(
        f"Feedback recorded: user={user_id}, anime={feedback_data.anime_id}, "
        f"action={feedback_data.action.value}"
    )
    
    return FeedbackResponse(
        id=new_feedback.id,
        anime_id=new_feedback.anime_id,
        action=FeedbackAction(new_feedback.action),
        recorded_at=new_feedback.recorded_at,
        message="Feedback recorded successfully"
    )


@router.post("/bulk", response_model=BulkFeedbackResponse, status_code=status.HTTP_201_CREATED)
async def submit_bulk_feedback(
    bulk_data: BulkFeedbackCreate,
    db: Session = Depends(get_db),
    current_user: Optional[User] = Depends(get_optional_current_user)
):
    """Submit multiple feedback items at once."""
    user_id = current_user.id if current_user else None
    
    feedback_records = [
        RecommendationFeedback(
            user_id=user_id,
            anime_id=item.anime_id,
            action=item.action.value,
            recommendation_request_id=item.recommendation_request_id
        )
        for item in bulk_data.feedback_items
    ]
    
    db.add_all(feedback_records)
    db.commit()
    
    logger.info(f"Bulk feedback recorded: user={user_id}, count={len(feedback_records)}")
    
    return BulkFeedbackResponse(
        recorded_count=len(feedback_records),
        message=f"Successfully recorded {len(feedback_records)} feedback items"
    )


@router.get("/stats", response_model=FeedbackStatsResponse)
async def get_feedback_stats(
    days: int = Query(default=30, ge=1, le=365, description="Number of days to include"),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Get aggregated feedback statistics for the current user."""
    from datetime import datetime, timedelta, timezone
    
    cutoff_date = datetime.now(timezone.utc) - timedelta(days=days)
    
    feedback_query = (
        db.query(RecommendationFeedback)
        .filter(
            RecommendationFeedback.user_id == current_user.id,
            RecommendationFeedback.recorded_at >= cutoff_date
        )
    )
    
    all_feedback = feedback_query.all()
    
    action_counts = Counter(f.action for f in all_feedback)
    
    favorited_feedback = [f for f in all_feedback if f.action == FeedbackAction.FAVORITED.value]
    favorited_counts = Counter(f.anime_id for f in favorited_feedback)
    top_favorited = [
        {"anime_id": anime_id, "favorite_count": count}
        for anime_id, count in favorited_counts.most_common(10)
    ]
    
    dismissed_feedback = [f for f in all_feedback if f.action == FeedbackAction.DISMISSED.value]
    dismissed_counts = Counter(f.anime_id for f in dismissed_feedback)
    top_dismissed = [
        {"anime_id": anime_id, "dismiss_count": count}
        for anime_id, count in dismissed_counts.most_common(10)
    ]
    
    return FeedbackStatsResponse(
        total_feedback=len(all_feedback),
        feedback_by_action=dict(action_counts),
        top_favorited_anime=top_favorited,
        top_dismissed_anime=top_dismissed
    )


@router.get("/history")
async def get_feedback_history(
    limit: int = Query(default=50, ge=1, le=200),
    offset: int = Query(default=0, ge=0),
    action: Optional[FeedbackAction] = Query(default=None),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Get paginated feedback history for the current user."""
    query = (
        db.query(RecommendationFeedback)
        .filter(RecommendationFeedback.user_id == current_user.id)
    )
    
    if action:
        query = query.filter(RecommendationFeedback.action == action.value)
    
    total = query.count()
    
    feedback_items = (
        query
        .order_by(RecommendationFeedback.recorded_at.desc())
        .offset(offset)
        .limit(limit)
        .all()
    )
    
    return {
        "items": [
            {
                "id": f.id,
                "anime_id": f.anime_id,
                "action": f.action,
                "recorded_at": f.recorded_at.isoformat(),
                "recommendation_request_id": f.recommendation_request_id
            }
            for f in feedback_items
        ],
        "total": total,
        "limit": limit,
        "offset": offset
    }
