import logging
from typing import Optional

from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session

from api.auth.dependencies import get_current_user, get_optional_current_user
from api.dependencies import get_db
from api.feedback.schemas import (
    BulkFeedbackCreate,
    BulkFeedbackResponse,
    FeedbackAction,
    FeedbackCreate,
    FeedbackHistoryResponse,
    FeedbackResponse,
    FeedbackStatsResponse,
)
from api.feedback.service import FeedbackService
from api.models import User

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/feedback", tags=["Feedback"])


@router.post("", response_model=FeedbackResponse, status_code=201)
def submit_feedback(
    feedback_data: FeedbackCreate,
    db: Session = Depends(get_db),
    current_user: Optional[User] = Depends(get_optional_current_user),
):
    service = FeedbackService(db)
    user_id = current_user.id if current_user else None
    return service.submit_feedback(feedback_data, user_id)


@router.post("/bulk", response_model=BulkFeedbackResponse, status_code=201)
def submit_bulk_feedback(
    bulk_data: BulkFeedbackCreate,
    db: Session = Depends(get_db),
    current_user: Optional[User] = Depends(get_optional_current_user),
):
    service = FeedbackService(db)
    user_id = current_user.id if current_user else None
    return service.submit_bulk(bulk_data, user_id)


@router.get("/stats", response_model=FeedbackStatsResponse)
def get_feedback_stats(
    days: int = Query(default=30, ge=1, le=365, description="Number of days to include"),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    service = FeedbackService(db)
    return service.get_stats(current_user.id, days)


@router.get("/history", response_model=FeedbackHistoryResponse)
def get_feedback_history(
    limit: int = Query(default=50, ge=1, le=200),
    offset: int = Query(default=0, ge=0),
    action: Optional[FeedbackAction] = Query(default=None),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    service = FeedbackService(db)
    return service.get_history(current_user.id, limit, offset, action)
