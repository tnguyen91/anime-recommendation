import logging

from fastapi import APIRouter, Depends, Query, Request
from sqlalchemy.orm import Session

from api.app_state import AppState
from api.auth.dependencies import get_optional_current_user
from api.dependencies import get_app_state, get_db, limiter
from api.models import User
from api.recommendations.schemas import RecommendRequest, RecommendResponse, SearchResponse
from api.recommendations.service import RecommendationService

logger = logging.getLogger(__name__)
router = APIRouter(tags=["Recommendations"])


@router.post("/recommend", response_model=RecommendResponse)
@limiter.limit("30/minute")
def recommend(
    request: Request,
    body: RecommendRequest,
    app_state: AppState = Depends(get_app_state),
    db: Session = Depends(get_db),
    current_user: User | None = Depends(get_optional_current_user),
):
    service = RecommendationService(app_state, db)
    user_id = current_user.id if current_user else None

    results = service.recommend(
        liked_anime=body.liked_anime,
        top_n=body.top_n,
        exclude_ids=list(body.exclude_ids),
        user_id=user_id,
    )

    return RecommendResponse(
        recommendations=results,
        request_id=getattr(request.state, "request_id", None),
    )


@router.get("/search-anime", response_model=SearchResponse)
@limiter.limit("60/minute")
def search_anime(
    request: Request,
    query: str = Query("", max_length=100, description="Search term"),
    limit: int = Query(20, ge=1, le=100, description="Results per page"),
    offset: int = Query(0, ge=0, description="Number of results to skip"),
    app_state: AppState = Depends(get_app_state),
):
    query = query.strip()
    if not query:
        return SearchResponse(results=[], total=0, limit=limit, offset=offset)

    service = RecommendationService(app_state)
    results, total = service.search(query, limit, offset)

    return SearchResponse(results=results, total=total, limit=limit, offset=offset)
