from datetime import datetime

from sqlalchemy import func
from sqlalchemy.orm import Session

from api.models import RecommendationFeedback


class FeedbackRepository:
    def __init__(self, db: Session):
        self.db = db

    def create(
        self,
        anime_id: int,
        action: str,
        user_id: int | None = None,
        recommendation_request_id: str | None = None,
    ) -> RecommendationFeedback:
        feedback = RecommendationFeedback(
            user_id=user_id,
            anime_id=anime_id,
            action=action,
            recommendation_request_id=recommendation_request_id,
        )
        self.db.add(feedback)
        self.db.commit()
        self.db.refresh(feedback)
        return feedback

    def create_bulk(self, items: list[dict]) -> int:
        records = [RecommendationFeedback(**item) for item in items]
        self.db.add_all(records)
        self.db.commit()
        return len(records)

    def get_action_counts(self, user_id: int, since: datetime) -> dict[str, int]:
        rows = (
            self.db.query(
                RecommendationFeedback.action,
                func.count(RecommendationFeedback.id),
            )
            .filter(
                RecommendationFeedback.user_id == user_id,
                RecommendationFeedback.recorded_at >= since,
            )
            .group_by(RecommendationFeedback.action)
            .all()
        )
        return {action: count for action, count in rows}

    def get_top_anime_by_action(
        self, user_id: int, action: str, since: datetime, limit: int = 10
    ) -> list[dict]:
        rows = (
            self.db.query(
                RecommendationFeedback.anime_id,
                func.count(RecommendationFeedback.id).label("count"),
            )
            .filter(
                RecommendationFeedback.user_id == user_id,
                RecommendationFeedback.action == action,
                RecommendationFeedback.recorded_at >= since,
            )
            .group_by(RecommendationFeedback.anime_id)
            .order_by(func.count(RecommendationFeedback.id).desc())
            .limit(limit)
            .all()
        )
        count_key = "favorite_count" if action == "favorited" else "dismiss_count"
        return [{"anime_id": anime_id, count_key: count} for anime_id, count in rows]

    def get_history(
        self,
        user_id: int,
        action: str | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> tuple[list[RecommendationFeedback], int]:
        query = self.db.query(RecommendationFeedback).filter(
            RecommendationFeedback.user_id == user_id
        )
        if action:
            query = query.filter(RecommendationFeedback.action == action)

        total = query.count()
        items = (
            query.order_by(RecommendationFeedback.recorded_at.desc())
            .offset(offset)
            .limit(limit)
            .all()
        )
        return items, total

    def get_excluded_anime_ids(
        self, user_id: int, actions: list[str] | None = None
    ) -> list[int]:
        if actions is None:
            actions = ["dismissed", "watched"]
        rows = (
            self.db.query(RecommendationFeedback.anime_id)
            .filter(
                RecommendationFeedback.user_id == user_id,
                RecommendationFeedback.action.in_(actions),
            )
            .distinct()
            .all()
        )
        return [r.anime_id for r in rows]
