import logging
import time

import pandas as pd
import torch
from sqlalchemy.orm import Session

from api.app_state import AppState
from api.exceptions import ServiceUnavailableError, ValidationError
from api.feedback.repository import FeedbackRepository
from api.inference.recommender import get_recommendations
from api.monitoring import get_prediction_logger
from api.recommendations.schemas import AnimeResult

logger = logging.getLogger(__name__)


def _safe_str(value) -> str:
    return "" if pd.isnull(value) else str(value)


class RecommendationService:
    def __init__(self, app_state: AppState, db: Session | None = None):
        self.app_state = app_state
        self.db = db
        self.pred_logger = get_prediction_logger()

    def recommend(
        self,
        liked_anime: list[str],
        top_n: int = 10,
        exclude_ids: list[int] | None = None,
        user_id: int | None = None,
    ) -> list[AnimeResult]:
        if self.app_state.anime_df is None:
            raise ServiceUnavailableError("Dataset not loaded")

        matched_ids = self.app_state.anime_df[
            self.app_state.anime_df["name"].isin(liked_anime)
        ]["anime_id"].tolist()

        if not matched_ids:
            raise ValidationError("No matching anime found in our database")

        all_exclusions = list(exclude_ids or [])
        if user_id and self.db:
            feedback_repo = FeedbackRepository(self.db)
            feedback_exclusions = feedback_repo.get_excluded_anime_ids(user_id)
            all_exclusions = list(set(all_exclusions + feedback_exclusions))
            if feedback_exclusions:
                logger.info(
                    "User %d: excluding %d anime from feedback",
                    user_id,
                    len(feedback_exclusions),
                )

        prediction_start = time.time()
        try:
            input_vec = torch.FloatTensor(
                [[1 if a in matched_ids else 0 for a in self.app_state.anime_ids]]
            ).to(self.app_state.device)

            recs = get_recommendations(
                input_vec.squeeze(0),
                self.app_state.rbm,
                self.app_state.anime_ids,
                self.app_state.anime_df,
                top_n=top_n,
                exclude_ids=all_exclusions,
                device=self.app_state.device,
            )

            output_ids = recs["anime_id"].tolist() if not recs.empty else []
            latency_ms = (time.time() - prediction_start) * 1000
            self._log_prediction(
                matched_ids, output_ids, latency_ms, user_id, success=True
            )

            return [self._build_result(row) for _, row in recs.iterrows()]

        except Exception:
            latency_ms = (time.time() - prediction_start) * 1000
            self._log_prediction(
                matched_ids, [], latency_ms, user_id, success=False
            )
            raise

    def search(
        self, query: str, limit: int = 20, offset: int = 0
    ) -> tuple[list[AnimeResult], int]:
        if self.app_state.anime_df is None:
            raise ServiceUnavailableError("Dataset not loaded")

        anime_df = self.app_state.anime_df
        name_cols = [
            c for c in ["name", "title_english", "title_japanese"] if c in anime_df.columns
        ]

        mask = pd.Series(False, index=anime_df.index)
        for col in name_cols:
            mask |= anime_df[col].astype(str).str.contains(query, case=False, na=False)

        matches = anime_df[mask]
        cols = [
            c
            for c in ["anime_id", "name", "title_english", "title_japanese"]
            if c in matches.columns
        ]
        matches = matches[cols].drop_duplicates()

        total = len(matches)
        page = matches.iloc[offset : offset + limit]

        results = []
        for _, row in page.iterrows():
            anime_id = row.get("anime_id")
            if pd.isnull(anime_id):
                continue
            try:
                int(anime_id)
            except (TypeError, ValueError):
                continue
            results.append(self._build_result(row))

        return results, total

    def _build_result(self, row: pd.Series) -> AnimeResult:
        anime_id = int(row.get("anime_id", 0))
        info = self.app_state.get_metadata(anime_id)

        return AnimeResult(
            anime_id=anime_id,
            name=_safe_str(row.get("name")),
            title_english=_safe_str(row.get("title_english")),
            title_japanese=_safe_str(row.get("title_japanese")),
            image_url=info.get("image_url"),
            genre=info.get("genres", []),
            synopsis=info.get("synopsis"),
        )

    def _log_prediction(
        self,
        input_ids: list[int],
        output_ids: list[int],
        latency_ms: float,
        user_id: int | None,
        *,
        success: bool,
        error: str | None = None,
    ) -> None:
        entry = self.pred_logger.create_log_entry(
            input_anime_ids=input_ids,
            output_anime_ids=output_ids,
            latency_ms=latency_ms,
            user_id=user_id,
            success=success,
            error_message=error,
        )
        self.pred_logger.log(entry)
