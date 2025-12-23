"""Collects user interaction data from the application database."""

import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import pandas as pd
from sqlalchemy import create_engine, text
from sqlalchemy.orm import Session, sessionmaker

from data_pipeline.config import APP_LOGS_DIR

logger = logging.getLogger(__name__)


class AppCollector:
    """Collects user interaction data from the application database."""

    def __init__(
        self,
        database_url: Optional[str] = None,
        session: Optional[Session] = None,
    ):
        """Initialize the App Collector."""
        self._owns_session = False

        if session is not None:
            self.session = session
        elif database_url is not None:
            engine = create_engine(database_url)
            SessionLocal = sessionmaker(bind=engine)
            self.session = SessionLocal()
            self._owns_session = True
        else:
            try:
                from api.settings import settings
                db_url = settings.get_database_url()
                if db_url:
                    engine = create_engine(db_url)
                    SessionLocal = sessionmaker(bind=engine)
                    self.session = SessionLocal()
                    self._owns_session = True
                else:
                    raise ValueError("No database URL configured")
            except ImportError:
                raise ValueError(
                    "Must provide database_url or session, "
                    "or have api.settings configured"
                )

        APP_LOGS_DIR.mkdir(parents=True, exist_ok=True)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def close(self):
        if self._owns_session and self.session:
            self.session.close()

    def collect_user_favorites(self) -> pd.DataFrame:
        """Collect all user favorites from the database."""
        logger.info("Collecting user favorites from database...")
        query = text("""
            SELECT
                user_id,
                anime_id as mal_id,
                added_at
            FROM user_favorites
            ORDER BY user_id, added_at
        """)
        result = self.session.execute(query)
        rows = result.fetchall()

        if not rows:
            logger.info("No user favorites found in database")
            return pd.DataFrame(columns=["user_id", "mal_id", "liked", "added_at"])

        df = pd.DataFrame(rows, columns=["user_id", "mal_id", "added_at"])
        df["user_id"] = "app_" + df["user_id"].astype(str)
        df["liked"] = 1

        logger.info(f"Collected {len(df)} favorites from {df['user_id'].nunique()} users")
        return df

    def collect_recommendation_engagement(self) -> pd.DataFrame:
        """Collect recommendation engagement data."""
        logger.info("Collecting recommendation engagement from database...")
        query = text("""
            SELECT
                user_id,
                anime_id as mal_id,
                clicked,
                favorited,
                recommended_at
            FROM recommendation_history
            WHERE clicked = true OR favorited = true
            ORDER BY user_id, recommended_at
        """)
        result = self.session.execute(query)
        rows = result.fetchall()

        if not rows:
            logger.info("No recommendation engagement found")
            return pd.DataFrame(columns=["user_id", "mal_id", "clicked", "favorited", "recommended_at"])

        df = pd.DataFrame(rows, columns=["user_id", "mal_id", "clicked", "favorited", "recommended_at"])
        df["user_id"] = "app_" + df["user_id"].astype(str)

        logger.info(f"Collected {len(df)} engagement records")
        return df

    def collect_all_interactions(self) -> pd.DataFrame:
        """Collect all interactions (favorites and favorited recommendations)."""
        favorites_df = self.collect_user_favorites()
        engagement_df = self.collect_recommendation_engagement()

        favorited_recs = engagement_df[engagement_df["favorited"] == True][["user_id", "mal_id"]].copy()
        favorited_recs["liked"] = 1

        combined = pd.concat([
            favorites_df[["user_id", "mal_id", "liked"]],
            favorited_recs
        ], ignore_index=True)

        combined = combined.drop_duplicates(subset=["user_id", "mal_id"])
        logger.info(f"Total unique interactions: {len(combined)}")
        logger.info(f"  From favorites: {len(favorites_df)}")
        logger.info(f"  From recommendations: {len(favorited_recs)}")
        return combined

    def get_user_stats(self) -> dict:
        """Get statistics about users and interactions."""
        stats = {}

        result = self.session.execute(text("SELECT COUNT(*) FROM users"))
        stats["total_users"] = result.scalar()

        result = self.session.execute(text(
            "SELECT COUNT(DISTINCT user_id) FROM user_favorites"
        ))
        stats["users_with_favorites"] = result.scalar()

        result = self.session.execute(text("SELECT COUNT(*) FROM user_favorites"))
        stats["total_favorites"] = result.scalar()

        result = self.session.execute(text(
            "SELECT COUNT(DISTINCT anime_id) FROM user_favorites"
        ))
        stats["unique_anime_favorited"] = result.scalar()

        result = self.session.execute(text("SELECT COUNT(*) FROM recommendation_history"))
        stats["total_recommendations"] = result.scalar()

        result = self.session.execute(text(
            "SELECT COUNT(*) FROM recommendation_history WHERE clicked = true"
        ))
        stats["clicked_recommendations"] = result.scalar()

        result = self.session.execute(text(
            "SELECT COUNT(*) FROM recommendation_history WHERE favorited = true"
        ))
        stats["favorited_recommendations"] = result.scalar()
        return stats

    def save_interactions(
        self,
        interactions_df: pd.DataFrame,
        filename: Optional[str] = None
    ) -> Path:
        """Save interactions to a file for pipeline processing."""
        if filename is None:
            timestamp = datetime.now(timezone.utc).strftime("%Y%m%d")
            filename = f"app_interactions_{timestamp}.parquet"

        output_path = APP_LOGS_DIR / filename
        interactions_df.to_parquet(output_path, index=False)
        logger.info(f"Saved {len(interactions_df)} interactions to {output_path}")
        return output_path

    @staticmethod
    def load_latest_interactions() -> Optional[pd.DataFrame]:
        """Load the most recent app interactions file."""
        if not APP_LOGS_DIR.exists():
            return None

        parquet_files = sorted(APP_LOGS_DIR.glob("app_interactions_*.parquet"))
        if not parquet_files:
            return None

        latest_file = parquet_files[-1]
        logger.info(f"Loading app interactions from {latest_file}")
        return pd.read_parquet(latest_file)


def collect_app_data(database_url: Optional[str] = None, save: bool = True) -> pd.DataFrame:
    """Collect all app user data."""
    with AppCollector(database_url=database_url) as collector:
        stats = collector.get_user_stats()
        logger.info("Database statistics:")
        for key, value in stats.items():
            logger.info(f"  {key}: {value}")
        interactions_df = collector.collect_all_interactions()

        if save and len(interactions_df) > 0:
            collector.save_interactions(interactions_df)
        return interactions_df


if __name__ == "__main__":
    import sys
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    print("=" * 60)
    print("App User Data Collector")
    print("=" * 60)
    try:
        interactions = collect_app_data(save=True)
        if len(interactions) > 0:
            print(f"\nCollected {len(interactions)} interactions")
            print(f"Unique users: {interactions['user_id'].nunique()}")
            print(f"Unique anime: {interactions['mal_id'].nunique()}")
        else:
            print("\nNo interactions found in database")

    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
