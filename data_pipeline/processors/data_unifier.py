"""Unifies anime data from multiple sources into a single dataset."""

import json
import logging
from datetime import datetime
from typing import Optional

import pandas as pd

from data_pipeline.config import (
    get_kaggle_anime_path,
    get_kaggle_reviews_path,
    get_processed_anime_path,
    get_processed_interactions_path,
    JIKAN_DATA_DIR,
    APP_LOGS_DIR,
    PROCESSED_DATA_DIR,
    RATING_THRESHOLD,
    MIN_LIKES_PER_USER,
    MIN_LIKES_PER_ANIME,
)

logger = logging.getLogger(__name__)


class DataUnifier:
    """Unifies anime data from multiple sources into a single dataset."""

    def __init__(
        self,
        rating_threshold: int = RATING_THRESHOLD,
        min_likes_per_user: int = MIN_LIKES_PER_USER,
        min_likes_per_anime: int = MIN_LIKES_PER_ANIME,
    ):
        """Initialize the DataUnifier."""
        self.rating_threshold = rating_threshold
        self.min_likes_per_user = min_likes_per_user
        self.min_likes_per_anime = min_likes_per_anime
        PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)

    def load_kaggle_anime(self) -> pd.DataFrame:
        """Load anime metadata from Kaggle's Anime.csv."""
        anime_path = get_kaggle_anime_path()
        logger.info(f"Loading Kaggle anime from {anime_path}")
        df = pd.read_csv(anime_path)
        logger.info(f"Loaded {len(df):,} anime from Kaggle")
        df = df.rename(columns={
            "anime_id": "mal_id",
            "name": "title",
            "rating": "age_rating",
        })

        df["genres"] = df["genre"].apply(self._parse_comma_list)
        df["studios"] = df["studio"].apply(self._parse_comma_list)
        df = df.drop(columns=["genre", "studio"], errors="ignore")
        df["data_source"] = "kaggle"
        return df

    def load_kaggle_reviews(self) -> pd.DataFrame:
        """Load user reviews/ratings from Kaggle's User-AnimeReview.csv."""
        reviews_path = get_kaggle_reviews_path()
        logger.info(f"Loading Kaggle reviews from {reviews_path}")
        chunks = []
        chunk_size = 1_000_000

        for i, chunk in enumerate(pd.read_csv(reviews_path, chunksize=chunk_size)):
            chunks.append(chunk)
            if (i + 1) % 10 == 0:
                logger.info(f"Loaded {(i+1) * chunk_size:,} rows...")

        df = pd.concat(chunks, ignore_index=True)
        logger.info(f"Loaded {len(df):,} reviews from Kaggle")
        df = df.rename(columns={"anime_id": "mal_id"})
        return df

    def load_jikan_anime(self) -> pd.DataFrame:
        """Load anime data from Jikan JSON files."""
        if not JIKAN_DATA_DIR.exists():
            logger.info("No Jikan data directory found")
            return pd.DataFrame()

        json_files = list(JIKAN_DATA_DIR.glob("*.json"))
        if not json_files:
            logger.info("No Jikan JSON files found")
            return pd.DataFrame()

        logger.info(f"Found {len(json_files)} Jikan data files")
        all_anime = []
        seen_ids = set()

        for json_path in json_files:
            try:
                with open(json_path, "r", encoding="utf-8") as f:
                    data = json.load(f)

                for item in data:
                    mal_id = item.get("mal_id")
                    if mal_id and mal_id not in seen_ids:
                        seen_ids.add(mal_id)
                        all_anime.append(item)

            except (json.JSONDecodeError, KeyError) as e:
                logger.warning(f"Error reading {json_path}: {e}")
                continue

        if not all_anime:
            return pd.DataFrame()

        df = pd.DataFrame(all_anime)
        logger.info(f"Loaded {len(df):,} unique anime from Jikan")

        df["data_source"] = "jikan"
        return df

    def load_and_merge_anime(self) -> pd.DataFrame:
        """Load anime from all sources and merge into unified catalog."""
        kaggle_df = self.load_kaggle_anime()
        jikan_df = self.load_jikan_anime()

        if jikan_df.empty:
            logger.info("No Jikan data to merge, using Kaggle only")
            return self._standardize_anime_columns(kaggle_df)

        kaggle_df = self._standardize_anime_columns(kaggle_df)
        jikan_df = self._standardize_anime_columns(jikan_df)

        kaggle_ids = set(kaggle_df["mal_id"].unique())
        jikan_only = jikan_df[~jikan_df["mal_id"].isin(kaggle_ids)]

        logger.info(f"Found {len(jikan_only):,} anime in Jikan but not in Kaggle")
        merged = pd.concat([kaggle_df, jikan_only], ignore_index=True)
        jikan_updates = jikan_df[jikan_df["mal_id"].isin(kaggle_ids)]
        if not jikan_updates.empty:
            logger.info(f"Could update {len(jikan_updates):,} anime with Jikan data")

        logger.info(f"Unified anime catalog: {len(merged):,} total anime")
        return merged

    def _standardize_anime_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Ensure DataFrame has all expected columns with correct types."""
        expected_columns = {
            "mal_id": None,
            "title": None,
            "title_english": None,
            "title_japanese": None,
            "synopsis": None,
            "genres": [],
            "type": None,
            "episodes": None,
            "status": None,
            "popularity": None,
            "members": None,
            "aired_from": None,
            "aired_to": None,
            "studios": [],
            "source": None,
            "age_rating": None,
            "data_source": "unknown",
        }
        for col, default in expected_columns.items():
            if col not in df.columns:
                df[col] = default

        return df[list(expected_columns.keys())].copy()

    def load_app_interactions(self) -> pd.DataFrame:
        """Load user interactions from app database exports."""
        if not APP_LOGS_DIR.exists():
            logger.info("No app logs directory found")
            return pd.DataFrame()

        parquet_files = sorted(APP_LOGS_DIR.glob("app_interactions_*.parquet"))
        if not parquet_files:
            logger.info("No app interaction files found")
            return pd.DataFrame()

        logger.info(f"Found {len(parquet_files)} app interaction files")
        all_interactions = []
        for parquet_path in parquet_files:
            try:
                df = pd.read_parquet(parquet_path)
                all_interactions.append(df)
                logger.info(f"Loaded {len(df)} interactions from {parquet_path.name}")
            except Exception as e:
                logger.warning(f"Error reading {parquet_path}: {e}")
                continue

        if not all_interactions:
            return pd.DataFrame()

        combined = pd.concat(all_interactions, ignore_index=True)
        combined = combined.drop_duplicates(subset=["user_id", "mal_id"])
        logger.info(f"Total app interactions: {len(combined)}")
        logger.info(f"  Unique app users: {combined['user_id'].nunique()}")
        logger.info(f"  Unique anime: {combined['mal_id'].nunique()}")
        return combined

    def load_and_process_interactions(
        self,
        anime_df: Optional[pd.DataFrame] = None,
        include_app_users: bool = True
    ) -> pd.DataFrame:
        """Load user interactions and convert to binary preferences."""
        if anime_df is None:
            anime_df = self.load_and_merge_anime()

        valid_anime_ids = set(anime_df["mal_id"].unique())
        logger.info(f"Filtering interactions to {len(valid_anime_ids):,} valid anime")
        reviews_df = self.load_kaggle_reviews()
        initial_count = len(reviews_df)
        reviews_df = reviews_df.dropna(subset=["score"])
        logger.info(f"Dropped {initial_count - len(reviews_df):,} unrated entries")

        reviews_df["liked"] = (reviews_df["score"] >= self.rating_threshold).astype(int)
        kaggle_interactions = reviews_df[["user_id", "mal_id", "liked"]].copy()
        logger.info(f"Kaggle interactions: {len(kaggle_interactions):,}")

        if include_app_users:
            app_interactions = self.load_app_interactions()
            if not app_interactions.empty:
                app_interactions = app_interactions[["user_id", "mal_id", "liked"]].copy()
                all_interactions = pd.concat(
                    [kaggle_interactions, app_interactions],
                    ignore_index=True
                )
                logger.info(f"Combined with {len(app_interactions):,} app interactions")
            else:
                all_interactions = kaggle_interactions
        else:
            all_interactions = kaggle_interactions

        all_interactions = all_interactions[all_interactions["mal_id"].isin(valid_anime_ids)]
        logger.info(f"After anime filter: {len(all_interactions):,} interactions")

        likes_per_user = all_interactions[all_interactions["liked"] == 1].groupby("user_id").size()
        active_users = likes_per_user[likes_per_user >= self.min_likes_per_user].index
        all_interactions = all_interactions[all_interactions["user_id"].isin(active_users)]
        logger.info(f"Active users (>= {self.min_likes_per_user} likes): {len(active_users):,}")

        likes_per_anime = all_interactions[all_interactions["liked"] == 1].groupby("mal_id").size()
        popular_anime = likes_per_anime[likes_per_anime >= self.min_likes_per_anime].index
        all_interactions = all_interactions[all_interactions["mal_id"].isin(popular_anime)]
        logger.info(f"Popular anime (>= {self.min_likes_per_anime} likes): {len(popular_anime):,}")
        interactions = all_interactions.drop_duplicates(subset=["user_id", "mal_id"])
        logger.info(f"Final interactions: {len(interactions):,} rows")
        logger.info(f"  Unique users: {interactions['user_id'].nunique():,}")
        logger.info(f"  Unique anime: {interactions['mal_id'].nunique():,}")
        logger.info(f"  Like ratio: {interactions['liked'].mean():.1%}")

        if include_app_users:
            app_user_count = interactions["user_id"].str.startswith("app_").sum()
            if app_user_count > 0:
                logger.info(f"  App user interactions: {app_user_count:,}")
        return interactions

    def run(self, save: bool = True) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Run the full data unification pipeline."""
        logger.info("=" * 60)
        logger.info("Starting Data Unification Pipeline")
        logger.info("=" * 60)
        start_time = datetime.now()
        logger.info("\nLoading and merging anime catalogs...")
        anime_df = self.load_and_merge_anime()
        logger.info("\nProcessing user interactions...")
        interactions_df = self.load_and_process_interactions(anime_df)
        anime_with_interactions = set(interactions_df["mal_id"].unique())
        anime_df = anime_df[anime_df["mal_id"].isin(anime_with_interactions)]
        logger.info(f"Anime with interactions: {len(anime_df):,}")

        if save:
            logger.info("\nSaving to Parquet...")
            self._save_outputs(anime_df, interactions_df)
        elapsed = datetime.now() - start_time
        logger.info(f"\nPipeline completed in {elapsed}")
        logger.info("=" * 60)
        return anime_df, interactions_df

    def _save_outputs(self, anime_df: pd.DataFrame, interactions_df: pd.DataFrame):
        """Save processed data to Parquet files."""
        anime_path = get_processed_anime_path()
        interactions_path = get_processed_interactions_path()

        anime_df = anime_df.copy()
        for col in ["genres", "studios"]:
            if col in anime_df.columns:
                anime_df[col] = anime_df[col].apply(
                    lambda x: json.dumps(x) if isinstance(x, list) else x
                )
        anime_df.to_parquet(anime_path, index=False)
        logger.info(f"Saved anime catalog to {anime_path}")

        interactions_df.to_parquet(interactions_path, index=False)
        logger.info(f"Saved interactions to {interactions_path}")

    @staticmethod
    def _parse_comma_list(value) -> list:
        """Parse a comma-separated string into a list."""
        if pd.isna(value) or value == "":
            return []
        if isinstance(value, list):
            return value
        return [item.strip() for item in str(value).split(",") if item.strip()]

    @staticmethod
    def load_processed_anime() -> pd.DataFrame:
        """Load the processed anime catalog from disk."""
        path = get_processed_anime_path()
        if not path.exists():
            raise FileNotFoundError(f"Processed anime not found at {path}. Run DataUnifier first.")

        df = pd.read_parquet(path)
        for col in ["genres", "studios"]:
            if col in df.columns:
                df[col] = df[col].apply(
                    lambda x: json.loads(x) if isinstance(x, str) else x
                )
        return df

    @staticmethod
    def load_processed_interactions() -> pd.DataFrame:
        """Load the processed interactions from disk."""
        path = get_processed_interactions_path()
        if not path.exists():
            raise FileNotFoundError(f"Processed interactions not found at {path}. Run DataUnifier first.")
        return pd.read_parquet(path)


if __name__ == "__main__":
    import sys

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    print("=" * 60)
    print("Data Unifier - Processing Pipeline")
    print("=" * 60)

    try:
        unifier = DataUnifier()
        anime_df, interactions_df = unifier.run()

        print("\nCompleted.")
        print(f"  Anime catalog: {len(anime_df):,} entries")
        print(f"  Interactions: {len(interactions_df):,} entries")

    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
