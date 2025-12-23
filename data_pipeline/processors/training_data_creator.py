"""Creates training data for RBM model training."""

import json
import logging
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import torch

from data_pipeline.config import (
    TRAINING_DATA_DIR,
    HOLDOUT_RATIO,
    get_processed_interactions_path,
    get_training_data_path,
    get_test_data_path,
    METADATA_FILE,
    ANIME_ID_MAP_FILE,
)

logger = logging.getLogger(__name__)

SEED = 42


class TrainingDataCreator:
    """Creates training data for the RBM from processed interactions."""

    def __init__(
        self,
        holdout_ratio: float = HOLDOUT_RATIO,
        seed: int = SEED,
    ):
        """Initialize the TrainingDataCreator."""
        self.holdout_ratio = holdout_ratio
        self.seed = seed
        TRAINING_DATA_DIR.mkdir(parents=True, exist_ok=True)

    def load_interactions(self) -> pd.DataFrame:
        """Load processed interactions from DataUnifier output."""
        interactions_path = get_processed_interactions_path()
        if not interactions_path.exists():
            raise FileNotFoundError(
                f"Processed interactions not found at {interactions_path}. "
                "Run DataUnifier first to create this file."
            )

        df = pd.read_parquet(interactions_path)
        logger.info(f"Loaded {len(df):,} interactions")
        return df

    def create_user_anime_matrix(
        self,
        interactions_df: pd.DataFrame
    ) -> tuple[pd.DataFrame, dict, dict]:
        """Create a user-anime pivot matrix from interactions."""
        logger.info("Creating user-anime pivot matrix...")
        matrix = interactions_df.pivot_table(
            index="user_id",
            columns="mal_id",
            values="liked",
            fill_value=0,
            aggfunc="max"
        )

        user_ids = list(matrix.index)
        anime_ids = list(matrix.columns)

        user_to_idx = {user_id: idx for idx, user_id in enumerate(user_ids)}
        anime_to_idx = {int(anime_id): idx for idx, anime_id in enumerate(anime_ids)}

        idx_to_user = {idx: user_id for user_id, idx in user_to_idx.items()}
        idx_to_anime = {idx: int(anime_id) for anime_id, idx in anime_to_idx.items()}

        logger.info(f"Matrix shape: {matrix.shape} (users x anime)")
        logger.info(f"Sparsity: {(matrix.values == 0).mean():.1%}")
        logger.info(f"Mean likes per user: {(matrix.values == 1).sum(axis=1).mean():.1f}")

        self._user_to_idx = user_to_idx
        self._anime_to_idx = anime_to_idx
        self._idx_to_user = idx_to_user
        self._idx_to_anime = idx_to_anime
        return matrix, user_to_idx, anime_to_idx

    def make_train_test_split(
        self,
        matrix: pd.DataFrame
    ) -> tuple[np.ndarray, np.ndarray]:
        """Split user-anime matrix into train/test with per-user holdout."""
        logger.info(f"Creating train/test split (holdout ratio: {self.holdout_ratio})...")
        np.random.seed(self.seed)

        data = matrix.values.copy()
        train = data.copy()
        test = np.zeros_like(data)

        n_users = data.shape[0]
        total_held_out = 0

        for user_idx in range(n_users):
            liked_indices = np.where(data[user_idx] == 1)[0]

            if len(liked_indices) == 0:
                continue

            n_holdout = max(1, int(np.floor(len(liked_indices) * self.holdout_ratio)))

            holdout_indices = np.random.choice(
                liked_indices,
                size=n_holdout,
                replace=False
            )

            train[user_idx, holdout_indices] = 0
            test[user_idx, holdout_indices] = 1
            total_held_out += n_holdout

        logger.info(f"Held out {total_held_out:,} items for testing")
        logger.info(f"Average held out per user: {total_held_out / n_users:.1f}")
        return train, test

    def create_metadata(
        self,
        interactions_df: pd.DataFrame,
        train_shape: tuple,
    ) -> dict:
        """Create metadata dictionary for training and inference."""
        metadata = {
            "n_users": train_shape[0],
            "n_anime": train_shape[1],
            "user_to_idx": self._user_to_idx,
            "idx_to_user": self._idx_to_user,
            "anime_to_idx": self._anime_to_idx,
            "idx_to_anime": self._idx_to_anime,
            "total_interactions": len(interactions_df),
            "total_likes": int(interactions_df["liked"].sum()),
            "like_ratio": float(interactions_df["liked"].mean()),
            "holdout_ratio": self.holdout_ratio,
            "seed": self.seed,
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        return metadata

    def save_outputs(
        self,
        train: np.ndarray,
        test: np.ndarray,
        metadata: dict,
    ):
        """Save training data and metadata to disk."""
        train_path = get_training_data_path()
        train_df = pd.DataFrame(
            train,
            columns=[str(metadata["idx_to_anime"][i]) for i in range(train.shape[1])]
        )
        train_df.to_parquet(train_path, index=False)
        logger.info(f"Saved training data to {train_path}")

        test_path = get_test_data_path()
        np.save(test_path, test)
        logger.info(f"Saved test data to {test_path}")

        metadata_path = TRAINING_DATA_DIR / METADATA_FILE
        metadata_json = self._convert_for_json(metadata)
        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(metadata_json, f, indent=2)
        logger.info(f"Saved metadata to {metadata_path}")

        anime_map_path = TRAINING_DATA_DIR / ANIME_ID_MAP_FILE
        anime_map = {str(k): v for k, v in metadata["anime_to_idx"].items()}
        with open(anime_map_path, "w", encoding="utf-8") as f:
            json.dump(anime_map, f, indent=2)
        logger.info(f"Saved anime ID map to {anime_map_path}")

    def _convert_for_json(self, obj):
        """Convert numpy types to Python types for JSON serialization."""
        if isinstance(obj, dict):
            return {str(k): self._convert_for_json(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [self._convert_for_json(item) for item in obj]
        elif isinstance(obj, (np.integer, np.int64, np.int32)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64, np.float32)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return obj

    def run(
        self,
        save: bool = True
    ) -> tuple[torch.Tensor, torch.Tensor, dict]:
        """Run the full training data creation pipeline."""
        logger.info("=" * 60)
        logger.info("Starting Training Data Creation")
        logger.info("=" * 60)
        start_time = datetime.now()
        logger.info("\nLoading interactions...")
        interactions_df = self.load_interactions()
        logger.info("\nCreating user-anime matrix...")
        matrix, user_to_idx, anime_to_idx = self.create_user_anime_matrix(interactions_df)
        logger.info("\nCreating train/test split...")
        train, test = self.make_train_test_split(matrix)
        logger.info("\nCreating metadata...")
        metadata = self.create_metadata(interactions_df, train.shape)
        if save:
            logger.info("\nSaving outputs...")
            self.save_outputs(train, test, metadata)
        train_tensor = torch.FloatTensor(train)
        test_tensor = torch.FloatTensor(test)
        elapsed = datetime.now() - start_time
        logger.info(f"\nPipeline completed in {elapsed}")
        logger.info("=" * 60)
        return train_tensor, test_tensor, metadata

    @staticmethod
    def load_training_data() -> tuple[torch.Tensor, torch.Tensor, dict]:
        """Load previously created training data from disk."""
        train_path = get_training_data_path()
        if not train_path.exists():
            raise FileNotFoundError(
                f"Training data not found at {train_path}. "
                "Run TrainingDataCreator first."
            )
        train_df = pd.read_parquet(train_path)
        train = train_df.values

        test_path = get_test_data_path()
        if not test_path.exists():
            raise FileNotFoundError(
                f"Test data not found at {test_path}. "
                "Run TrainingDataCreator first."
            )
        test = np.load(test_path)

        metadata_path = TRAINING_DATA_DIR / METADATA_FILE
        if not metadata_path.exists():
            raise FileNotFoundError(
                f"Metadata not found at {metadata_path}. "
                "Run TrainingDataCreator first."
            )
        with open(metadata_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)

        train_tensor = torch.FloatTensor(train)
        test_tensor = torch.FloatTensor(test)

        logger.info(f"Loaded training data: {train_tensor.shape}")
        return train_tensor, test_tensor, metadata

    @staticmethod
    def load_anime_id_map() -> dict[int, int]:
        """Load the anime ID to index mapping."""
        map_path = TRAINING_DATA_DIR / ANIME_ID_MAP_FILE
        if not map_path.exists():
            raise FileNotFoundError(
                f"Anime ID map not found at {map_path}. "
                "Run TrainingDataCreator first."
            )
        with open(map_path, "r", encoding="utf-8") as f:
            raw_map = json.load(f)
        return {int(k): v for k, v in raw_map.items()}

    @staticmethod
    def get_anime_ids() -> list[int]:
        """Get list of anime IDs in column order of the training matrix."""
        anime_map = TrainingDataCreator.load_anime_id_map()
        sorted_items = sorted(anime_map.items(), key=lambda x: x[1])
        return [mal_id for mal_id, _ in sorted_items]


if __name__ == "__main__":
    import sys

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    print("=" * 60)
    print("Training Data Creator")
    print("=" * 60)

    try:
        creator = TrainingDataCreator()
        train_tensor, test_tensor, metadata = creator.run()

        print("\nCompleted.")
        print(f"  Train tensor shape: {train_tensor.shape}")
        print(f"  Test tensor shape: {test_tensor.shape}")
        print(f"  Number of users: {metadata['n_users']:,}")
        print(f"  Number of anime: {metadata['n_anime']:,}")

        train_likes = (train_tensor == 1).sum().item()
        test_likes = (test_tensor == 1).sum().item()
        print(f"\n  Train likes: {train_likes:,}")
        print(f"  Test likes (held out): {test_likes:,}")
        print(f"  Holdout ratio actual: {test_likes / (train_likes + test_likes):.1%}")

    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
