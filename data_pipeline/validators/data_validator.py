"""Validates data pipeline outputs."""

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from data_pipeline.config import (
    get_kaggle_anime_path,
    get_kaggle_reviews_path,
    get_processed_anime_path,
    get_processed_interactions_path,
    get_training_data_path,
    get_test_data_path,
    TRAINING_DATA_DIR,
    METADATA_FILE,
    ANIME_ID_MAP_FILE,
    find_kaggle_data_dir,
)

logger = logging.getLogger(__name__)


@dataclass
class ValidationResult:
    """Result of a validation check."""
    is_valid: bool
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    stats: dict = field(default_factory=dict)
    stage: str = "unknown"
    validated_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    def __str__(self) -> str:
        """Human-readable summary."""
        status = "PASSED" if self.is_valid else "FAILED"
        lines = [f"Validation [{self.stage}]: {status}"]

        if self.errors:
            lines.append(f"  Errors ({len(self.errors)}):")
            for err in self.errors:
                lines.append(f"    - {err}")
        if self.warnings:
            lines.append(f"  Warnings ({len(self.warnings)}):")
            for warn in self.warnings:
                lines.append(f"    - {warn}")
        if self.stats:
            lines.append("  Stats:")
            for key, value in self.stats.items():
                lines.append(f"    - {key}: {value}")
        return "\n".join(lines)

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "is_valid": self.is_valid,
            "errors": self.errors,
            "warnings": self.warnings,
            "stats": self.stats,
            "stage": self.stage,
            "validated_at": self.validated_at,
        }


class DataValidator:
    """Validates data quality at each stage of the pipeline."""

    def __init__(self):
        """Initialize the validator."""
        pass

    def validate_raw_data(self) -> ValidationResult:
        """Validate raw Kaggle data files."""
        errors = []
        warnings = []
        stats = {}

        kaggle_dir = find_kaggle_data_dir()
        if kaggle_dir is None:
            return ValidationResult(
                is_valid=False,
                errors=["Kaggle data directory not found"],
                stage="raw_data",
            )

        try:
            anime_path = get_kaggle_anime_path()
            anime_df = pd.read_csv(anime_path)

            stats["anime_file"] = str(anime_path)
            stats["anime_rows"] = len(anime_df)
            required_cols = ["anime_id", "name"]
            missing = [c for c in required_cols if c not in anime_df.columns]
            if missing:
                errors.append(f"Anime.csv missing columns: {missing}")

            if "anime_id" in anime_df.columns:
                n_duplicates = anime_df["anime_id"].duplicated().sum()
                if n_duplicates > 0:
                    errors.append(f"Anime.csv has {n_duplicates} duplicate anime_ids")

                invalid_ids = (anime_df["anime_id"] <= 0).sum()
                if invalid_ids > 0:
                    errors.append(f"Anime.csv has {invalid_ids} non-positive anime_ids")

            if "name" in anime_df.columns:
                empty_names = anime_df["name"].isna().sum()
                if empty_names > 0:
                    warnings.append(f"Anime.csv has {empty_names} empty names")
        except Exception as e:
            errors.append(f"Failed to read Anime.csv: {e}")

        try:
            reviews_path = get_kaggle_reviews_path()
            reviews_sample = pd.read_csv(reviews_path, nrows=10000)

            stats["reviews_file"] = str(reviews_path)
            required_cols = ["user_id", "anime_id", "score"]
            missing = [c for c in required_cols if c not in reviews_sample.columns]
            if missing:
                errors.append(f"User-AnimeReview.csv missing columns: {missing}")

            if "score" in reviews_sample.columns:
                valid_scores = reviews_sample["score"].dropna()
                if len(valid_scores) > 0:
                    min_score = valid_scores.min()
                    max_score = valid_scores.max()
                    stats["score_range"] = f"{min_score} - {max_score}"

                    if min_score < 1 or max_score > 10:
                        warnings.append(
                            f"Some scores outside 1-10 range: {min_score} to {max_score}"
                        )
        except Exception as e:
            errors.append(f"Failed to read User-AnimeReview.csv: {e}")

        return ValidationResult(
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
            stats=stats,
            stage="raw_data",
        )

    def validate_processed_data(self) -> ValidationResult:
        """Validate processed data files."""
        errors = []
        warnings = []
        stats = {}

        anime_path = get_processed_anime_path()
        if not anime_path.exists():
            errors.append(f"Anime catalog not found: {anime_path}")
        else:
            try:
                anime_df = pd.read_parquet(anime_path)
                stats["anime_path"] = str(anime_path)
                stats["anime_count"] = len(anime_df)

                required = ["mal_id", "title"]
                missing = [c for c in required if c not in anime_df.columns]
                if missing:
                    errors.append(f"Anime catalog missing columns: {missing}")

                if "mal_id" in anime_df.columns:
                    n_dups = anime_df["mal_id"].duplicated().sum()
                    if n_dups > 0:
                        errors.append(f"Anime catalog has {n_dups} duplicate mal_ids")

                    stats["anime_id_range"] = f"{anime_df['mal_id'].min()} - {anime_df['mal_id'].max()}"
            except Exception as e:
                errors.append(f"Failed to read anime catalog: {e}")

        interactions_path = get_processed_interactions_path()
        if not interactions_path.exists():
            errors.append(f"Interactions not found: {interactions_path}")
        else:
            try:
                interactions_df = pd.read_parquet(interactions_path)
                stats["interactions_path"] = str(interactions_path)
                stats["interactions_count"] = len(interactions_df)

                required = ["user_id", "mal_id", "liked"]
                missing = [c for c in required if c not in interactions_df.columns]
                if missing:
                    errors.append(f"Interactions missing columns: {missing}")

                if "liked" in interactions_df.columns:
                    unique_values = interactions_df["liked"].unique()
                    if not set(unique_values).issubset({0, 1}):
                        errors.append(
                            f"'liked' column has non-binary values: {unique_values}"
                        )

                    stats["like_ratio"] = f"{interactions_df['liked'].mean():.1%}"

                if "user_id" in interactions_df.columns:
                    stats["unique_users"] = interactions_df["user_id"].nunique()
                if "mal_id" in interactions_df.columns:
                    stats["unique_anime"] = interactions_df["mal_id"].nunique()

                if anime_path.exists() and "mal_id" in interactions_df.columns:
                    anime_df = pd.read_parquet(anime_path)
                    if "mal_id" in anime_df.columns:
                        valid_ids = set(anime_df["mal_id"])
                        interaction_ids = set(interactions_df["mal_id"])
                        orphan_ids = interaction_ids - valid_ids

                        if orphan_ids:
                            warnings.append(
                                f"{len(orphan_ids)} anime IDs in interactions not in catalog"
                            )

            except Exception as e:
                errors.append(f"Failed to read interactions: {e}")

        return ValidationResult(
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
            stats=stats,
            stage="processed_data",
        )

    def validate_training_data(self) -> ValidationResult:
        """Validate training data files."""
        errors = []
        warnings = []
        stats = {}

        train_path = get_training_data_path()
        test_path = get_test_data_path()
        metadata_path = TRAINING_DATA_DIR / METADATA_FILE
        anime_map_path = TRAINING_DATA_DIR / ANIME_ID_MAP_FILE

        missing_files = []
        if not train_path.exists():
            missing_files.append("train.parquet")
        if not test_path.exists():
            missing_files.append("test.npy")
        if not metadata_path.exists():
            missing_files.append("metadata.json")
        if not anime_map_path.exists():
            missing_files.append("anime_id_map.json")

        if missing_files:
            errors.append(f"Missing training files: {missing_files}")
            return ValidationResult(
                is_valid=False,
                errors=errors,
                stage="training_data",
            )

        try:
            with open(metadata_path, "r") as f:
                metadata = json.load(f)

            required_fields = ["n_users", "n_anime", "anime_to_idx", "idx_to_anime"]
            missing = [f for f in required_fields if f not in metadata]
            if missing:
                errors.append(f"Metadata missing fields: {missing}")

            stats["n_users_metadata"] = metadata.get("n_users")
            stats["n_anime_metadata"] = metadata.get("n_anime")
            stats["holdout_ratio"] = metadata.get("holdout_ratio")
            stats["created_at"] = metadata.get("created_at")
        except Exception as e:
            errors.append(f"Failed to read metadata: {e}")
            return ValidationResult(
                is_valid=False,
                errors=errors,
                stage="training_data",
            )

        try:
            train_df = pd.read_parquet(train_path)
            train = train_df.values

            stats["train_shape"] = f"{train.shape[0]} x {train.shape[1]}"

            if train.shape[0] != metadata.get("n_users"):
                errors.append(
                    f"Train rows ({train.shape[0]}) != metadata n_users ({metadata.get('n_users')})"
                )
            if train.shape[1] != metadata.get("n_anime"):
                errors.append(
                    f"Train cols ({train.shape[1]}) != metadata n_anime ({metadata.get('n_anime')})"
                )

            unique_values = np.unique(train)
            if not set(unique_values).issubset({0, 1, 0.0, 1.0}):
                errors.append(f"Train has non-binary values: {unique_values[:10]}")

            users_with_no_likes = (train.sum(axis=1) == 0).sum()
            if users_with_no_likes > 0:
                warnings.append(f"{users_with_no_likes} users have no likes in train set")

            stats["train_likes"] = int(train.sum())
            stats["train_sparsity"] = f"{(train == 0).mean():.1%}"
        except Exception as e:
            errors.append(f"Failed to read train data: {e}")

        try:
            test = np.load(test_path)

            stats["test_shape"] = f"{test.shape[0]} x {test.shape[1]}"

            if train.shape != test.shape:
                errors.append(
                    f"Train shape {train.shape} != test shape {test.shape}"
                )

            unique_values = np.unique(test)
            if not set(unique_values).issubset({0, 1, 0.0, 1.0}):
                errors.append(f"Test has non-binary values: {unique_values[:10]}")

            stats["test_likes"] = int(test.sum())

            total_likes = train.sum() + test.sum()
            actual_holdout = test.sum() / total_likes if total_likes > 0 else 0
            expected_holdout = metadata.get("holdout_ratio", 0.1)

            stats["actual_holdout_ratio"] = f"{actual_holdout:.1%}"

            if abs(actual_holdout - expected_holdout) > 0.02:
                warnings.append(
                    f"Actual holdout ({actual_holdout:.1%}) differs from expected ({expected_holdout:.1%})"
                )
        except Exception as e:
            errors.append(f"Failed to read test data: {e}")

        try:
            with open(anime_map_path, "r") as f:
                anime_map = json.load(f)

            stats["anime_map_size"] = len(anime_map)

            if len(anime_map) != metadata.get("n_anime"):
                warnings.append(
                    f"Anime map size ({len(anime_map)}) != n_anime ({metadata.get('n_anime')})"
                )
        except Exception as e:
            errors.append(f"Failed to read anime ID map: {e}")

        return ValidationResult(
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
            stats=stats,
            stage="training_data",
        )

    def validate_all(self) -> ValidationResult:
        """Run all validation checks."""
        logger.info("Running full data validation...")
        all_errors = []
        all_warnings = []
        all_stats = {}

        stages = [
            ("raw", self.validate_raw_data),
            ("processed", self.validate_processed_data),
            ("training", self.validate_training_data),
        ]

        for stage_name, validator_func in stages:
            logger.info(f"Validating {stage_name} data...")
            result = validator_func()

            all_errors.extend([f"[{stage_name}] {e}" for e in result.errors])
            all_warnings.extend([f"[{stage_name}] {w}" for w in result.warnings])
            all_stats[stage_name] = result.stats

            if result.is_valid:
                logger.info(f"  {stage_name}: PASSED")
            else:
                logger.warning(f"  {stage_name}: FAILED ({len(result.errors)} errors)")

        return ValidationResult(
            is_valid=len(all_errors) == 0,
            errors=all_errors,
            warnings=all_warnings,
            stats=all_stats,
            stage="all",
        )

    def generate_report(self, output_path: Optional[Path] = None) -> str:
        """Generate a full validation report."""
        result = self.validate_all()

        report_lines = [
            "=" * 60,
            "DATA VALIDATION REPORT",
            f"Generated: {result.validated_at}",
            "=" * 60,
            "",
            f"Overall Status: {'PASSED' if result.is_valid else 'FAILED'}",
            "",
        ]

        if result.errors:
            report_lines.append(f"ERRORS ({len(result.errors)}):")
            for err in result.errors:
                report_lines.append(f"  - {err}")
            report_lines.append("")
        if result.warnings:
            report_lines.append(f"WARNINGS ({len(result.warnings)}):")
            for warn in result.warnings:
                report_lines.append(f"  - {warn}")
            report_lines.append("")

        report_lines.append("STATISTICS:")
        for stage, stats in result.stats.items():
            report_lines.append(f"  [{stage}]")
            for key, value in stats.items():
                report_lines.append(f"    {key}: {value}")
        report_lines.append("")
        report_lines.append("=" * 60)

        report = "\n".join(report_lines)
        if output_path:
            output_path = Path(output_path)
            with open(output_path, "w") as f:
                json.dump(result.to_dict(), f, indent=2)
            logger.info(f"Report saved to {output_path}")
        return report


if __name__ == "__main__":
    import sys

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    print("=" * 60)
    print("Data Validation")
    print("=" * 60)

    validator = DataValidator()
    report = validator.generate_report()
    print(report)

    result = validator.validate_all()
    sys.exit(0 if result.is_valid else 1)
