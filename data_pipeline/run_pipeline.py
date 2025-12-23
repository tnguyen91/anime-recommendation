"""CLI for running the data pipeline."""

import argparse
import logging
import sys
from datetime import datetime, timezone
from data_pipeline.config import ensure_directories, LOG_FORMAT, LOG_DATE_FORMAT
from data_pipeline.collectors import collect_recent_anime, collect_app_data
from data_pipeline.processors import DataUnifier, TrainingDataCreator
from data_pipeline.validators import DataValidator

logging.basicConfig(level=logging.INFO, format=LOG_FORMAT, datefmt=LOG_DATE_FORMAT)
logger = logging.getLogger(__name__)


def step_collect(args):
    """Collect anime data from Jikan API."""
    logger.info("=" * 60)
    logger.info("STEP 1a: Collecting anime data from Jikan API")
    logger.info("=" * 60)
    ensure_directories()
    try:
        anime_list = collect_recent_anime(save=True)
        logger.info(f"Collected {len(anime_list)} unique anime")
        return True
    except Exception as e:
        logger.error(f"Collection failed: {e}")
        return False


def step_collect_app(args):
    """Export user data from app database."""
    logger.info("=" * 60)
    logger.info("STEP 1b: Exporting app user data")
    logger.info("=" * 60)
    ensure_directories()
    try:
        interactions = collect_app_data(save=True)
        if len(interactions) > 0:
            logger.info(f"Exported {len(interactions)} user interactions")
        else:
            logger.info("No user interactions found in database")
        return True
    except Exception as e:
        logger.error(f"App data export failed: {e}")
        return False


def step_process(args):
    """Process and unify data from all sources."""
    logger.info("=" * 60)
    logger.info("STEP 2: Processing and unifying data")
    logger.info("=" * 60)
    ensure_directories()
    try:
        unifier = DataUnifier()
        anime_df, interactions_df = unifier.run()
        logger.info(f"Processed {len(anime_df)} anime, {len(interactions_df)} interactions")
        return True
    except Exception as e:
        logger.error(f"Processing failed: {e}")
        return False


def step_prepare(args):
    """Prepare training data for RBM."""
    logger.info("=" * 60)
    logger.info("STEP 3: Preparing training data")
    logger.info("=" * 60)
    ensure_directories()
    try:
        creator = TrainingDataCreator()
        train, test, metadata = creator.run()
        logger.info(f"Created training data: {train.shape}")
        return True
    except Exception as e:
        logger.error(f"Preparation failed: {e}")
        return False


def step_validate(args):
    """Validate all data."""
    logger.info("=" * 60)
    logger.info("STEP 4: Validating data")
    logger.info("=" * 60)
    validator = DataValidator()
    report = validator.generate_report()
    print(report)
    result = validator.validate_all()
    return result.is_valid


def run_full_pipeline(args):
    """Run complete pipeline."""
    logger.info("=" * 60)
    logger.info("RUNNING FULL DATA PIPELINE")
    logger.info(f"Started at: {datetime.now(timezone.utc).isoformat()}")
    logger.info("=" * 60)

    steps = [
        ("Collect", step_collect),
        ("Process", step_process),
        ("Prepare", step_prepare),
        ("Validate", step_validate),
    ]

    results = {}
    for step_name, step_func in steps:
        logger.info(f"\n{'='*60}")
        logger.info(f"Running: {step_name}")
        logger.info(f"{'='*60}")
        success = step_func(args)
        results[step_name] = success
        if not success:
            logger.error(f"Step '{step_name}' failed. Stopping pipeline.")
            break

    logger.info("\n" + "=" * 60)
    logger.info("PIPELINE SUMMARY")
    logger.info("=" * 60)
    for step_name, success in results.items():
        status = "PASSED" if success else "FAILED"
        logger.info(f"  {step_name}: {status}")

    all_passed = all(results.values())
    logger.info(f"\nOverall: {'SUCCESS' if all_passed else 'FAILED'}")
    logger.info(f"Completed at: {datetime.now(timezone.utc).isoformat()}")

    return all_passed


def run_update(args):
    """Run quick update without validation."""
    logger.info("=" * 60)
    logger.info("RUNNING QUICK UPDATE")
    logger.info(f"Started at: {datetime.now(timezone.utc).isoformat()}")
    logger.info("=" * 60)

    steps = [
        ("Collect", step_collect),
        ("Process", step_process),
        ("Prepare", step_prepare),
    ]
    for step_name, step_func in steps:
        success = step_func(args)
        if not success:
            logger.error(f"Update failed at step: {step_name}")
            return False

    logger.info("\nUpdate completed successfully!")
    return True


def main():
    """Main entry point for the pipeline CLI."""
    parser = argparse.ArgumentParser(description="Anime Recommendation Data Pipeline")
    subparsers = parser.add_subparsers(dest="command", help="Pipeline command")

    subparsers.add_parser("full", help="Run complete pipeline").set_defaults(func=run_full_pipeline)
    subparsers.add_parser("update", help="Quick update (no validation)").set_defaults(func=run_update)
    subparsers.add_parser("collect", help="Collect anime from Jikan API").set_defaults(func=step_collect)
    subparsers.add_parser("collect-app", help="Export app user data").set_defaults(func=step_collect_app)
    subparsers.add_parser("process", help="Process and unify all data").set_defaults(func=step_process)
    subparsers.add_parser("prepare", help="Prepare training data").set_defaults(func=step_prepare)
    subparsers.add_parser("validate", help="Validate all data").set_defaults(func=step_validate)

    args = parser.parse_args()
    if args.command is None:
        parser.print_help()
        sys.exit(1)
    success = args.func(args)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
