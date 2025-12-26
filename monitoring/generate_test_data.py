"""Generate synthetic prediction logs for testing the monitoring system."""
import json
import random
from datetime import datetime, timedelta, timezone
from pathlib import Path

POPULAR_ANIME_IDS = [
    1, 5, 20, 21, 22, 23, 30, 121, 164, 199, 223, 269,
    431, 457, 512, 813, 1535, 1575, 2001, 2904, 5114, 6547, 9253, 11061, 16498
]
COMMON_INPUT_IDS = POPULAR_ANIME_IDS[:15]
COMMON_OUTPUT_IDS = POPULAR_ANIME_IDS[5:]


def generate_prediction(
    timestamp: datetime,
    input_count_range: tuple = (3, 8),
    output_count: int = 10,
    latency_range: tuple = (100, 300),
    error_rate: float = 0.02
) -> dict:
    """Generate a single synthetic prediction log entry."""
    request_id = f"{random.randint(10000000, 99999999):08x}"[:8]
    input_count = random.randint(*input_count_range)
    input_anime_ids = random.sample(COMMON_INPUT_IDS, min(input_count, len(COMMON_INPUT_IDS)))
    success = random.random() > error_rate

    if success:
        output_anime_ids = random.sample(COMMON_OUTPUT_IDS, min(output_count, len(COMMON_OUTPUT_IDS)))
        latency_ms = random.uniform(*latency_range)
        error_message = None
    else:
        output_anime_ids = []
        latency_ms = random.uniform(50, 150)
        error_message = random.choice([
            "Model inference failed",
            "Dataset not loaded",
            "Timeout during prediction"
        ])

    return {
        "request_id": request_id,
        "timestamp": timestamp.isoformat().replace("+00:00", "Z"),
        "user_id": random.choice([None, None, None, random.randint(1, 100)]),
        "input_anime_ids": input_anime_ids,
        "input_count": len(input_anime_ids),
        "output_anime_ids": output_anime_ids,
        "output_count": len(output_anime_ids),
        "latency_ms": round(latency_ms, 2),
        "success": success,
        "error_message": error_message
    }


def generate_predictions_for_period(
    start_date: datetime,
    end_date: datetime,
    predictions_per_day: int = 50,
    **kwargs
) -> list[dict]:
    """Generate predictions for a date range with realistic daily patterns."""
    predictions = []
    current_date = start_date

    while current_date < end_date:
        day_of_week = current_date.weekday()
        daily_count = predictions_per_day
        if day_of_week >= 5:
            daily_count = int(predictions_per_day * 0.6)

        daily_count = int(daily_count * random.uniform(0.7, 1.3))

        for _ in range(daily_count):
            hour = random.randint(6, 23)
            minute = random.randint(0, 59)
            second = random.randint(0, 59)

            timestamp = current_date.replace(
                hour=hour, minute=minute, second=second,
                tzinfo=timezone.utc
            )

            predictions.append(generate_prediction(timestamp, **kwargs))

        current_date += timedelta(days=1)

    predictions.sort(key=lambda x: x["timestamp"])
    return predictions


def main():
    """Generate test data and write to log file."""
    log_dir = Path("logs")
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / "predictions.jsonl"

    print("Generating test prediction logs...")

    now = datetime.now(timezone.utc)

    historical_start = now - timedelta(days=30)
    historical_end = now - timedelta(days=7)

    historical_predictions = generate_predictions_for_period(
        historical_start,
        historical_end,
        predictions_per_day=40,
        input_count_range=(3, 7),
        latency_range=(120, 250),
        error_rate=0.01
    )
    print(f"  Generated {len(historical_predictions)} historical predictions (days 30-7)")

    current_start = now - timedelta(days=7)
    current_end = now

    current_predictions = generate_predictions_for_period(
        current_start,
        current_end,
        predictions_per_day=55,
        input_count_range=(4, 9),
        latency_range=(130, 280),
        error_rate=0.015
    )
    print(f"  Generated {len(current_predictions)} current predictions (last 7 days)")

    all_predictions = historical_predictions + current_predictions

    with open(log_file, "w", encoding="utf-8") as f:
        for pred in all_predictions:
            f.write(json.dumps(pred) + "\n")

    print(f"\nWritten {len(all_predictions)} predictions to: {log_file.absolute()}")
    print("\nYou can now test:")
    print("  1. python -m monitoring.analyze_predictions")
    print("  2. python -m monitoring.drift_detection")
    print("  3. streamlit run monitoring/dashboard.py")


if __name__ == "__main__":
    main()
