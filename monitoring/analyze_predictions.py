"""Prediction log analysis for computing metrics and generating reports."""
import argparse
import json
from collections import Counter
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

DEFAULT_LOG_FILE = Path("logs/predictions.jsonl")


def load_predictions(log_file: Path, days: Optional[int] = None) -> list[dict]:
    """Load prediction logs from JSON Lines file, optionally filtered by days."""
    if not log_file.exists():
        print(f"Log file not found: {log_file}")
        print("No predictions have been logged yet. Make some API calls first!")
        return []

    predictions = []
    cutoff_time = None

    if days is not None:
        cutoff_time = datetime.now(timezone.utc) - timedelta(days=days)

    with open(log_file, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue

            try:
                pred = json.loads(line)
                if cutoff_time:
                    pred_time = datetime.fromisoformat(pred["timestamp"].replace("Z", "+00:00"))
                    if pred_time < cutoff_time:
                        continue
                predictions.append(pred)
            except json.JSONDecodeError as e:
                print(f"Warning: Could not parse line {line_num}: {e}")

    return predictions


def compute_basic_metrics(predictions: list[dict]) -> dict:
    """Compute volume, latency, and error rate metrics."""
    if not predictions:
        return {"error": "No predictions to analyze"}

    latencies = [p["latency_ms"] for p in predictions]
    successes = [p["success"] for p in predictions]

    total_predictions = len(predictions)
    successful_predictions = sum(successes)
    failed_predictions = total_predictions - successful_predictions

    successful_latencies = [p["latency_ms"] for p in predictions if p["success"]]

    if successful_latencies:
        avg_latency = sum(successful_latencies) / len(successful_latencies)
        sorted_latencies = sorted(successful_latencies)
        p50_latency = sorted_latencies[len(sorted_latencies) // 2]
        p95_idx = int(len(sorted_latencies) * 0.95)
        p95_latency = sorted_latencies[min(p95_idx, len(sorted_latencies) - 1)]
        max_latency = max(successful_latencies)
        min_latency = min(successful_latencies)
    else:
        avg_latency = p50_latency = p95_latency = max_latency = min_latency = 0

    return {
        "total_predictions": total_predictions,
        "successful_predictions": successful_predictions,
        "failed_predictions": failed_predictions,
        "error_rate_percent": (failed_predictions / total_predictions) * 100,
        "latency_avg_ms": round(avg_latency, 2),
        "latency_p50_ms": round(p50_latency, 2),
        "latency_p95_ms": round(p95_latency, 2),
        "latency_min_ms": round(min_latency, 2),
        "latency_max_ms": round(max_latency, 2),
    }


def compute_input_metrics(predictions: list[dict]) -> dict:
    """Analyze input and output patterns for drift detection."""
    if not predictions:
        return {"error": "No predictions to analyze"}

    input_counts = [p["input_count"] for p in predictions]
    avg_input_count = sum(input_counts) / len(input_counts)
    input_count_distribution = Counter(input_counts)

    all_input_anime = []
    for p in predictions:
        all_input_anime.extend(p["input_anime_ids"])
    most_common_inputs = Counter(all_input_anime).most_common(10)

    output_counts = [p["output_count"] for p in predictions]
    avg_output_count = sum(output_counts) / len(output_counts)

    all_output_anime = []
    for p in predictions:
        all_output_anime.extend(p["output_anime_ids"])
    most_common_outputs = Counter(all_output_anime).most_common(10)

    return {
        "avg_input_count": round(avg_input_count, 2),
        "min_input_count": min(input_counts),
        "max_input_count": max(input_counts),
        "input_count_distribution": dict(sorted(input_count_distribution.items())),
        "most_common_input_anime_ids": most_common_inputs,
        "avg_output_count": round(avg_output_count, 2),
        "most_common_output_anime_ids": most_common_outputs,
    }


def compute_temporal_metrics(predictions: list[dict]) -> dict:
    """Analyze daily and hourly prediction patterns."""
    if not predictions:
        return {"error": "No predictions to analyze"}

    predictions_by_date = {}
    for p in predictions:
        ts = datetime.fromisoformat(p["timestamp"].replace("Z", "+00:00"))
        date_str = ts.strftime("%Y-%m-%d")

        if date_str not in predictions_by_date:
            predictions_by_date[date_str] = []
        predictions_by_date[date_str].append(p)

    daily_metrics = {}
    for date_str, day_preds in sorted(predictions_by_date.items()):
        latencies = [p["latency_ms"] for p in day_preds if p["success"]]
        errors = sum(1 for p in day_preds if not p["success"])

        daily_metrics[date_str] = {
            "count": len(day_preds),
            "errors": errors,
            "avg_latency_ms": round(sum(latencies) / len(latencies), 2) if latencies else 0,
        }

    predictions_by_hour = Counter()
    for p in predictions:
        ts = datetime.fromisoformat(p["timestamp"].replace("Z", "+00:00"))
        predictions_by_hour[ts.hour] += 1

    return {
        "daily_metrics": daily_metrics,
        "predictions_by_hour": dict(sorted(predictions_by_hour.items())),
    }


def detect_anomalies(predictions: list[dict], basic_metrics: dict) -> list[str]:
    """Check for high error rate, latency, or unusual input patterns."""
    warnings = []

    if not predictions:
        return ["No predictions to analyze"]

    error_rate = basic_metrics.get("error_rate_percent", 0)
    if error_rate > 5:
        warnings.append(f"HIGH ERROR RATE: {error_rate:.1f}% of predictions are failing")
    elif error_rate > 1:
        warnings.append(f"Elevated error rate: {error_rate:.1f}% (target: < 1%)")

    p95_latency = basic_metrics.get("latency_p95_ms", 0)
    if p95_latency > 1000:
        warnings.append(f"HIGH LATENCY: p95 is {p95_latency:.0f}ms (target: < 500ms)")
    elif p95_latency > 500:
        warnings.append(f"Elevated latency: p95 is {p95_latency:.0f}ms")

    input_counts = [p["input_count"] for p in predictions]
    if input_counts:
        avg_count = sum(input_counts) / len(input_counts)
        if avg_count < 2:
            warnings.append(f"Low average input count: {avg_count:.1f}")
        if max(input_counts) > 50:
            warnings.append(f"Very high input count detected: {max(input_counts)}")

    if not warnings:
        warnings.append("No anomalies detected - system looks healthy!")

    return warnings


def print_report(
    basic_metrics: dict,
    input_metrics: dict,
    temporal_metrics: dict,
    anomalies: list[str],
    log_file: Path,
    days: Optional[int]
):
    """Print formatted monitoring report to console."""
    print("\n" + "=" * 60)
    print("       ANIME RECOMMENDATION SYSTEM - MONITORING REPORT")
    print("=" * 60)

    print(f"\nLog file: {log_file}")
    if days:
        print(f"Time range: Last {days} days")
    print(f"Report generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    print("\n" + "-" * 60)
    print("HEALTH CHECK")
    print("-" * 60)
    for warning in anomalies:
        prefix = "[!]" if "HIGH" in warning or "Elevated" in warning else "[OK]"
        print(f"  {prefix} {warning}")

    print("\n" + "-" * 60)
    print("BASIC METRICS")
    print("-" * 60)
    if "error" not in basic_metrics:
        print(f"  Total predictions:    {basic_metrics['total_predictions']:,}")
        print(f"  Successful:           {basic_metrics['successful_predictions']:,}")
        print(f"  Failed:               {basic_metrics['failed_predictions']:,}")
        print(f"  Error rate:           {basic_metrics['error_rate_percent']:.2f}%")
        print()
        print(f"  Latency (avg):        {basic_metrics['latency_avg_ms']:.1f} ms")
        print(f"  Latency (p50/median): {basic_metrics['latency_p50_ms']:.1f} ms")
        print(f"  Latency (p95):        {basic_metrics['latency_p95_ms']:.1f} ms")
        print(f"  Latency (min/max):    {basic_metrics['latency_min_ms']:.1f} / {basic_metrics['latency_max_ms']:.1f} ms")

    print("\n" + "-" * 60)
    print("INPUT ANALYSIS")
    print("-" * 60)
    if "error" not in input_metrics:
        print(f"  Avg favorites per request: {input_metrics['avg_input_count']:.1f}")
        print(f"  Range: {input_metrics['min_input_count']} - {input_metrics['max_input_count']}")
        print()
        print("  Most commonly requested anime (by ID):")
        for anime_id, count in input_metrics["most_common_input_anime_ids"][:5]:
            print(f"    - ID {anime_id}: {count} requests")

    print("\n" + "-" * 60)
    print("OUTPUT ANALYSIS")
    print("-" * 60)
    if "error" not in input_metrics:
        print(f"  Avg recommendations per request: {input_metrics['avg_output_count']:.1f}")
        print()
        print("  Most frequently recommended anime (by ID):")
        for anime_id, count in input_metrics["most_common_output_anime_ids"][:5]:
            print(f"    - ID {anime_id}: recommended {count} times")

    print("\n" + "-" * 60)
    print("DAILY BREAKDOWN")
    print("-" * 60)
    if "error" not in temporal_metrics:
        daily = temporal_metrics.get("daily_metrics", {})
        if daily:
            print(f"  {'Date':<12} {'Count':>8} {'Errors':>8} {'Avg Latency':>12}")
            print(f"  {'-'*12} {'-'*8} {'-'*8} {'-'*12}")
            for date_str, metrics in list(daily.items())[-7:]:
                print(f"  {date_str:<12} {metrics['count']:>8} {metrics['errors']:>8} {metrics['avg_latency_ms']:>10.1f}ms")

    print("\n" + "=" * 60)
    print("END OF REPORT")
    print("=" * 60 + "\n")


def main():
    """CLI entry point for prediction log analysis."""
    parser = argparse.ArgumentParser(
        description="Analyze prediction logs for the anime recommendation system"
    )
    parser.add_argument(
        "--log-file",
        type=Path,
        default=DEFAULT_LOG_FILE,
        help=f"Path to the prediction log file (default: {DEFAULT_LOG_FILE})"
    )
    parser.add_argument(
        "--days",
        type=int,
        default=None,
        help="Only analyze logs from the last N days (default: all)"
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output metrics as JSON instead of formatted report"
    )

    args = parser.parse_args()
    predictions = load_predictions(args.log_file, args.days)

    if not predictions:
        return

    basic_metrics = compute_basic_metrics(predictions)
    input_metrics = compute_input_metrics(predictions)
    temporal_metrics = compute_temporal_metrics(predictions)
    anomalies = detect_anomalies(predictions, basic_metrics)

    if args.json:
        output = {
            "basic_metrics": basic_metrics,
            "input_metrics": input_metrics,
            "temporal_metrics": temporal_metrics,
            "anomalies": anomalies,
        }
        print(json.dumps(output, indent=2, default=str))
    else:
        print_report(
            basic_metrics,
            input_metrics,
            temporal_metrics,
            anomalies,
            args.log_file,
            args.days
        )


if __name__ == "__main__":
    main()
