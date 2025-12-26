"""Drift detection for ML prediction patterns."""
import argparse
import json
from collections import Counter
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional


def load_predictions_by_period(
    log_file: Path,
    reference_days: int = 30,
    current_days: int = 7
) -> tuple[list[dict], list[dict]]:
    """Load predictions split into reference and current periods."""
    if not log_file.exists():
        return [], []

    now = datetime.now(timezone.utc)
    current_cutoff = now - timedelta(days=current_days)
    reference_cutoff = now - timedelta(days=reference_days)

    reference_predictions = []
    current_predictions = []

    with open(log_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            try:
                pred = json.loads(line)
                ts = datetime.fromisoformat(pred["timestamp"].replace("Z", "+00:00"))

                if ts >= current_cutoff:
                    current_predictions.append(pred)
                elif ts >= reference_cutoff:
                    reference_predictions.append(pred)

            except (json.JSONDecodeError, KeyError):
                continue

    return reference_predictions, current_predictions


def compute_distribution_stats(predictions: list[dict]) -> dict:
    """Compute summary statistics for a set of predictions."""
    if not predictions:
        return {}

    input_counts = [p["input_count"] for p in predictions]
    latencies = [p["latency_ms"] for p in predictions if p["success"]]

    input_anime_counter = Counter()
    output_anime_counter = Counter()
    for p in predictions:
        input_anime_counter.update(p["input_anime_ids"])
        output_anime_counter.update(p["output_anime_ids"])

    return {
        "count": len(predictions),
        "avg_input_count": sum(input_counts) / len(input_counts) if input_counts else 0,
        "avg_latency": sum(latencies) / len(latencies) if latencies else 0,
        "error_rate": sum(1 for p in predictions if not p["success"]) / len(predictions),
        "top_input_anime": input_anime_counter.most_common(20),
        "top_output_anime": output_anime_counter.most_common(20),
    }


def compute_overlap(list1: list[tuple], list2: list[tuple]) -> float:
    """Compute Jaccard similarity between two ranked lists (0 to 1)."""
    if not list1 or not list2:
        return 0.0

    set1 = set(item[0] for item in list1)
    set2 = set(item[0] for item in list2)

    if not set1 or not set2:
        return 0.0

    intersection = len(set1 & set2)
    union = len(set1 | set2)

    return intersection / union if union > 0 else 0.0


def detect_drift(
    log_file: Path = Path("logs/predictions.jsonl"),
    reference_days: int = 30,
    current_days: int = 7,
    thresholds: Optional[dict] = None
) -> dict:
    """Compare current predictions to reference period and detect drift."""
    if thresholds is None:
        thresholds = {
            "input_count_change_percent": 30,
            "latency_change_percent": 50,
            "error_rate_change": 0.05,
            "anime_overlap_min": 0.5,
        }

    reference_preds, current_preds = load_predictions_by_period(
        log_file, reference_days, current_days
    )

    result = {
        "reference_period_days": reference_days,
        "current_period_days": current_days,
        "reference_count": len(reference_preds),
        "current_count": len(current_preds),
        "drift_detected": False,
        "alerts": [],
        "metrics": {},
    }

    if len(reference_preds) < 10:
        result["alerts"].append("Not enough reference data (need at least 10 predictions)")
        return result

    if len(current_preds) < 5:
        result["alerts"].append("Not enough current data (need at least 5 predictions)")
        return result

    ref_stats = compute_distribution_stats(reference_preds)
    cur_stats = compute_distribution_stats(current_preds)

    # Check input count drift
    if ref_stats["avg_input_count"] > 0:
        input_count_change = (
            (cur_stats["avg_input_count"] - ref_stats["avg_input_count"])
            / ref_stats["avg_input_count"]
            * 100
        )
        result["metrics"]["input_count_change_percent"] = round(input_count_change, 1)

        if abs(input_count_change) > thresholds["input_count_change_percent"]:
            direction = "increased" if input_count_change > 0 else "decreased"
            result["alerts"].append(
                f"INPUT DRIFT: Average favorites per request {direction} by "
                f"{abs(input_count_change):.1f}% (from {ref_stats['avg_input_count']:.1f} "
                f"to {cur_stats['avg_input_count']:.1f})"
            )
            result["drift_detected"] = True

    # Check latency drift
    if ref_stats["avg_latency"] > 0:
        latency_change = (
            (cur_stats["avg_latency"] - ref_stats["avg_latency"])
            / ref_stats["avg_latency"]
            * 100
        )
        result["metrics"]["latency_change_percent"] = round(latency_change, 1)

        if latency_change > thresholds["latency_change_percent"]:
            result["alerts"].append(
                f"LATENCY DRIFT: Average latency increased by {latency_change:.1f}% "
                f"(from {ref_stats['avg_latency']:.0f}ms to {cur_stats['avg_latency']:.0f}ms)"
            )
            result["drift_detected"] = True

    # Check error rate drift
    error_rate_change = cur_stats["error_rate"] - ref_stats["error_rate"]
    result["metrics"]["error_rate_change"] = round(error_rate_change, 3)

    if error_rate_change > thresholds["error_rate_change"]:
        result["alerts"].append(
            f"ERROR RATE DRIFT: Error rate increased by {error_rate_change * 100:.1f}% "
            f"(from {ref_stats['error_rate'] * 100:.1f}% to {cur_stats['error_rate'] * 100:.1f}%)"
        )
        result["drift_detected"] = True

    # Check anime popularity drift
    input_overlap = compute_overlap(
        ref_stats["top_input_anime"],
        cur_stats["top_input_anime"]
    )
    result["metrics"]["input_anime_overlap"] = round(input_overlap, 2)

    if input_overlap < thresholds["anime_overlap_min"]:
        result["alerts"].append(
            f"ANIME POPULARITY DRIFT: Only {input_overlap * 100:.0f}% overlap in most "
            f"requested anime between reference and current periods"
        )
        result["drift_detected"] = True

    output_overlap = compute_overlap(
        ref_stats["top_output_anime"],
        cur_stats["top_output_anime"]
    )
    result["metrics"]["output_anime_overlap"] = round(output_overlap, 2)

    if output_overlap < thresholds["anime_overlap_min"]:
        result["alerts"].append(
            f"RECOMMENDATION DRIFT: Only {output_overlap * 100:.0f}% overlap in most "
            f"recommended anime between reference and current periods"
        )
        result["drift_detected"] = True

    if not result["drift_detected"]:
        result["alerts"].append("No significant drift detected - model inputs look stable")

    return result


def print_drift_report(result: dict):
    """Print formatted drift detection report to console."""
    print("\n" + "=" * 60)
    print("           DRIFT DETECTION REPORT")
    print("=" * 60)

    print(f"\nReference period: last {result['reference_period_days']} days "
          f"({result['reference_count']} predictions)")
    print(f"Current period: last {result['current_period_days']} days "
          f"({result['current_count']} predictions)")

    print("\n" + "-" * 60)
    status = "DRIFT DETECTED" if result["drift_detected"] else "STABLE"
    print(f"Status: {status}")
    print("-" * 60)

    for alert in result["alerts"]:
        prefix = "[!]" if "DRIFT" in alert else "[OK]"
        print(f"  {prefix} {alert}")

    if result["metrics"]:
        print("\n" + "-" * 60)
        print("Metrics:")
        print("-" * 60)
        for key, value in result["metrics"].items():
            print(f"  {key}: {value}")

    print("\n" + "=" * 60 + "\n")


def main():
    """CLI entry point for drift detection."""
    parser = argparse.ArgumentParser(description="Detect drift in prediction patterns")
    parser.add_argument(
        "--log-file",
        type=Path,
        default=Path("logs/predictions.jsonl"),
        help="Path to prediction log file"
    )
    parser.add_argument(
        "--reference-days",
        type=int,
        default=30,
        help="Days to use as reference period"
    )
    parser.add_argument(
        "--current-days",
        type=int,
        default=7,
        help="Days to use as current period"
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output as JSON"
    )

    args = parser.parse_args()

    result = detect_drift(
        log_file=args.log_file,
        reference_days=args.reference_days,
        current_days=args.current_days
    )

    if args.json:
        print(json.dumps(result, indent=2))
    else:
        print_drift_report(result)


if __name__ == "__main__":
    main()
