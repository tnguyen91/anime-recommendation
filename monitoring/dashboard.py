"""Streamlit monitoring dashboard for the anime recommendation system."""
import json
from collections import Counter
from datetime import datetime, timedelta, timezone
from pathlib import Path

import streamlit as st

st.set_page_config(
    page_title="Anime Recommender Monitoring",
    page_icon="🎯",
    layout="wide"
)

LOG_FILE = Path("logs/predictions.jsonl")


def load_predictions(days: int = 30) -> list[dict]:
    """Load predictions from the last N days."""
    if not LOG_FILE.exists():
        return []

    cutoff = datetime.now(timezone.utc) - timedelta(days=days)
    predictions = []

    with open(LOG_FILE, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                pred = json.loads(line)
                ts = datetime.fromisoformat(pred["timestamp"].replace("Z", "+00:00"))
                if ts >= cutoff:
                    predictions.append(pred)
            except (json.JSONDecodeError, KeyError):
                continue

    return predictions


def main():
    """Render the monitoring dashboard."""
    st.title("🎯 Anime Recommender - Monitoring Dashboard")
    st.markdown("Real-time view of your ML system's health")

    st.sidebar.header("Settings")
    days_to_show = st.sidebar.slider("Days to analyze", 1, 30, 7)

    predictions = load_predictions(days=days_to_show)

    if not predictions:
        st.warning("No prediction logs found. Make some API calls to generate data!")
        st.info(f"Looking for logs at: {LOG_FILE.absolute()}")
        return

    # Key metrics
    st.header("Key Metrics")

    col1, col2, col3, col4 = st.columns(4)

    total = len(predictions)
    errors = sum(1 for p in predictions if not p["success"])
    error_rate = (errors / total) * 100 if total > 0 else 0

    latencies = [p["latency_ms"] for p in predictions if p["success"]]
    avg_latency = sum(latencies) / len(latencies) if latencies else 0
    p95_latency = sorted(latencies)[int(len(latencies) * 0.95)] if len(latencies) > 1 else avg_latency

    with col1:
        st.metric(
            label="Total Predictions",
            value=f"{total:,}",
            delta=f"Last {days_to_show} days"
        )

    with col2:
        st.metric(
            label="Error Rate",
            value=f"{error_rate:.1f}%",
            delta="Target: < 1%",
            delta_color="inverse" if error_rate > 1 else "normal"
        )

    with col3:
        st.metric(
            label="Avg Latency",
            value=f"{avg_latency:.0f} ms",
            delta=f"p95: {p95_latency:.0f} ms"
        )

    with col4:
        avg_inputs = sum(p["input_count"] for p in predictions) / len(predictions)
        st.metric(
            label="Avg Favorites/Request",
            value=f"{avg_inputs:.1f}"
        )

    # Health status
    st.header("Health Status")

    health_checks = []

    if error_rate > 5:
        health_checks.append(("Error Rate", "critical", f"{error_rate:.1f}% errors"))
    elif error_rate > 1:
        health_checks.append(("Error Rate", "warning", f"{error_rate:.1f}% errors"))
    else:
        health_checks.append(("Error Rate", "healthy", f"{error_rate:.1f}% errors"))

    if p95_latency > 1000:
        health_checks.append(("Latency", "critical", f"p95: {p95_latency:.0f}ms"))
    elif p95_latency > 500:
        health_checks.append(("Latency", "warning", f"p95: {p95_latency:.0f}ms"))
    else:
        health_checks.append(("Latency", "healthy", f"p95: {p95_latency:.0f}ms"))

    cols = st.columns(len(health_checks))
    for i, (name, status, detail) in enumerate(health_checks):
        with cols[i]:
            if status == "healthy":
                st.success(f"**{name}**: {detail}")
            elif status == "warning":
                st.warning(f"**{name}**: {detail}")
            else:
                st.error(f"**{name}**: {detail}")

    # Trends
    st.header("Trends Over Time")

    daily_data = {}
    for p in predictions:
        date = p["timestamp"][:10]
        if date not in daily_data:
            daily_data[date] = {"count": 0, "errors": 0, "latencies": []}
        daily_data[date]["count"] += 1
        if not p["success"]:
            daily_data[date]["errors"] += 1
        else:
            daily_data[date]["latencies"].append(p["latency_ms"])

    if daily_data:
        dates = sorted(daily_data.keys())

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Daily Prediction Volume")
            volume_data = {d: daily_data[d]["count"] for d in dates}
            st.bar_chart(volume_data)

        with col2:
            st.subheader("Daily Average Latency (ms)")
            latency_data = {}
            for d in dates:
                lats = daily_data[d]["latencies"]
                latency_data[d] = sum(lats) / len(lats) if lats else 0
            st.line_chart(latency_data)

    # Anime analysis
    st.header("Anime Analysis")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Most Requested Anime (Input)")
        input_counter = Counter()
        for p in predictions:
            input_counter.update(p["input_anime_ids"])

        top_inputs = input_counter.most_common(10)
        if top_inputs:
            input_data = {f"ID {aid}": count for aid, count in top_inputs}
            st.bar_chart(input_data)
        else:
            st.info("No input data yet")

    with col2:
        st.subheader("Most Recommended Anime (Output)")
        output_counter = Counter()
        for p in predictions:
            output_counter.update(p["output_anime_ids"])

        top_outputs = output_counter.most_common(10)
        if top_outputs:
            output_data = {f"ID {aid}": count for aid, count in top_outputs}
            st.bar_chart(output_data)
        else:
            st.info("No output data yet")

    # Input distribution
    st.header("Input Size Distribution")

    input_counts = Counter(p["input_count"] for p in predictions)
    if input_counts:
        count_data = {f"{k} anime": v for k, v in sorted(input_counts.items())}
        st.bar_chart(count_data)

    # Recent predictions
    st.header("Recent Predictions")

    recent = predictions[-10:][::-1]
    for p in recent:
        ts = p["timestamp"][:19].replace("T", " ")
        status_emoji = "✅" if p["success"] else "❌"

        with st.expander(f"{status_emoji} {ts} - {p['input_count']} inputs → {p['output_count']} outputs"):
            col1, col2, col3 = st.columns(3)
            with col1:
                st.write(f"**Request ID:** {p['request_id']}")
                st.write(f"**Latency:** {p['latency_ms']:.1f} ms")
            with col2:
                st.write(f"**Input IDs:** {p['input_anime_ids'][:5]}...")
            with col3:
                st.write(f"**Output IDs:** {p['output_anime_ids'][:5]}...")

            if not p["success"]:
                st.error(f"Error: {p.get('error_message', 'Unknown')}")

    st.markdown("---")
    st.caption(f"Data from: {LOG_FILE.absolute()}")
    st.caption(f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    main()
