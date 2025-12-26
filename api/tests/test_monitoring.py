"""Unit tests for the monitoring module."""
import json
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import pytest

# Add project root to path for monitoring module imports
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from api.monitoring import PredictionLog, PredictionLogger


class TestPredictionLog:
    """Tests for PredictionLog model."""

    def test_create_successful_prediction(self):
        """Test creating a successful prediction log."""
        log = PredictionLog(
            request_id="abc12345",
            timestamp="2025-12-26T10:00:00Z",
            user_id=42,
            input_anime_ids=[1, 2, 3],
            input_count=3,
            output_anime_ids=[10, 20, 30, 40, 50],
            output_count=5,
            latency_ms=150.5,
            success=True,
            error_message=None
        )

        assert log.request_id == "abc12345"
        assert log.user_id == 42
        assert log.input_count == 3
        assert log.output_count == 5
        assert log.latency_ms == 150.5
        assert log.success is True
        assert log.error_message is None

    def test_create_failed_prediction(self):
        """Test creating a failed prediction log."""
        log = PredictionLog(
            request_id="def67890",
            timestamp="2025-12-26T10:00:00Z",
            user_id=None,
            input_anime_ids=[1, 2],
            input_count=2,
            output_anime_ids=[],
            output_count=0,
            latency_ms=50.0,
            success=False,
            error_message="Model inference failed"
        )

        assert log.success is False
        assert log.error_message == "Model inference failed"
        assert log.output_count == 0

    def test_prediction_log_to_json(self):
        """Test serializing prediction log to JSON."""
        log = PredictionLog(
            request_id="test123",
            timestamp="2025-12-26T10:00:00Z",
            user_id=None,
            input_anime_ids=[1, 2, 3],
            input_count=3,
            output_anime_ids=[10, 20],
            output_count=2,
            latency_ms=100.0,
            success=True
        )

        json_str = log.model_dump_json()
        parsed = json.loads(json_str)

        assert parsed["request_id"] == "test123"
        assert parsed["input_count"] == 3
        assert parsed["latency_ms"] == 100.0


class TestPredictionLogger:
    """Tests for PredictionLogger class."""

    def test_logger_creates_directory(self):
        """Test that logger creates log directory if it doesn't exist."""
        with tempfile.TemporaryDirectory() as tmpdir:
            log_dir = Path(tmpdir) / "nested" / "logs"
            logger = PredictionLogger(log_dir=log_dir)

            assert log_dir.exists()
            assert logger.log_file == log_dir / "predictions.jsonl"

    def test_generate_request_id(self):
        """Test that request IDs are generated correctly."""
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = PredictionLogger(log_dir=Path(tmpdir))

            id1 = logger.generate_request_id()
            id2 = logger.generate_request_id()

            assert len(id1) == 8
            assert len(id2) == 8
            assert id1 != id2  # Should be unique

    def test_create_log_entry(self):
        """Test creating a log entry with auto-generated fields."""
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = PredictionLogger(log_dir=Path(tmpdir))

            entry = logger.create_log_entry(
                input_anime_ids=[1, 2, 3],
                output_anime_ids=[10, 20, 30, 40, 50],
                latency_ms=200.5,
                user_id=42
            )

            assert len(entry.request_id) == 8
            assert entry.input_count == 3
            assert entry.output_count == 5
            assert entry.latency_ms == 200.5
            assert entry.user_id == 42
            assert entry.success is True
            # Timestamp should be recent
            ts = datetime.fromisoformat(entry.timestamp.replace("Z", "+00:00"))
            now = datetime.now(timezone.utc)
            assert (now - ts).total_seconds() < 5

    def test_log_writes_to_file(self):
        """Test that logging writes entries to the file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = PredictionLogger(log_dir=Path(tmpdir))

            # Log multiple entries
            for i in range(3):
                entry = logger.create_log_entry(
                    input_anime_ids=[i, i + 1],
                    output_anime_ids=[i * 10],
                    latency_ms=100.0 + i
                )
                logger.log(entry)

            # Read and verify
            with open(logger.log_file) as f:
                lines = f.readlines()

            assert len(lines) == 3

            # Verify each line is valid JSON
            for i, line in enumerate(lines):
                data = json.loads(line.strip())
                assert data["input_count"] == 2
                assert data["latency_ms"] == 100.0 + i

    def test_log_handles_errors_gracefully(self):
        """Test that logging errors don't crash the application."""
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = PredictionLogger(log_dir=Path(tmpdir))

            # Make the log file a directory to cause write errors
            logger.log_file.mkdir(parents=True, exist_ok=True)

            entry = logger.create_log_entry(
                input_anime_ids=[1],
                output_anime_ids=[10],
                latency_ms=100.0
            )

            # This should not raise an exception
            logger.log(entry)


class TestLogFileOperations:
    """Tests for log file operations using the PredictionLogger."""

    def test_multiple_loggers_append_to_same_file(self):
        """Test that multiple logger instances can append to the same file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            log_dir = Path(tmpdir)

            # First logger writes entries
            logger1 = PredictionLogger(log_dir=log_dir)
            entry1 = logger1.create_log_entry(
                input_anime_ids=[1],
                output_anime_ids=[10],
                latency_ms=100.0
            )
            logger1.log(entry1)

            # Second logger (simulating app restart) appends
            logger2 = PredictionLogger(log_dir=log_dir)
            entry2 = logger2.create_log_entry(
                input_anime_ids=[2],
                output_anime_ids=[20],
                latency_ms=200.0
            )
            logger2.log(entry2)

            # Verify both entries are in the file
            with open(log_dir / "predictions.jsonl") as f:
                lines = f.readlines()

            assert len(lines) == 2

    def test_log_entry_timestamps_are_utc(self):
        """Test that timestamps are in UTC format."""
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = PredictionLogger(log_dir=Path(tmpdir))

            entry = logger.create_log_entry(
                input_anime_ids=[1],
                output_anime_ids=[10],
                latency_ms=100.0
            )

            # Should end with Z (UTC) or +00:00
            assert entry.timestamp.endswith("Z") or "+00:00" in entry.timestamp

    def test_log_preserves_all_anime_ids(self):
        """Test that all anime IDs are preserved in the log."""
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = PredictionLogger(log_dir=Path(tmpdir))

            input_ids = [1, 5, 20, 50, 100, 200, 500]
            output_ids = [10, 25, 30, 45, 60, 75, 80, 90, 95, 99]

            entry = logger.create_log_entry(
                input_anime_ids=input_ids,
                output_anime_ids=output_ids,
                latency_ms=150.0
            )
            logger.log(entry)

            # Read back and verify
            with open(logger.log_file) as f:
                data = json.loads(f.readline())

            assert data["input_anime_ids"] == input_ids
            assert data["output_anime_ids"] == output_ids
            assert data["input_count"] == len(input_ids)
            assert data["output_count"] == len(output_ids)
