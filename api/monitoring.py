"""Monitoring module for tracking ML predictions and system health."""
import logging
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from pydantic import BaseModel

logger = logging.getLogger(__name__)


class PredictionLog(BaseModel):
    """Single prediction event for monitoring."""

    request_id: str
    timestamp: str
    user_id: Optional[int] = None
    input_anime_ids: list[int]
    input_count: int
    output_anime_ids: list[int]
    output_count: int
    latency_ms: float
    success: bool = True
    error_message: Optional[str] = None


class PredictionLogger:
    """Logs predictions to a JSON Lines file for analysis."""

    def __init__(self, log_dir: Path = Path("logs")):
        """Initialize logger with target directory."""
        self.log_dir = log_dir
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.log_file = self.log_dir / "predictions.jsonl"
        logger.info(f"PredictionLogger initialized: {self.log_file}")

    def generate_request_id(self) -> str:
        """Generate a unique request identifier."""
        return str(uuid.uuid4())[:8]

    def log(self, prediction: PredictionLog) -> None:
        """Append prediction entry to log file."""
        try:
            with open(self.log_file, "a", encoding="utf-8") as f:
                f.write(prediction.model_dump_json() + "\n")
        except Exception as e:
            logger.error(f"Failed to write prediction log: {e}")

    def create_log_entry(
        self,
        input_anime_ids: list[int],
        output_anime_ids: list[int],
        latency_ms: float,
        user_id: Optional[int] = None,
        success: bool = True,
        error_message: Optional[str] = None
    ) -> PredictionLog:
        """Create a PredictionLog with auto-generated timestamp and request ID."""
        return PredictionLog(
            request_id=self.generate_request_id(),
            timestamp=datetime.now(timezone.utc).isoformat(),
            user_id=user_id,
            input_anime_ids=input_anime_ids,
            input_count=len(input_anime_ids),
            output_anime_ids=output_anime_ids,
            output_count=len(output_anime_ids),
            latency_ms=latency_ms,
            success=success,
            error_message=error_message
        )


prediction_logger = PredictionLogger()


def get_prediction_logger() -> PredictionLogger:
    """Get the global prediction logger instance."""
    return prediction_logger
