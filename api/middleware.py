import contextvars
import logging
import time
import uuid

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

request_id_ctx: contextvars.ContextVar[str] = contextvars.ContextVar("request_id", default="-")


class RequestIDFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        record.request_id = request_id_ctx.get("-")  # type: ignore[attr-defined]
        return True


class RequestIDMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next) -> Response:
        request_id = request.headers.get("X-Request-ID") or uuid.uuid4().hex[:12]
        request.state.request_id = request_id
        token = request_id_ctx.set(request_id)

        start = time.monotonic()
        response = await call_next(request)
        duration = time.monotonic() - start

        response.headers["X-Request-ID"] = request_id

        if request.url.path not in ("/health", "/"):
            logger = logging.getLogger("api.access")
            logger.info(
                "%s %s | %d | %.3fs",
                request.method,
                request.url.path,
                response.status_code,
                duration,
            )

        request_id_ctx.reset(token)
        return response


def configure_logging(log_level: str = "INFO") -> None:
    root = logging.getLogger()
    root.setLevel(getattr(logging, log_level, logging.INFO))

    handler = logging.StreamHandler()
    handler.setFormatter(
        logging.Formatter(
            "%(asctime)s | %(levelname)s | [%(request_id)s] %(name)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    )
    handler.addFilter(RequestIDFilter())

    root.handlers.clear()
    root.addHandler(handler)
