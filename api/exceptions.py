import logging

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

logger = logging.getLogger(__name__)


class AppError(Exception):
    def __init__(self, message: str, status_code: int = 500):
        self.message = message
        self.status_code = status_code
        super().__init__(message)


class NotFoundError(AppError):
    def __init__(self, resource: str, identifier: str | int | None = None):
        detail = f"{resource} not found"
        if identifier is not None:
            detail = f"{resource} with id '{identifier}' not found"
        super().__init__(detail, status_code=404)


class ConflictError(AppError):
    def __init__(self, message: str):
        super().__init__(message, status_code=409)


class ValidationError(AppError):
    def __init__(self, message: str):
        super().__init__(message, status_code=400)


class AuthenticationError(AppError):
    def __init__(self, message: str = "Invalid credentials"):
        super().__init__(message, status_code=401)


class ForbiddenError(AppError):
    def __init__(self, message: str = "Access denied"):
        super().__init__(message, status_code=403)


class ServiceUnavailableError(AppError):
    def __init__(self, message: str = "Service temporarily unavailable"):
        super().__init__(message, status_code=503)


def _error_response(status_code: int, message: str, request_id: str | None = None) -> JSONResponse:
    body: dict = {
        "error": {
            "code": status_code,
            "message": message,
        }
    }
    if request_id:
        body["error"]["request_id"] = request_id
    return JSONResponse(status_code=status_code, content=body)


def register_exception_handlers(app: FastAPI) -> None:
    @app.exception_handler(AppError)
    async def app_error_handler(request: Request, exc: AppError):
        request_id = getattr(request.state, "request_id", None)
        return _error_response(exc.status_code, exc.message, request_id)

    @app.exception_handler(StarletteHTTPException)
    async def http_exception_handler(request: Request, exc: StarletteHTTPException):
        request_id = getattr(request.state, "request_id", None)
        return _error_response(exc.status_code, str(exc.detail), request_id)

    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(request: Request, exc: RequestValidationError):
        request_id = getattr(request.state, "request_id", None)
        errors = exc.errors()
        message = "; ".join(f"{'.'.join(str(loc) for loc in e['loc'])}: {e['msg']}" for e in errors)
        return _error_response(422, message, request_id)
