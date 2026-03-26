from fastapi import Request
from slowapi import Limiter
from slowapi.util import get_remote_address

from api.app_state import AppState
from api.database import get_db

limiter = Limiter(key_func=get_remote_address)


def get_app_state(request: Request) -> AppState:
    return request.app.state.app_state


__all__ = ["get_app_state", "get_db", "limiter"]
