import logging

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from api.app_state import AppState
from api.auth.dependencies import get_current_user
from api.dependencies import get_app_state, get_db
from api.favorites.schemas import (
    FavoriteCheckResponse,
    FavoriteCreate,
    FavoriteListResponse,
    FavoriteResponse,
    MessageResponse,
)
from api.favorites.service import FavoriteService
from api.models import User

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/favorites", tags=["Favorites"])


@router.get("", response_model=FavoriteListResponse)
def list_favorites(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
    app_state: AppState = Depends(get_app_state),
):
    service = FavoriteService(db, app_state)
    return service.list_favorites(current_user.id)


@router.post("", response_model=FavoriteResponse, status_code=201)
def add_favorite(
    favorite_data: FavoriteCreate,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
    app_state: AppState = Depends(get_app_state),
):
    service = FavoriteService(db, app_state)
    return service.add_favorite(current_user.id, favorite_data.anime_id)


@router.delete("/{favorite_id}", response_model=MessageResponse)
def remove_favorite(
    favorite_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
    app_state: AppState = Depends(get_app_state),
):
    service = FavoriteService(db, app_state)
    service.remove_by_favorite_id(favorite_id, current_user.id)
    return MessageResponse(message="Favorite removed successfully")


@router.delete("/anime/{anime_id}", response_model=MessageResponse)
def remove_favorite_by_anime(
    anime_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
    app_state: AppState = Depends(get_app_state),
):
    service = FavoriteService(db, app_state)
    service.remove_by_anime_id(anime_id, current_user.id)
    return MessageResponse(message="Favorite removed successfully")


@router.get("/check/{anime_id}", response_model=FavoriteCheckResponse)
def check_favorite(
    anime_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
    app_state: AppState = Depends(get_app_state),
):
    service = FavoriteService(db, app_state)
    return service.check_favorite(anime_id, current_user.id)
