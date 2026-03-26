import logging

from sqlalchemy.orm import Session

from api.app_state import AppState
from api.exceptions import ConflictError, NotFoundError
from api.favorites.repository import FavoriteRepository
from api.favorites.schemas import FavoriteCheckResponse, FavoriteListResponse, FavoriteResponse

logger = logging.getLogger(__name__)


class FavoriteService:
    def __init__(self, db: Session, app_state: AppState):
        self.repo = FavoriteRepository(db)
        self.app_state = app_state

    def list_favorites(self, user_id: int) -> FavoriteListResponse:
        favorites = self.repo.get_by_user(user_id)
        enriched = [self._enrich(fav) for fav in favorites]
        return FavoriteListResponse(favorites=enriched, total=len(enriched))

    def add_favorite(self, user_id: int, anime_id: int) -> FavoriteResponse:
        if self.repo.find_by_anime_and_user(anime_id, user_id):
            raise ConflictError("Anime already in favorites")

        favorite = self.repo.create(user_id, anime_id)
        logger.info("User %d added anime %d to favorites", user_id, anime_id)
        return self._enrich(favorite)

    def remove_by_favorite_id(self, favorite_id: int, user_id: int) -> None:
        favorite = self.repo.find_by_id_and_user(favorite_id, user_id)
        if not favorite:
            raise NotFoundError("Favorite", favorite_id)
        self.repo.delete(favorite)
        logger.info("User %d removed favorite %d", user_id, favorite_id)

    def remove_by_anime_id(self, anime_id: int, user_id: int) -> None:
        favorite = self.repo.find_by_anime_and_user(anime_id, user_id)
        if not favorite:
            raise NotFoundError("Favorite for anime", anime_id)
        self.repo.delete(favorite)
        logger.info("User %d removed anime %d from favorites", user_id, anime_id)

    def check_favorite(self, anime_id: int, user_id: int) -> FavoriteCheckResponse:
        exists = self.repo.find_by_anime_and_user(anime_id, user_id) is not None
        return FavoriteCheckResponse(is_favorite=exists, anime_id=anime_id)

    def _enrich(self, fav) -> FavoriteResponse:
        info = self.app_state.get_anime_info(fav.anime_id)
        return FavoriteResponse(
            id=fav.id,
            anime_id=fav.anime_id,
            added_at=fav.added_at,
            name=info.get("name"),
            title_english=info.get("title_english"),
            image_url=info.get("image_url"),
        )
