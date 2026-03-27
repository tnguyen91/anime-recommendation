from sqlalchemy.orm import Session

from api.models import UserFavorite


class FavoriteRepository:
    def __init__(self, db: Session):
        self.db = db

    def get_by_user(self, user_id: int) -> list[UserFavorite]:
        return (
            self.db.query(UserFavorite)
            .filter(UserFavorite.user_id == user_id)
            .order_by(UserFavorite.added_at.desc())
            .all()
        )

    def find_by_id_and_user(self, favorite_id: int, user_id: int) -> UserFavorite | None:
        return (
            self.db.query(UserFavorite).filter(UserFavorite.id == favorite_id, UserFavorite.user_id == user_id).first()
        )

    def find_by_anime_and_user(self, anime_id: int, user_id: int) -> UserFavorite | None:
        return (
            self.db.query(UserFavorite)
            .filter(UserFavorite.anime_id == anime_id, UserFavorite.user_id == user_id)
            .first()
        )

    def create(self, user_id: int, anime_id: int) -> UserFavorite:
        favorite = UserFavorite(user_id=user_id, anime_id=anime_id)
        self.db.add(favorite)
        self.db.commit()
        self.db.refresh(favorite)
        return favorite

    def delete(self, favorite: UserFavorite) -> None:
        self.db.delete(favorite)
        self.db.commit()
