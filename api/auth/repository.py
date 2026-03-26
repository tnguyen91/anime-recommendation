from sqlalchemy.orm import Session

from api.models import User


class UserRepository:
    def __init__(self, db: Session):
        self.db = db

    def find_by_email(self, email: str) -> User | None:
        return self.db.query(User).filter(User.email == email).first()

    def find_by_username(self, username: str) -> User | None:
        return self.db.query(User).filter(User.username == username).first()

    def find_by_id(self, user_id: int) -> User | None:
        return self.db.query(User).filter(User.id == user_id).first()

    def create(self, email: str, hashed_password: str, username: str | None = None) -> User:
        user = User(
            email=email,
            username=username,
            hashed_password=hashed_password,
            is_active=True,
        )
        self.db.add(user)
        self.db.commit()
        self.db.refresh(user)
        return user
