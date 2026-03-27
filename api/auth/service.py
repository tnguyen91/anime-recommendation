import logging

from sqlalchemy.orm import Session

from api.auth.repository import UserRepository
from api.auth.security import create_access_token, get_password_hash, verify_password
from api.exceptions import AuthenticationError, ConflictError, ForbiddenError
from api.models import User

logger = logging.getLogger(__name__)

_DUMMY_HASH = get_password_hash("timing-safe-dummy")


class AuthService:
    def __init__(self, db: Session):
        self.repo = UserRepository(db)

    def register(self, email: str, password: str, username: str | None = None) -> User:
        if self.repo.find_by_email(email):
            raise ConflictError("Email already registered")

        if username and self.repo.find_by_username(username):
            raise ConflictError("Username already taken")

        hashed = get_password_hash(password)
        user = self.repo.create(email=email, hashed_password=hashed, username=username)
        logger.info("New user registered: id=%d", user.id)
        return user

    def login(self, email: str, password: str) -> str:
        user = self.repo.find_by_email(email)

        if not user:
            verify_password(password, _DUMMY_HASH)
            raise AuthenticationError("Incorrect email or password")

        if not verify_password(password, user.hashed_password):
            raise AuthenticationError("Incorrect email or password")

        if not user.is_active:
            raise ForbiddenError("Account is disabled")

        access_token = create_access_token(data={"sub": str(user.id), "email": user.email})
        logger.info("User logged in: id=%d", user.id)
        return access_token
