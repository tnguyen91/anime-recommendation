"""
Pydantic schemas for authentication requests and responses.

Schemas define the shape of data for API requests and responses.
They provide automatic validation, serialization, and API documentation.

Note: Login uses OAuth2PasswordRequestForm (username + password form data),
not a JSON body, so there's no UserLogin schema needed.
"""
from pydantic import BaseModel, ConfigDict, EmailStr, Field


class UserCreate(BaseModel):
    """Schema for user registration request."""
    email: EmailStr
    password: str = Field(..., min_length=8, description="Password must be at least 8 characters")
    username: str | None = Field(None, min_length=3, max_length=50)


class UserResponse(BaseModel):
    """Schema for user response (excludes sensitive data like password)."""
    id: int
    email: str
    username: str | None
    is_active: bool

    model_config = ConfigDict(from_attributes=True)


class Token(BaseModel):
    """Schema for JWT token response after successful login."""
    access_token: str
    token_type: str = "bearer"


class TokenData(BaseModel):
    """Schema for data extracted from a decoded JWT token."""
    user_id: int | None = None
    email: str | None = None
