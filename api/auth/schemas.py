"""Pydantic schemas for authentication requests and responses."""
from pydantic import BaseModel, ConfigDict, EmailStr, Field


class UserCreate(BaseModel):
    """Schema for user registration request."""
    email: EmailStr
    password: str = Field(..., min_length=8, description="Password must be at least 8 characters")
    username: str | None = Field(None, min_length=3, max_length=50)


class UserLogin(BaseModel):
    """Schema for user login request."""
    email: EmailStr
    password: str


class UserResponse(BaseModel):
    """Schema for user response (excludes password)."""
    id: int
    email: str
    username: str | None
    is_active: bool

    model_config = ConfigDict(from_attributes=True)


class Token(BaseModel):
    """Schema for JWT token response."""
    access_token: str
    token_type: str = "bearer"


class TokenData(BaseModel):
    """Schema for decoded token data."""
    user_id: int | None = None
    email: str | None = None
