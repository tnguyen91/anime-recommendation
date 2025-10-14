from datetime import datetime
from sqlalchemy import Column, Integer, String, DateTime, ForeignKey, Boolean
from sqlalchemy.orm import relationship
from api.database import Base


class User(Base):
    __tablename__ = "users"  
    
    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    
    email = Column(String(255), unique=True, index=True, nullable=False)
    
    username = Column(String(100), unique=True, index=True, nullable=True)
    
    hashed_password = Column(String(255), nullable=False)
    
    is_active = Column(Boolean, default=True, nullable=False)
    
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    
    favorites = relationship("UserFavorite", back_populates="user", cascade="all, delete-orphan")
    recommendations = relationship("RecommendationHistory", back_populates="user", cascade="all, delete-orphan")
    
    def __repr__(self):
        return f"<User(id={self.id}, email={self.email})>"


class UserFavorite(Base):
    __tablename__ = "user_favorites"
    
    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    
    user_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True)
    
    anime_id = Column(Integer, nullable=False, index=True)

    added_at = Column(DateTime, default=datetime.timezone.utc, nullable=False)

    user = relationship("User", back_populates="favorites")
    
    def __repr__(self):
        return f"<UserFavorite(user_id={self.user_id}, anime_id={self.anime_id})>"


class RecommendationHistory(Base):
    __tablename__ = "recommendation_history"
    
    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    
    user_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True)
    
    anime_id = Column(Integer, nullable=False, index=True)

    recommended_at = Column(DateTime, default=datetime.timezone.utc, nullable=False)

    clicked = Column(Boolean, default=False, nullable=False)
    
    favorited = Column(Boolean, default=False, nullable=False)
    
    user = relationship("User", back_populates="recommendations")
    
    def __repr__(self):
        return f"<RecommendationHistory(user_id={self.user_id}, anime_id={self.anime_id})>"
