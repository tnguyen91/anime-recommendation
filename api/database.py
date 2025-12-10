"""
Database configuration and session management.

This module sets up SQLAlchemy to connect to PostgreSQL and provides
a dependency function for FastAPI to inject database sessions into endpoints.

Usage:
    from api.database import get_db, Base
    
    # In your models:
    class User(Base):
        __tablename__ = "users"
        ...
    
    # In your endpoints:
    @router.get("/users")
    def get_users(db: Session = Depends(get_db)):
        return db.query(User).all()
"""
import os
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base

DATABASE_URL = os.getenv("DATABASE_URL")

Base = declarative_base()

engine = None
SessionLocal = None

if DATABASE_URL:
    engine = create_engine(
        DATABASE_URL,
        pool_pre_ping=True,
        pool_size=5,
        max_overflow=10,
    )
    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def get_db():
    """
    FastAPI dependency that provides a database session.
    
    Yields a SQLAlchemy session and ensures it's closed after the request,
    even if an exception occurs.
    
    Raises:
        RuntimeError: If DATABASE_URL environment variable is not set.
    """
    if SessionLocal is None:
        raise RuntimeError("Database not configured. Set DATABASE_URL environment variable.")
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
