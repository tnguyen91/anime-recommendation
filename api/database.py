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
        max_overflow=10
    )
    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def get_db():
    """Dependency to get database session."""
    if SessionLocal is None:
        raise RuntimeError("Database not configured. Set DATABASE_URL environment variable.")
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
