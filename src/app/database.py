"""Database configuration and session management."""

from sqlalchemy import create_engine
from sqlalchemy.orm import DeclarativeBase, sessionmaker, Session

from app.config import settings


class Base(DeclarativeBase):
    """Base class for all SQLAlchemy models."""
    pass


# Create sync engine
engine = create_engine(
    settings.DATABASE_URL,
    echo=settings.DEBUG,
    future=True,
)

# Session factory
session_maker = sessionmaker(
    engine,
    class_=Session,
    expire_on_commit=False,
)


def get_db() -> Session:
    """Dependency for FastAPI to get DB session."""
    session = session_maker()
    try:
        yield session
    finally:
        session.close()


def init_db() -> None:
    """Initialize database tables on startup."""
    Base.metadata.create_all(engine)


def close_db() -> None:
    """Close database connections on shutdown."""
    engine.dispose()