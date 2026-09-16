"""Pytest fixtures and configuration."""

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session
from fastapi.testclient import TestClient

from app.config import settings
from app.database import Base, get_db
from app.main import create_app


@pytest.fixture(scope="session")
def test_db_path():
    """Path for test database."""
    return "sqlite:///test.db"


@pytest.fixture(scope="session")
def test_engine(test_db_path):
    """Create test engine."""
    engine = create_engine(test_db_path, echo=False)
    Base.metadata.create_all(engine)
    yield engine
    engine.dispose()


@pytest.fixture
def db_session(test_engine):
    """Create a test database session."""
    session_factory = sessionmaker(bind=test_engine, class_=Session, expire_on_commit=False)
    session = session_factory()
    try:
        yield session
    finally:
        session.close()


@pytest.fixture
def client():
    """Create test client."""
    app = create_app()
    return TestClient(app)


@pytest.fixture
def sample_patient_data():
    """Sample patient data for testing."""
    return {"name": "Juan Pérez", "gender": "M"}


@pytest.fixture
def sample_classification_features():
    """Sample classification features for testing."""
    return {
        "age": 50,
        "ap_hi": 120,
        "ap_lo": 80,
        "cholesterol": 2,
        "gluc": 1,
        "smoke": False,
        "alco": False,
        "active": True,
        "height": 170,
        "weight": 70,
        "gender": "M",
    }