"""Fixtures y configuración de pytest."""

import pytest
from sqlalchemy import create_engine, event
from sqlalchemy.orm import sessionmaker, Session
from fastapi.testclient import TestClient

from app.config import settings
from app.database import Base, get_db
from app.main import create_app


@pytest.fixture(scope="session")
def test_db_path():
    """Ruta de la base de datos de pruebas."""
    return "sqlite:///test.db"


@pytest.fixture(scope="session")
def test_engine(test_db_path):
    """Crea el motor de pruebas con FK habilitadas para SQLite."""
    engine = create_engine(test_db_path, echo=False)

    # SQLite NO aplica FK por defecto: PRAGMA foreign_keys=ON por conexión
    @event.listens_for(engine, "connect")
    def set_sqlite_pragma(dbapi_connection, connection_record):
        """Habilita el cumplimiento de FK en cada conexión SQLite."""
        cursor = dbapi_connection.cursor()
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.close()

    Base.metadata.create_all(engine)
    yield engine
    engine.dispose()


@pytest.fixture
def db_session(test_engine):
    """Crea una sesión de base de datos de pruebas."""
    session_factory = sessionmaker(bind=test_engine, class_=Session, expire_on_commit=False)
    session = session_factory()
    try:
        yield session
    finally:
        session.close()


@pytest.fixture
def client():
    """Crea el cliente de pruebas."""
    app = create_app()
    return TestClient(app)


@pytest.fixture
def sample_patient_data():
    """Datos de ejemplo de un paciente."""
    return {"name": "Juan Perez", "gender": "M"}


@pytest.fixture
def sample_classification_features():
    """Datos de ejemplo de clasificación (11 características clínicas)."""
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
