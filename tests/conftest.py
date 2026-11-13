"""Fixtures y configuración de pytest."""

import sys
from pathlib import Path

import pytest
from sqlalchemy import create_engine, event
from sqlalchemy.orm import sessionmaker, Session
from fastapi.testclient import TestClient

from app.config import settings
from app.database import Base, get_db
from app.main import create_app

# train.py vive en la raíz del repo (no es parte del paquete app)
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from train import DEFAULT_DATA_PATH, train


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

@pytest.fixture(scope="session")
def trained(tmp_path_factory):
    """Entrena el modelo UNA vez por sesion y comparte el artefacto temporal.

    Infraestructura compartida: la consumen los tests de entrenamiento
    (test_train.py) y los de inferencia (test_inference.py). Entrena a
    rutas temporales con cv_folds=3; NUNCA toca models/model.joblib real.
    """
    out = tmp_path_factory.mktemp("ml_artifacts")
    model_path = out / "model.joblib"
    report_path = out / "train_report.json"
    train(data_path=DEFAULT_DATA_PATH, model_path=model_path,
          report_path=report_path, cv_folds=3)
    return model_path, report_path
