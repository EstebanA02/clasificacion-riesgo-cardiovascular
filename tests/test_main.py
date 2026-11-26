"""Pruebas de integración del ciclo de vida y manejo de errores (Task 1.6).

Cubren: arranque con/sin modelo, Depends from_app_state, handlers 422 y 404.
Usan TestClient como context manager para ejecutar el lifespan completo.
"""

import pytest
from fastapi import Depends, FastAPI
from fastapi.testclient import TestClient
from pydantic import BaseModel, Field

from app.main import create_app
from app.ml.inference import MLService


class DummyInput(BaseModel):
    """Entrada dummy para probar validación 422."""
    ap_hi: int = Field(ge=70, le=250)


@pytest.fixture
def app_with_model(trained, monkeypatch):
    """Crea una app FastAPI con MODEL_PATH apuntando al modelo temporal entrenado."""
    model_path, _ = trained
    monkeypatch.setattr("app.config.settings.MODEL_PATH", model_path)
    return create_app()


@pytest.fixture
def app_without_model(monkeypatch):
    """Crea una app FastAPI con MODEL_PATH inexistente."""
    monkeypatch.setattr("app.config.settings.MODEL_PATH", "models/no_existe.joblib")
    return create_app()


def test_startup_loads_model(app_with_model):
    """Con modelo presente, al arrancar app.state.ml_service existe y es MLService."""
    with TestClient(app_with_model) as client:
        # Disparar el lifespan haciendo una request
        response = client.get("/")
        assert response.status_code == 404  # ruta inexistente, pero app arrancó

        # Verificar que ml_service está en app.state
        assert hasattr(app_with_model.state, "ml_service")
        svc = app_with_model.state.ml_service
        assert isinstance(svc, MLService)
        assert hasattr(svc.pipeline, "predict_proba")
        assert isinstance(svc.threshold, float)
        assert 0.0 < svc.threshold < 1.0


def test_startup_fails_without_model(app_without_model):
    """Sin modelo, el arranque lanza RuntimeError con mensaje en español."""
    with pytest.raises(RuntimeError) as exc_info:
        with TestClient(app_without_model):
            pass  # el context manager ejecuta startup y falla

    assert "Modelo no encontrado en" in str(exc_info.value)
    assert "Ejecuta 'uv run python train.py' primero" in str(exc_info.value)


def test_depends_from_app_state(app_with_model):
    """Depends(MLService.from_app_state) recupera la misma instancia de app.state."""
    # Registrar ruta temporal de prueba
    @app_with_model.get("/test-di")
    def test_di_endpoint(service: MLService = Depends(MLService.from_app_state)):
        return {"threshold": service.threshold}

    with TestClient(app_with_model) as client:
        response = client.get("/test-di")
        assert response.status_code == 200
        data = response.json()
        assert "threshold" in data
        assert isinstance(data["threshold"], float)
        assert 0.0 < data["threshold"] < 1.0

        # Verificar que es la MISMA instancia que app.state.ml_service
        assert data["threshold"] == app_with_model.state.ml_service.threshold


def test_validation_error_returns_422(app_with_model):
    """POST con body inválido devuelve 422 con errors como lista y detail correcto."""
    # Registrar ruta temporal con validación Pydantic
    @app_with_model.post("/test-validation")
    def test_validation_endpoint(payload: DummyInput):
        return {"ok": True}

    with TestClient(app_with_model) as client:
        # ap_hi=300 está fuera del rango [70, 250]
        response = client.post("/test-validation", json={"ap_hi": 300})

        assert response.status_code == 422
        body = response.json()
        assert body["detail"] == "Error de validación"
        assert "errors" in body
        assert isinstance(body["errors"], list)
        assert len(body["errors"]) >= 1
        # Verificar estructura del error
        error = body["errors"][0]
        assert "loc" in error
        assert "msg" in error
        assert "type" in error


def test_not_found_returns_404(app_with_model):
    """GET a ruta inexistente devuelve 404 con detail en español."""
    with TestClient(app_with_model) as client:
        response = client.get("/ruta-que-no-existe")

        assert response.status_code == 404
        body = response.json()
        assert body == {"detail": "Recurso no encontrado"}