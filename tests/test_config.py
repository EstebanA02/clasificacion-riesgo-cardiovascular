"""Pruebas de la configuración (settings)."""

import pytest
from pathlib import Path

from app.config import Settings, settings


class TestSettings:
    """Carga de la configuración desde variables de entorno."""

    def test_settings_load_from_env(self, monkeypatch):
        """Los settings se cargan desde variables de entorno."""
        monkeypatch.setenv("DATABASE_URL", "sqlite:///test.db")
        monkeypatch.setenv("MODEL_PATH", "models/test.joblib")
        monkeypatch.setenv("APP_TITLE", "Test App")
        monkeypatch.setenv("DEBUG", "false")

        test_settings = Settings()

        assert test_settings.DATABASE_URL == "sqlite:///test.db"
        assert test_settings.MODEL_PATH == Path("models/test.joblib")
        assert test_settings.APP_TITLE == "Test App"
        assert test_settings.DEBUG is False

    def test_settings_defaults(self):
        """Los valores por defecto se aplican sin variables de entorno."""
        # Este test usa la instancia global de settings
        assert settings.DATABASE_URL == "sqlite:///cv_risk.db"
        assert settings.MODEL_PATH == Path("models/model.joblib")
        assert settings.APP_TITLE == "Clasificación Riesgo Cardiovascular"
        assert settings.DEBUG is True

    def test_model_path_is_path_object(self):
        """MODEL_PATH es un objeto Path."""
        assert isinstance(settings.MODEL_PATH, Path)
