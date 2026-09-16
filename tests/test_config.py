"""Tests for configuration settings."""

import pytest
from pathlib import Path

from app.config import Settings, settings


class TestSettings:
    """Test configuration loading."""

    def test_settings_load_from_env(self, monkeypatch):
        """Test settings load from environment variables."""
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
        """Test default settings when no env vars provided."""
        # This test uses the global settings instance
        assert settings.DATABASE_URL == "sqlite:///cv_risk.db"
        assert settings.MODEL_PATH == Path("models/model.joblib")
        assert settings.APP_TITLE == "Clasificación Riesgo Cardiovascular"
        assert settings.DEBUG is True

    def test_model_path_is_path_object(self):
        """Test MODEL_PATH is a Path object."""
        assert isinstance(settings.MODEL_PATH, Path)