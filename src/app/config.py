"""Configuración del sistema de clasificación de riesgo cardiovascular."""

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Configuración de la aplicación cargada desde variables de entorno."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )

    # Base de datos
    DATABASE_URL: str = "sqlite:///cv_risk.db"

    # Modelo de clasificación
    MODEL_PATH: Path = Path("models/model.joblib")

    # Aplicación
    APP_TITLE: str = "Clasificación Riesgo Cardiovascular"
    APP_VERSION: str = "0.1.0"
    DEBUG: bool = True


settings = Settings()
