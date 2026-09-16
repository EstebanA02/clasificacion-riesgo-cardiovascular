"""Configuration settings for the cardiovascular risk classification system."""

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )

    # Database
    DATABASE_URL: str = "sqlite:///cv_risk.db"

    # ML Model
    MODEL_PATH: Path = Path("models/model.joblib")

    # App
    APP_TITLE: str = "Clasificación Riesgo Cardiovascular"
    APP_VERSION: str = "0.1.0"
    DEBUG: bool = True


settings = Settings()