"""FastAPI application factory and lifespan management."""

from contextlib import contextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.templating import Jinja2Templates

from app.config import settings
from app.database import init_db, close_db


# Templates
templates = Jinja2Templates(directory="src/app/templates")


@contextmanager
def lifespan(app: FastAPI):
    """Application lifespan: startup and shutdown (synchronous)."""
    # Startup
    init_db()

    # Verify model exists
    model_path = Path(settings.MODEL_PATH)
    if not model_path.exists():
        raise RuntimeError(
            f"Model not found at {model_path}. Run 'uv run python train.py' first."
        )

    # ML Service will be loaded here in Task 1.5
    # app.state.ml_service = MLService(model_path)

    yield

    # Shutdown
    close_db()


def create_app() -> FastAPI:
    """Create and configure FastAPI application."""
    app = FastAPI(
        title=settings.APP_TITLE,
        version=settings.APP_VERSION,
        lifespan=lifespan,
    )

    # Exception handlers
    @app.exception_handler(404)
    def not_found_handler(request: Request, exc: Exception):
        return JSONResponse(
            status_code=404,
            content={"detail": "Recurso no encontrado"},
        )

    @app.exception_handler(422)
    def validation_error_handler(request: Request, exc: Exception):
        return JSONResponse(
            status_code=422,
            content={"detail": "Error de validación", "errors": str(exc)},
        )

    # Routers will be added here in later tasks
    # app.include_router(patients_router)
    # app.include_router(classification_router)
    # app.include_router(history_router)
    # app.include_router(reportes_router)

    return app


app = create_app()