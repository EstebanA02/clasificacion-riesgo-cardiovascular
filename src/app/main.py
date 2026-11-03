"""Factory de la aplicación FastAPI y gestión del ciclo de vida"""

from contextlib import contextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.templating import Jinja2Templates

from app.config import settings
from app.database import init_db, close_db


templates = Jinja2Templates(directory="src/app/templates")


@contextmanager
def lifespan(app: FastAPI):
    """Ciclo de vida de la app: arranque y apagado (síncrono)."""
    init_db()

    # Verificar que el Modelo de clasificación exista antes de servir tráfico
    model_path = Path(settings.MODEL_PATH)
    if not model_path.exists():
        raise RuntimeError(
            f"Model not found at {model_path}. Run 'uv run python train.py' first."
        )

    # MLService se cargará aquí en la Task 1.5
    # app.state.ml_service = MLService(model_path)

    yield

    close_db()


def create_app() -> FastAPI:
    """Crea y configura la aplicación FastAPI."""
    app = FastAPI(
        title=settings.APP_TITLE,
        version=settings.APP_VERSION,
        lifespan=lifespan,
    )

    @app.exception_handler(404)
    def not_found_handler(request: Request, exc: Exception):
        """Responde 404"""
        return JSONResponse(
            status_code=404,
            content={"detail": "Recurso no encontrado"},
        )

    @app.exception_handler(422)
    def validation_error_handler(request: Request, exc: Exception):
        """Responde 422 con el detalle de los errores de validación."""
        return JSONResponse(
            status_code=422,
            content={"detail": "Error de validación", "errors": str(exc)},
        )

    # Los routers se agregarán en tareas posteriores
    # app.include_router(patients_router)
    # app.include_router(classification_router)
    # app.include_router(history_router)
    # app.include_router(reportes_router)

    return app


app = create_app()
