"""Factory de la aplicación FastAPI y gestión del ciclo de vida."""

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from fastapi.templating import Jinja2Templates
from fastapi.encoders import jsonable_encoder

from app.config import settings
from app.database import init_db, close_db
from app.ml.inference import MLService


templates = Jinja2Templates(directory=Path(__file__).parent / "templates")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Ciclo de vida de la app: arranque y apagado (asíncrono).

    Startup:
    - Inicializa la base de datos (create_all)
    - Verifica que el artefacto del Modelo de clasificación exista
    - Instancia MLService UNA sola vez y lo guarda en app.state.ml_service
      para acceso vía Depends(MLService.from_app_state) en los routers.

    Shutdown:
    - Cierra las conexiones de la base de datos.
    """
    init_db()

    model_path = Path(settings.MODEL_PATH)
    if not model_path.exists():
        raise RuntimeError(
            f"Modelo no encontrado en {model_path}. "
            f"Ejecuta 'uv run python train.py' primero."
        )

    app.state.ml_service = MLService(model_path)

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
        """Responde 404 con detalle en español."""
        return JSONResponse(
            status_code=404,
            content={"detail": "Recurso no encontrado"},
        )

    @app.exception_handler(RequestValidationError)
    def validation_error_handler(request: Request, exc: RequestValidationError):
        """Responde 422 con el detalle de los errores de validación."""
        return JSONResponse(
            status_code=422,
            content={
                "detail": "Error de validación",
                "errors": jsonable_encoder(exc.errors()),
            },
        )

    # Placeholders de routers (Fase 2): se descomentarán cuando existan
    # from app.api import patients, classification, history, reportes
    # app.include_router(patients.router, prefix="/pacientes", tags=["pacientes"])
    # app.include_router(classification.router, prefix="/clasificaciones", tags=["clasificaciones"])
    # app.include_router(history.router, prefix="/historial", tags=["historial"])
    # app.include_router(reportes.router, prefix="/reportes", tags=["reportes"])

    return app


app = create_app()