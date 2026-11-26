"""Configuración de base de datos y gestión de sesiones."""

from sqlalchemy import create_engine, event
from sqlalchemy.orm import DeclarativeBase, sessionmaker, Session

from app.config import settings


class Base(DeclarativeBase):
    """Clase base para todos los modelos de SQLAlchemy."""
    pass


engine = create_engine(
    settings.DATABASE_URL,
    echo=settings.DEBUG,
    future=True,
)

# SQLite NO aplica integridad referencial por defecto: el PRAGMA
# foreign_keys=ON debe habilitarse POR CONEXIÓN (no es un ajuste global),
# por eso se asocia mediante un listener al evento "connect".
if settings.DATABASE_URL.startswith("sqlite"):
    @event.listens_for(engine, "connect")
    def set_sqlite_pragma(dbapi_connection, connection_record):
        """Habilita el cumplimiento de FK en cada conexión SQLite."""
        cursor = dbapi_connection.cursor()
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.close()


session_maker = sessionmaker(
    engine,
    class_=Session,
    expire_on_commit=False,
)


def get_db() -> Session:
    """Dependencia de FastAPI para obtener una sesión de BD."""
    session = session_maker()
    try:
        yield session
    finally:
        session.close()


def init_db() -> None:
    """Crea las tablas de la base de datos al arrancar."""
    Base.metadata.create_all(engine)


def close_db() -> None:
    """Cierra las conexiones de la base de datos al apagar."""
    engine.dispose()
