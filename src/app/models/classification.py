"""Modelo SQLAlchemy de Classification (registro de clasificaciones)."""

import datetime
from sqlalchemy import ForeignKey, DateTime, func
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.database import Base


class Classification(Base):
    """Registro de una clasificación: los rasgos clínicos y el resultado.

    Cada fila guarda los 11 rasgos usados por el Modelo de clasificación
    junto con classification_result y risk_score (0.0-1.0).
    """

    __tablename__ = "classifications"

    id: Mapped[int] = mapped_column(primary_key=True)

    # FK RESTRICT: no se borra un Patient con clasificaciones asociadas
    patient_id: Mapped[int] = mapped_column(
         ForeignKey("patients.id", ondelete="RESTRICT"), index=True, nullable=False
    )

    age: Mapped[int]
    ap_hi: Mapped[int]          # presión sistólica (mmHg)
    ap_lo: Mapped[int]          # presión diastólica (mmHg)
    cholesterol: Mapped[int]    # 1=normal, 2=elevado, 3=muy elevado
    gluc: Mapped[int]           # 1=normal, 2=elevado, 3=muy elevado
    smoke: Mapped[int]          # 0/1
    alco: Mapped[int]           # 0/1
    active: Mapped[int]         # 0/1
    height: Mapped[int]         # cm
    weight: Mapped[float]       # kg

    classification_result: Mapped[int]  # 0=no_riesgo, 1=riesgo
    risk_score: Mapped[float]           # 0.0-1.0 continuo
    classification_date: Mapped[datetime.datetime] = mapped_column(
         DateTime, default=func.now(), nullable=False
    )

    patient: Mapped["Patient"] = relationship(back_populates="classifications")

    def __repr__(self) -> str:
        return f"<Classification(id={self.id})>"
