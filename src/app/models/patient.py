"""Modelo SQLAlchemy de Patient (paciente)."""

from sqlalchemy import String, CheckConstraint
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.database import Base


class Patient(Base):
    """Paciente registrado en el sistema.

    Guarda solo identidad mínima (nombre y género); lo clínico vive en
    Classification (relación 1:N).
    """

    __tablename__ = "patients"

    id: Mapped[int] = mapped_column(primary_key=True)

    name: Mapped[str] = mapped_column(String(100), nullable=False)
    gender: Mapped[str] = mapped_column(String(1), nullable=False)  # 'M' o 'F'

    # Restricción a nivel BD: solo 'M' o 'F'
    __table_args__ = (
        CheckConstraint("gender IN ('M', 'F')", name="ck_patient_gender"),
    )

    classifications: Mapped[list["Classification"]] = relationship(
         back_populates="patient", cascade="all, delete-orphan"
    )

    def __repr__(self) -> str:
        return f"<Patient(id={self.id}, name='{self.name}', gender='{self.gender}')>"
