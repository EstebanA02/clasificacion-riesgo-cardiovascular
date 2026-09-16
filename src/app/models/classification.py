"""Classification SQLAlchemy model."""

import datetime
from sqlalchemy import ForeignKey, DateTime, func
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.database import Base


class Classification(Base):

    __tablename__ = "classifications"

    # Primary key
    id: Mapped[int] = mapped_column(primary_key=True)

    # Foreign key to Patient
    patient_id: Mapped[int] = mapped_column(
         ForeignKey("patients.id", ondelete="RESTRICT"), index=True, nullable=False
    )

    # Field
    age: Mapped[int]
    ap_hi: Mapped[int]          # Systolic blood pressure
    ap_lo: Mapped[int]          # Diastolic blood pressure
    cholesterol: Mapped[int]    # 1=normal, 2=above, 3=well above
    gluc: Mapped[int]           # 1=normal, 2=above, 3=well above
    smoke: Mapped[int]          # 0/1
    alco: Mapped[int]           # 0/1
    active: Mapped[int]         # 0/1
    height: Mapped[int]         # cm
    weight: Mapped[float]       # kg

    # Output fields
    classification_result: Mapped[int]  # 0=no_riesgo, 1=riesgo
    risk_score: Mapped[float]           # 0.0-1.0 continuous
    classification_date: Mapped[datetime.datetime] = mapped_column(
         DateTime, default=func.now(), nullable=False
    )

    # Relationship
    patient: Mapped["Patient"] = relationship(back_populates="classifications")

    def __repr__(self) -> str:
        return f"<Classification(id={self.id})>"