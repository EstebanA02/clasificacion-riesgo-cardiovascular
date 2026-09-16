"""Patient SQLAlchemy model."""

from sqlalchemy import String
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.database import Base


class Patient(Base):

    __tablename__ = "patients"

    # Primary key
    id: Mapped[int] = mapped_column(primary_key=True)

    #Fields
    name: Mapped[str] = mapped_column(String(100), nullable=False)
    gender: Mapped[str] = mapped_column(String(1), nullable=False)  # 'M' or 'F'

    # Relationship
    classifications: Mapped[list["Classification"]] = relationship(
         back_populates="patient", cascade="all, delete-orphan"
    )

    def __repr__(self) -> str:
        return f"<Patient(id={self.id})>"