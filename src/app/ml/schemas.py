"""Esquemas de entrada y salida del Modelo de clasificación (Pydantic v2).

ClassificationInput valida los datos que llegan desde la API; los límites
inferior/superior provienen del análisis de exploración de datos (EDA) y
son INCLUSIVOS. ClassificationOutput refleja lo que se guarda en la tabla
classifications.
"""

from datetime import datetime
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class ClassificationInput(BaseModel):
    """Entrada del Modelo de clasificación: 11 características clínicas."""

    age: Annotated[int, Field(ge=29, le=64, description="Edad en años")]
    ap_hi: Annotated[int, Field(ge=70, le=250, description="Presión sistólica (mmHg)")]
    ap_lo: Annotated[int, Field(ge=40, le=200, description="Presión diastólica (mmHg)")]
    cholesterol: Literal[1, 2, 3]
    gluc: Literal[1, 2, 3]
    smoke: bool
    alco: bool
    active: bool
    height: Annotated[int, Field(ge=120, le=220, description="Altura en cm")]
    weight: Annotated[float, Field(ge=30, le=200, description="Peso en kg")]
    gender: Literal["M", "F"]

    @model_validator(mode="after")
    def validate_bp(self) -> "ClassificationInput":
        """Validar que la presión sistólica (ap_hi) sea mayor que la diastólica (ap_lo)."""
        if self.ap_hi <= self.ap_lo:
            raise ValueError("Presión sistólica debe ser > diastólica")
        return self


class ClassificationOutput(BaseModel):
    """Resultado de la clasificación, tal como se guarda en la base de datos."""

    classification_result: Literal["riesgo", "no_riesgo"]
    risk_score: Annotated[float, Field(ge=0.0, le=1.0)]
    classification_date: datetime

    model_config = ConfigDict(from_attributes=True)
