"""Servicio de inferencia del Modelo de clasificación.

MLService encapsula el artefacto serializado por train.py (Task 1.4):
un dict {"pipeline": <Pipeline>, "threshold": <float>}.

Se instancia UNA sola vez al arrancar la aplicación y se accede por
request vía from_app_state. Nunca se recarga el modelo por request.

Mapeo de género: la BD guarda 'M'/'F' (con CheckConstraint), pero el
dataset entrenó con 1=Mujer, 2=Hombre. La conversión vive en ESTE
módulo (y no en la API ni en los modelos de BD) porque aísla los
detalles internos de entrenamiento del modelo de ML de las capas de
persistencia y transporte (API), actuando como un adaptador único de
representación humana a representación matemática del dataset.
"""

from pathlib import Path

import joblib
import pandas as pd
from fastapi import Request

from app.ml.feature import FEATURE_ORDER
from app.ml.schemas import ClassificationInput


class MLService:
    """Encapsula el Modelo de clasificación y su umbral de decisión.

    El umbral NO es 0.5: es el ajustado por el plan B con probabilidades
    out-of-fold (recall como restricción, precisión como desempate).
    Comparar contra 0.5 desharía la calibración del entrenamiento.
    """

    def __init__(self, model_path: Path) -> None:
        """Carga el artefacto joblib y extrae pipeline y threshold.

        Contrato del archivo (Task 1.4): dict {"pipeline", "threshold"}.
        Convierte el threshold a float aquí (una vez) para no repetir
        la conversión en cada classify().
        """
        artifact = joblib.load(model_path)
        self.pipeline = artifact["pipeline"]
        self.threshold = float(artifact["threshold"])

    def classify(self, features: dict) -> tuple[str, float]:
        """Clasifica un perfil y devuelve (label, probabilidad de riesgo).

        Valida la entrada contra ClassificationInput (los errores de
        validación burbujean: la capa API los traducirá a 422 en Task
        1.6), mapea el género a la representación del dataset y calcula
        P(riesgo) con el pipeline. La etiqueta se decide comparando la
        probabilidad contra self.threshold, nunca contra 0.5.
        """
        input_model = ClassificationInput.model_validate(features)
        data = input_model.model_dump()

        # Adaptador de representación: 'F'->1, 'M'->2 (ver docstring de módulo)
        data["gender"] = 1 if data["gender"] == "F" else 2
        
        for field in ("smoke", "alco", "active"):
            data[field] = int(data[field])

        row_data = {col: [data[col]] for col in FEATURE_ORDER}
        df_features = pd.DataFrame(row_data)

        # Columna 1 = clase positiva (riesgo); columna 0 = no_riesgo
        probabilities = self.pipeline.predict_proba(df_features)
        risk_probability = float(probabilities[0][1])

        is_high_risk = risk_probability >= self.threshold
        label = "riesgo" if is_high_risk else "no_riesgo"

        return label, risk_probability

    @classmethod
    def from_app_state(cls, request: Request) -> "MLService":
        """Recupera la instancia única guardada en app.state por el lifespan.

        Se usará como dependencia de FastAPI (Task 1.6):
        service = Depends(MLService.from_app_state)
        """
        return request.app.state.ml_service
