"""Pruebas del servicio de inferencia MLService (Task 1.5).

Cubren: lectura del dict {"pipeline", "threshold"}, contrato de
classify() (tuple[str, float]), decisión de la etiqueta contra
self.threshold, mapeo de género, adaptador bool/int, rechazo de
entradas inválidas y from_app_state.

La fixture trained (conftest.py, scope session) provee el artefacto
temporal entrenado en la misma corrida; los tests de inferencia no
entrenan por su cuenta.
"""

from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from app.ml.inference import MLService

# Perfil clínico válido (dentro de los rangos del EDA) para los tests de MLService
VALID_PROFILE = {
    "age": 55, "ap_hi": 150, "ap_lo": 95, "cholesterol": 3, "gluc": 2,
    "smoke": True, "alco": False, "active": True, "height": 168,
    "weight": 88.0, "gender": "M",
}


@pytest.fixture(scope="module")
def ml_service(trained):
    """MLService cargado desde el artefacto temporal de la fixture trained."""
    model_path, _ = trained
    return MLService(model_path)


@pytest.mark.slow
def test_ml_service_loads_artifact(ml_service):
    """MLService lee el dict {pipeline, threshold} y expone ambos atributos."""
    assert hasattr(ml_service.pipeline, "predict_proba")
    assert isinstance(ml_service.threshold, float)
    assert 0.0 < ml_service.threshold < 1.0


@pytest.mark.slow
def test_classify_returns_valid_tuple(ml_service):
    """classify retorna (str en {riesgo, no_riesgo}, float en [0, 1])."""
    label, prob = ml_service.classify(VALID_PROFILE)

    assert isinstance(label, str)
    assert label in ("riesgo", "no_riesgo")
    assert type(prob) is float  # float de Python, no numpy.float32
    assert 0.0 <= prob <= 1.0


@pytest.mark.slow
def test_label_uses_configured_threshold(ml_service):
    """La etiqueta se decide contra self.threshold, NO contra 0.5.

    Prueba directa del plan B: se mueve el umbral a valores extremos y
    la etiqueta debe seguirlo (si comparara contra 0.5 fijo, no lo haría).
    Restaura el umbral original para no contaminar otros tests.
    """
    label, prob = ml_service.classify(VALID_PROFILE)
    # Consistencia con el umbral por defecto (xgboost ajustado ~0.3245)
    assert (prob >= ml_service.threshold) == (label == "riesgo")

    umbral_original = ml_service.threshold
    try:
        ml_service.threshold = 1.1
        label_alto, _ = ml_service.classify(VALID_PROFILE)
        assert label_alto == "no_riesgo"  # prob < 1.1 siempre

        ml_service.threshold = 0.0
        label_bajo, _ = ml_service.classify(VALID_PROFILE)
        assert label_bajo == "riesgo"  # prob >= 0.0 siempre
    finally:
        ml_service.threshold = umbral_original


@pytest.mark.slow
def test_classify_accepts_both_genders(ml_service):
    """El mapeo 'F'->1 / 'M'->2 clasifica ambos géneros sin errores."""
    label_f, prob_f = ml_service.classify({**VALID_PROFILE, "gender": "F"})
    label_m, prob_m = ml_service.classify(VALID_PROFILE)

    assert label_f in ("riesgo", "no_riesgo")
    assert label_m in ("riesgo", "no_riesgo")
    assert 0.0 <= prob_f <= 1.0
    assert 0.0 <= prob_m <= 1.0


@pytest.mark.slow
@pytest.mark.parametrize("invalid,campo", [
    ({"age": 500}, "age fuera de rango EDA"),
    ({"gender": "X"}, "gender invalido"),
    ({"ap_hi": 80, "ap_lo": 120}, "ap_hi <= ap_lo"),
])
def test_classify_rejects_invalid_input(ml_service, invalid, campo):
    """Entrada inválida levanta ValidationError de Pydantic, no crash del pipeline."""
    with pytest.raises(ValidationError):
        ml_service.classify({**VALID_PROFILE, **invalid})


@pytest.mark.slow
def test_classify_bool_and_int_equivalent(ml_service):
    """El adaptador normaliza los bools: mismo resultado con True/False que con 1/0.

    Fija el comportamiento del adaptador de representación: el dataset
    entrenó con 0/1 y clasificar con bool no debe depender de coerción
    implícita de sklearn/XGBoost (puede variar entre versiones).
    """
    base_int = dict(VALID_PROFILE)
    base_int.update({"smoke": 1, "alco": 0, "active": 1})

    _, prob_bool = ml_service.classify(VALID_PROFILE)
    _, prob_int = ml_service.classify(base_int)

    assert prob_bool == prob_int


@pytest.mark.slow
def test_from_app_state_returns_instance(ml_service):
    """from_app_state recupera la instancia guardada en app.state."""
    fake_request = SimpleNamespace(
        app=SimpleNamespace(state=SimpleNamespace(ml_service=ml_service))
    )
    assert MLService.from_app_state(fake_request) is ml_service
