"""Pruebas de los esquemas ML — validadores de Pydantic v2."""

import pytest
from datetime import datetime

from app.ml.schemas import ClassificationInput, ClassificationOutput


class TestClassificationInput:
    """Validadores de ClassificationInput."""

    # --- Datos base válidos para reutilizar (dentro de los rangos del EDA) ---
    @pytest.fixture
    def valid_input(self):
        """Datos base válidos dentro de los rangos derivados del EDA."""
        return {
            "age": 50,           # 29-64
            "ap_hi": 120,        # 70-250
            "ap_lo": 80,         # 40-200
            "cholesterol": 1,
            "gluc": 1,
            "smoke": False,
            "alco": False,
            "active": True,
            "height": 170,       # 120-220
            "weight": 70.0,      # 30-200
            "gender": "M",
        }

    # --- Entrada válida pasa ---
    def test_valid_input_passes(self, valid_input):
        """Una entrada válida pasa todos los validadores."""
        ClassificationInput(**valid_input)

    # --- límites de age (29-64 años, INCLUSIVOS) ---
    @pytest.mark.parametrize("age,should_pass", [
        (28, False),  # < 29
        (29, True),   # límite inferior (ge=29)
        (50, True),   # típico
        (64, True),   # límite superior (le=64)
        (65, False),  # > 64
    ])
    def test_age_bounds(self, valid_input, age, should_pass):
        """age debe estar entre 29 y 64 años, inclusive (ge=29, le=64)."""
        data = {**valid_input, "age": age}
        if should_pass:
            ClassificationInput(**data)
        else:
            with pytest.raises(ValueError):
                ClassificationInput(**data)

    # --- límites de ap_hi (70-250 mmHg, INCLUSIVOS) ---
    @pytest.mark.parametrize("ap_hi,ap_lo,should_pass", [
        (69, 60, False),   # < 70
        (70, 60, True),    # límite inferior (ge=70)
        (71, 60, True),    # > 70
        (120, 80, True),   # típico
        (250, 200, True),  # límite superior (le=250)
        (251, 200, False), # > 250
    ])
    def test_ap_hi_bounds(self, valid_input, ap_hi, ap_lo, should_pass):
        """La presión sistólica debe ser >=70 y <=250 mmHg (ge=70, le=250).

        Ajusta ap_lo para que quede por debajo de ap_hi y no interfiera
        la validación entre campos.
        """
        data = {**valid_input, "ap_hi": ap_hi, "ap_lo": ap_lo}
        if should_pass:
            ClassificationInput(**data)
        else:
            with pytest.raises(ValueError):
                ClassificationInput(**data)

    # --- límites de ap_lo (40-200 mmHg, INCLUSIVOS) ---
    @pytest.mark.parametrize("ap_lo,should_pass", [
        (39, False),   # < 40
        (40, True),    # límite inferior (ge=40)
        (41, True),    # > 40
        (80, True),    # típico
        (199, True),   # < 200
        (200, True),   # límite superior (le=200)
        (201, False),  # > 200
    ])
    def test_ap_lo_bounds(self, valid_input, ap_lo, should_pass):
        """La presión diastólica debe ser >=40 y <=200 mmHg (ge=40, le=200).

        Ajusta ap_hi para que quede por encima de ap_lo y no interfiera
        la validación entre campos.
        """
        data = {**valid_input, "ap_lo": ap_lo}
        # Para los límites superiores se necesita ap_hi > ap_lo
        if ap_lo >= 150:
            data["ap_hi"] = ap_lo + 10
        if should_pass:
            ClassificationInput(**data)
        else:
            with pytest.raises(ValueError):
                ClassificationInput(**data)

    # --- entre campos: ap_hi > ap_lo ---
    @pytest.mark.parametrize("ap_hi,ap_lo,should_pass", [
        (120, 80, True),   # válido
        (120, 120, False), # igual no válido
        (80, 120, False),  # invertido no válido
        (100, 99, True),   # diferencia mínima válida
    ])
    def test_ap_hi_gt_ap_lo(self, valid_input, ap_hi, ap_lo, should_pass):
        """La presión sistólica debe ser estrictamente mayor que la diastólica."""
        data = {**valid_input, "ap_hi": ap_hi, "ap_lo": ap_lo}
        if should_pass:
            ClassificationInput(**data)
        else:
            with pytest.raises(ValueError, match="Presión sistólica debe ser > diastólica"):
                ClassificationInput(**data)

    # --- enum de cholesterol (1, 2, 3) ---
    @pytest.mark.parametrize("chol,should_pass", [
        (0, False),  # fuera de rango
        (1, True),   # normal
        (2, True),   # elevado
        (3, True),   # muy elevado
        (4, False),  # fuera de rango
    ])
    def test_cholesterol_enum(self, valid_input, chol, should_pass):
        """cholesterol debe ser 1, 2 o 3."""
        data = {**valid_input, "cholesterol": chol}
        if should_pass:
            ClassificationInput(**data)
        else:
            with pytest.raises(ValueError):
                ClassificationInput(**data)

    # --- enum de gluc (1, 2, 3) ---
    @pytest.mark.parametrize("gluc,should_pass", [
        (0, False),
        (1, True),
        (2, True),
        (3, True),
        (4, False),
    ])
    def test_gluc_enum(self, valid_input, gluc, should_pass):
        """gluc debe ser 1, 2 o 3."""
        data = {**valid_input, "gluc": gluc}
        if should_pass:
            ClassificationInput(**data)
        else:
            with pytest.raises(ValueError):
                ClassificationInput(**data)

    # --- enum de gender (M, F) ---
    @pytest.mark.parametrize("gender,should_pass", [
        ("M", True),
        ("F", True),
        ("X", False),
        ("m", False),  # sensible a mayúsculas
        ("", False),
    ])
    def test_gender_enum(self, valid_input, gender, should_pass):
        """gender debe ser 'M' o 'F' (sensible a mayúsculas)."""
        data = {**valid_input, "gender": gender}
        if should_pass:
            ClassificationInput(**data)
        else:
            with pytest.raises(ValueError):
                ClassificationInput(**data)

    # --- límites de height (120-220 cm, INCLUSIVOS) ---
    @pytest.mark.parametrize("height,should_pass", [
        (119, False),  # < 120
        (120, True),   # límite inferior (ge=120)
        (170, True),   # típico
        (220, True),   # límite superior (le=220)
        (221, False),  # > 220
    ])
    def test_height_bounds(self, valid_input, height, should_pass):
        """height debe estar entre 120 y 220 cm, inclusive (ge=120, le=220)."""
        data = {**valid_input, "height": height}
        if should_pass:
            ClassificationInput(**data)
        else:
            with pytest.raises(ValueError):
                ClassificationInput(**data)

    # --- límites de weight (30-200 kg, INCLUSIVOS) ---
    @pytest.mark.parametrize("weight,should_pass", [
        (29.9, False),  # < 30
        (30.0, True),   # límite inferior (ge=30)
        (70.0, True),   # típico
        (200.0, True),  # límite superior (le=200)
        (200.1, False), # > 200
    ])
    def test_weight_bounds(self, valid_input, weight, should_pass):
        """weight debe estar entre 30 y 200 kg, inclusive (ge=30, le=200)."""
        data = {**valid_input, "weight": weight}
        if should_pass:
            ClassificationInput(**data)
        else:
            with pytest.raises(ValueError):
                ClassificationInput(**data)

    # --- campos booleanos aceptan bool ---
    @pytest.mark.parametrize("field,value", [
        ("smoke", True), ("smoke", False),
        ("alco", True), ("alco", False),
        ("active", True), ("active", False),
    ])
    def test_boolean_fields_accept_bool(self, valid_input, field, value):
        """Los campos booleanos aceptan True/False."""
        data = {**valid_input, field: value}
        ClassificationInput(**data)


class TestClassificationOutput:
    """Esquema ClassificationOutput."""

    def test_output_from_attributes(self):
        """ClassificationOutput funciona con from_attributes (ORM)."""
        # Simular un objeto ORM
        class MockORM:
            classification_result = "riesgo"
            risk_score = 0.85
            classification_date = datetime(2024, 1, 15, 10, 30)

        output = ClassificationOutput.model_validate(MockORM())
        assert output.classification_result == "riesgo"
        assert output.risk_score == 0.85
        assert output.classification_date == datetime(2024, 1, 15, 10, 30)

    def test_output_literal_values(self):
        """La salida acepta solo 'riesgo' o 'no_riesgo'."""
        # Válido
        ClassificationOutput(
            classification_result="riesgo",
            risk_score=0.8,
            classification_date=datetime.now()
        )
        ClassificationOutput(
            classification_result="no_riesgo",
            risk_score=0.2,
            classification_date=datetime.now()
        )
        # Inválido
        with pytest.raises(ValueError):
            ClassificationOutput(
                classification_result="alto",
                risk_score=0.8,
                classification_date=datetime.now()
            )

    def test_risk_score_float_bounds(self):
        """risk_score debe ser float entre 0.0 y 1.0, inclusive (ge=0.0, le=1.0)."""
        # Fronteras válidas
        ClassificationOutput(
            classification_result="riesgo",
            risk_score=0.0,
            classification_date=datetime.now()
        )
        ClassificationOutput(
            classification_result="riesgo",
            risk_score=1.0,
            classification_date=datetime.now()
        )
        ClassificationOutput(
            classification_result="riesgo",
            risk_score=0.5,
            classification_date=datetime.now()
        )
        # Fronteras inválidas
        with pytest.raises(ValueError):
            ClassificationOutput(
                classification_result="riesgo",
                risk_score=-0.1,
                classification_date=datetime.now()
            )
        with pytest.raises(ValueError):
            ClassificationOutput(
                classification_result="riesgo",
                risk_score=1.1,
                classification_date=datetime.now()
            )
