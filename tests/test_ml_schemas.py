"""Tests for ML Schemas - Pydantic v2 validators."""

import pytest
from datetime import datetime

from app.ml.schemas import ClassificationInput, ClassificationOutput


class TestClassificationInput:
    """Tests for ClassificationInput validators."""

    # --- Valid base data for reuse (within new EDA-based ranges) ---
    @pytest.fixture
    def valid_input(self):
        """Base valid input data within EDA-based ranges."""
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

    # --- Valid input passes ---
    def test_valid_input_passes(self, valid_input):
        """Valid input should pass all validators."""
        ClassificationInput(**valid_input)

    # --- age boundaries (29-64 años, INCLUSIVOS) ---
    @pytest.mark.parametrize("age,should_pass", [
        (28, False),  # < 29
        (29, True),   # límite inferior (ge=29)
        (50, True),   # típico
        (64, True),   # límite superior (le=64)
        (65, False),  # > 64
    ])
    def test_age_bounds(self, valid_input, age, should_pass):
        """Age must be between 29 and 64 years inclusive (ge=29, le=64)."""
        data = {**valid_input, "age": age}
        if should_pass:
            ClassificationInput(**data)
        else:
            with pytest.raises(ValueError):
                ClassificationInput(**data)

    # --- ap_hi boundaries (70-250 mmHg, INCLUSIVOS) ---
    @pytest.mark.parametrize("ap_hi,ap_lo,should_pass", [
        (69, 60, False),   # < 70
        (70, 60, True),    # límite inferior (ge=70)
        (71, 60, True),    # > 70
        (120, 80, True),   # típico
        (250, 200, True),  # límite superior (le=250)
        (251, 200, False), # > 250
    ])
    def test_ap_hi_bounds(self, valid_input, ap_hi, ap_lo, should_pass):
        """Systolic pressure must be >=70 and <=250 mmHg (ge=70, le=250).
        
        Adjusts ap_lo to be < ap_hi to avoid cross-field validation conflict.
        """
        data = {**valid_input, "ap_hi": ap_hi, "ap_lo": ap_lo}
        if should_pass:
            ClassificationInput(**data)
        else:
            with pytest.raises(ValueError):
                ClassificationInput(**data)

    # --- ap_lo boundaries (40-200 mmHg, INCLUSIVOS) ---
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
        """Diastolic pressure must be >=40 and <=200 mmHg (ge=40, le=200).
        
        Adjusts ap_hi to be > ap_lo to avoid cross-field validation conflict.
        """
        data = {**valid_input, "ap_lo": ap_lo}
        # For upper bound test, need ap_hi > ap_lo
        if ap_lo >= 150:
            data["ap_hi"] = ap_lo + 10
        if should_pass:
            ClassificationInput(**data)
        else:
            with pytest.raises(ValueError):
                ClassificationInput(**data)

    # --- cross-field: ap_hi > ap_lo ---
    @pytest.mark.parametrize("ap_hi,ap_lo,should_pass", [
        (120, 80, True),   # válido
        (120, 120, False), # igual no válido
        (80, 120, False),  # invertido no válido
        (100, 99, True),   # diferencia mínima válida
    ])
    def test_ap_hi_gt_ap_lo(self, valid_input, ap_hi, ap_lo, should_pass):
        """Systolic pressure must be strictly greater than diastolic."""
        data = {**valid_input, "ap_hi": ap_hi, "ap_lo": ap_lo}
        if should_pass:
            ClassificationInput(**data)
        else:
            with pytest.raises(ValueError, match="Presión sistólica debe ser > diastólica"):
                ClassificationInput(**data)

    # --- cholesterol enum (1, 2, 3) ---
    @pytest.mark.parametrize("chol,should_pass", [
        (0, False),  # fuera de rango
        (1, True),   # normal
        (2, True),   # above normal
        (3, True),   # well above normal
        (4, False),  # fuera de rango
    ])
    def test_cholesterol_enum(self, valid_input, chol, should_pass):
        """Cholesterol must be 1, 2, or 3."""
        data = {**valid_input, "cholesterol": chol}
        if should_pass:
            ClassificationInput(**data)
        else:
            with pytest.raises(ValueError):
                ClassificationInput(**data)

    # --- gluc enum (1, 2, 3) ---
    @pytest.mark.parametrize("gluc,should_pass", [
        (0, False),
        (1, True),
        (2, True),
        (3, True),
        (4, False),
    ])
    def test_gluc_enum(self, valid_input, gluc, should_pass):
        """Glucose must be 1, 2, or 3."""
        data = {**valid_input, "gluc": gluc}
        if should_pass:
            ClassificationInput(**data)
        else:
            with pytest.raises(ValueError):
                ClassificationInput(**data)

    # --- gender enum (M, F) ---
    @pytest.mark.parametrize("gender,should_pass", [
        ("M", True),
        ("F", True),
        ("X", False),
        ("m", False),  # case sensitive
        ("", False),
    ])
    def test_gender_enum(self, valid_input, gender, should_pass):
        """Gender must be 'M' or 'F' (case sensitive)."""
        data = {**valid_input, "gender": gender}
        if should_pass:
            ClassificationInput(**data)
        else:
            with pytest.raises(ValueError):
                ClassificationInput(**data)

    # --- height boundaries (120-220 cm, INCLUSIVOS) ---
    @pytest.mark.parametrize("height,should_pass", [
        (119, False),  # < 120
        (120, True),   # límite inferior (ge=120)
        (170, True),   # típico
        (220, True),   # límite superior (le=220)
        (221, False),  # > 220
    ])
    def test_height_bounds(self, valid_input, height, should_pass):
        """Height must be between 120-220 cm inclusive (ge=120, le=220)."""
        data = {**valid_input, "height": height}
        if should_pass:
            ClassificationInput(**data)
        else:
            with pytest.raises(ValueError):
                ClassificationInput(**data)

    # --- weight boundaries (30-200 kg, INCLUSIVOS) ---
    @pytest.mark.parametrize("weight,should_pass", [
        (29.9, False),  # < 30
        (30.0, True),   # límite inferior (ge=30)
        (70.0, True),   # típico
        (200.0, True),  # límite superior (le=200)
        (200.1, False), # > 200
    ])
    def test_weight_bounds(self, valid_input, weight, should_pass):
        """Weight must be between 30-200 kg inclusive (ge=30, le=200)."""
        data = {**valid_input, "weight": weight}
        if should_pass:
            ClassificationInput(**data)
        else:
            with pytest.raises(ValueError):
                ClassificationInput(**data)

    # --- boolean fields accept bool ---
    @pytest.mark.parametrize("field,value", [
        ("smoke", True), ("smoke", False),
        ("alco", True), ("alco", False),
        ("active", True), ("active", False),
    ])
    def test_boolean_fields_accept_bool(self, valid_input, field, value):
        """Boolean fields should accept True/False."""
        data = {**valid_input, field: value}
        ClassificationInput(**data)


class TestClassificationOutput:
    """Tests for ClassificationOutput schema."""

    def test_output_from_attributes(self):
        """ClassificationOutput should work with from_attributes (ORM)."""
        # Simulate ORM object
        class MockORM:
            classification_result = "riesgo"
            risk_score = 0.85
            classification_date = datetime(2024, 1, 15, 10, 30)

        output = ClassificationOutput.model_validate(MockORM())
        assert output.classification_result == "riesgo"
        assert output.risk_score == 0.85
        assert output.classification_date == datetime(2024, 1, 15, 10, 30)

    def test_output_literal_values(self):
        """Output accepts only 'riesgo' or 'no_riesgo'."""
        # Valid
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
        # Invalid
        with pytest.raises(ValueError):
            ClassificationOutput(
                classification_result="alto",
                risk_score=0.8,
                classification_date=datetime.now()
            )

    def test_risk_score_float_bounds(self):
        """risk_score must be float between 0.0 and 1.0 inclusive (ge=0.0, le=1.0)."""
        # Valid boundaries
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
        # Invalid boundaries
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