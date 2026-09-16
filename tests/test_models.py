"""Tests for database models."""

import pytest
from sqlalchemy.orm import Session

from app.models.patient import Patient
from app.models.classification import Classification
from app.database import Base, engine


class TestPatientModel:
    """Test Patient model."""

    def test_patient_model_exists(self):
        """Test Patient model can be imported."""
        assert Patient is not None

    def test_patient_has_required_fields(self):
        """Test Patient model has required fields: id, name, gender."""
        # TODO: Implement after model fields are defined
        pass

    def test_patient_create_and_query(self, db_session: Session):
        """Test creating and querying a patient."""
        # TODO: Implement after model fields are defined
        pass

    def test_patient_gender_constraint(self, db_session: Session):
        """Test gender only accepts 'M' or 'F'."""
        # TODO: Implement after model fields are defined
        pass


class TestClassificationModel:
    """Test Classification model."""

    def test_classification_model_exists(self):
        """Test Classification model can be imported."""
        assert Classification is not None

    def test_classification_has_required_fields(self):
        """Test Classification model has 11 clinical features + 3 outputs."""
        # TODO: Implement after model fields are defined
        pass

    def test_classification_foreign_key_to_patient(self, db_session: Session):
        """Test Classification references Patient via FK."""
        # TODO: Implement after model fields are defined
        pass

    def test_classification_boolean_fields_stored_as_int(self, db_session: Session):
        """Test smoke, alco, active stored as 0/1 int."""
        # TODO: Implement after model fields are defined
        pass

    def test_classification_result_binary(self, db_session: Session):
        """Test classification_result is 0 or 1."""
        # TODO: Implement after model fields are defined
        pass

    def test_risk_score_float_range(self, db_session: Session):
        """Test risk_score is float between 0.0 and 1.0."""
        # TODO: Implement after model fields are defined
        pass


class TestModelsRelationship:
    """Test Patient-Classification relationship."""

    def test_patient_has_classifications_relationship(self):
        """Test Patient has classifications relationship."""
        # TODO: Implement after relationship is defined
        pass

    def test_classification_has_patient_backref(self):
        """Test Classification has patient backref."""
        # TODO: Implement after relationship is defined
        pass

    def test_cascade_delete_orphan(self, db_session: Session):
        """Test deleting patient removes classifications (if cascade)."""
        # TODO: Implement after relationship is defined
        pass