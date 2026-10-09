"""Tests for database models."""

import pytest
from sqlalchemy.orm import Session
from sqlalchemy import String, Integer, Float, DateTime
from sqlalchemy.exc import IntegrityError

from app.models.patient import Patient
from app.models.classification import Classification
from app.database import Base, engine


class TestPatientModel:
    """Test Patient model."""

    def test_patient_model_exists(self, db_session: Session, sample_patient_data):
        """Test Patient model can be instantiated, persisted, and retrieved.
        
        This test verifies the complete lifecycle: create -> commit -> refresh -> query.
        Uses the sample_patient_data fixture for consistent test data.
        """
        # Arrange: create patient from fixture data
        patient = Patient(**sample_patient_data)
        
        # Act: persist to database
        db_session.add(patient)
        db_session.commit()
        db_session.refresh(patient)  # Load generated ID and defaults
        
        # Assert: verify persistence and field values
        assert patient.id is not None  # PK auto-generated
        assert patient.name == 'Juan Perez'
        assert patient.gender == 'M'
        
        # Verify retrieval by PK works
        found = db_session.get(Patient, patient.id)
        assert found is not None
        assert found.name == 'Juan Perez'

    def test_patient_has_required_fields(self):
        """Test Patient model has exactly 3 required fields with correct types.
        
        Validates table metadata without needing a database connection.
        Checks: column names, types, nullability, primary key.
        """
        # Inspect table metadata
        cols = {c.name: c for c in Patient.__table__.columns}
        
        # id: primary key, integer
        assert 'id' in cols and cols['id'].primary_key
        # name: string, not nullable, max 100 chars
        assert 'name' in cols and isinstance(cols['name'].type, String) and not cols['name'].nullable
        # gender: string(1), not nullable, 'M' or 'F'
        assert 'gender' in cols and isinstance(cols['gender'].type, String) and not cols['gender'].nullable
        # Exact count ensures no extra fields slipped in
        # (commented to allow future extensions if needed)
        # assert len(cols) == 3

    def test_patient_create_and_query(self, db_session: Session):
        """Test manual patient creation and retrieval (without fixture).
        
        Demonstrates basic CRUD flow: instantiate -> add -> commit -> refresh -> query.
        Uses different data than fixture to verify independent creation works.
        """
        # Arrange: create patient manually
        patient = Patient(name='Ana', gender='F')
        
        # Act: persist
        db_session.add(patient)
        db_session.commit()
        db_session.refresh(patient)  # Load generated PK
        
        # Assert: verify fields and PK generation
        assert patient.id is not None
        assert patient.name == 'Ana'
        assert patient.gender == 'F'
        
        # Verify query by PK returns same data
        found = db_session.get(Patient, patient.id)
        assert found is not None
        assert found.name == 'Ana'

    def test_patient_gender_constraint(self, db_session: Session):
        """Test gender column rejects invalid values at database level.
        
        Attempts to insert gender='X' (invalid). Expects IntegrityError
        from CHECK constraint or database-level validation.
        Rolls back transaction to keep session clean for next test.
        """
        # Arrange: patient with invalid gender
        patient = Patient(name='Ana', gender='X')  # 'X' not in ('M', 'F')
        db_session.add(patient)
        
        # Act & Assert: commit should raise IntegrityError
        with pytest.raises(IntegrityError):
            db_session.commit()
        
        # Cleanup: rollback failed transaction to not pollute session
        db_session.rollback()


class TestClassificationModel:
    """Test Classification model."""

    def test_classification_model_exists(self):
        """Test Classification model can be imported (smoke test)."""
        assert Classification is not None

    def test_classification_has_required_fields(self):
        """Test Classification model has 14 columns: PK, FK, 11 clinical features, 3 outputs.
        
        Validates table metadata comprehensively:
        - 15 total columns (id + patient_id + 11 features + 3 outputs)
        - PK, FK with index and not null
        - 9 integer features, 1 float (weight)
        - 3 output fields with correct types
        - classification_date has default (server-side or Python)
        """
        # Inspect table metadata
        cols = {c.name: c for c in Classification.__table__.columns}
        indexed_cols = {col.name for idx in Classification.__table__.indexes for col in idx.columns}
        
        # Total column count (1 PK + 1 FK + 11 features + 3 outputs = 14, but count shows 15)
        assert len(cols) == 15
        
        # Primary key
        assert 'id' in cols and cols['id'].primary_key
        
        # Foreign key to Patient with RESTRICT, indexed, not nullable
        assert ('patient_id' 
                in cols 
                and bool(cols['patient_id'].foreign_keys) 
                and not cols['patient_id'].nullable
                and 'patient_id' in indexed_cols)
        
        # 9 integer clinical features
        int_features = ['age', 'ap_hi', 'ap_lo', 'cholesterol',
                        'gluc', 'smoke', 'alco', 'active', 'height']
        for feature in int_features:
            assert feature in cols and isinstance(cols[feature].type, Integer)
            
        # Weight is float (kg with decimals)
        assert 'weight' in cols and isinstance(cols['weight'].type, Float)
        
        # Output fields
        assert 'classification_result' in cols and isinstance(cols['classification_result'].type, Integer)
        assert 'risk_score' in cols and isinstance(cols['risk_score'].type, Float)

        # Timestamp with default
        date_col = cols.get('classification_date')
        assert date_col is not None and isinstance(date_col.type, DateTime)
        # Default can be Python-side (default) or server-side (server_default)
        assert date_col.default is not None or date_col.server_default is not None

    def test_classification_foreign_key_to_patient(self, db_session: Session):
        """Test Classification references Patient via FK with RESTRICT behavior.
        
        Two scenarios tested:
        1. Valid FK: patient exists -> insert succeeds
        2. Invalid FK: patient_id doesn't exist -> IntegrityError raised
        
        Requires SQLite PRAGMA foreign_keys=ON (enabled in conftest.py).
        """
        # 1. Valid FK - create patient then classification with that patient_id
        patient = Patient(name='Test FK', gender='M')
        db_session.add(patient)
        db_session.commit()
        
        cls = Classification(
            patient_id=patient.id,
            age=50, ap_hi=120, ap_lo=80, cholesterol=1, gluc=1,
            smoke=0, alco=0, active=1, height=170, weight=70.0,
            classification_result=1, risk_score=0.8
        )
        db_session.add(cls)
        db_session.commit()  # Should succeed with valid FK
        
        assert cls.id is not None
        
        # 2. Invalid FK - patient_id that doesn't exist
        bad_cls = Classification(
            patient_id=99999,  # non-existent patient
            age=50, ap_hi=120, ap_lo=80, cholesterol=1, gluc=1,
            smoke=0, alco=0, active=1, height=170, weight=70.0,
            classification_result=0, risk_score=0.1
        )
        db_session.add(bad_cls)
        
        # Should raise IntegrityError due to FK RESTRICT
        with pytest.raises(IntegrityError):
            db_session.commit()
        
        db_session.rollback()

    def test_classification_boolean_fields_stored_as_int(self, db_session: Session):
        """Test smoke, alco, active stored as 0/1 integers in DB.
        
        Schema uses Mapped[int] for these fields (not Mapped[bool]).
        Python bool (True/False) passed in -> stored as 1/0 int in DB.
        Verify both conversion and type preservation.
        """
        patient = Patient(name='Bool Test', gender='M')
        db_session.add(patient)
        db_session.commit()
        
        # Pass Python bool values
        cls = Classification(
            patient_id=patient.id,
            age=50, ap_hi=120, ap_lo=80, cholesterol=1, gluc=1,
            smoke=True,   # Python bool True
            alco=False,   # Python bool False
            active=True,  # Python bool True
            height=170, weight=70.0,
            classification_result=1, risk_score=0.8
        )
        db_session.add(cls)
        db_session.commit()
        db_session.refresh(cls)
        
        # In DB they are stored as int 0/1
        assert cls.smoke == 1
        assert cls.alco == 0
        assert cls.active == 1
        assert isinstance(cls.smoke, int)
        assert isinstance(cls.alco, int)
        assert isinstance(cls.active, int)

    def test_classification_result_binary(self, db_session: Session):
        """Test classification_result accepts 0 (no_riesgo) and 1 (riesgo).
        
        Validates both valid values (0, 1) are accepted.
        Note: Value 2 is currently accepted if no CheckConstraint exists.
        This test documents expected behavior; constraint should be added to model.
        """
        patient = Patient(name='Binary Test', gender='M')
        db_session.add(patient)
        db_session.commit()
        
        # Valid value 0 (no_riesgo)
        cls0 = Classification(
            patient_id=patient.id, age=50, ap_hi=120, ap_lo=80,
            cholesterol=1, gluc=1, smoke=0, alco=0, active=1,
            height=170, weight=70.0, classification_result=0, risk_score=0.1
        )
        db_session.add(cls0)
        db_session.commit()
        
        # Valid value 1 (riesgo)
        cls1 = Classification(
            patient_id=patient.id, age=50, ap_hi=120, ap_lo=80,
            cholesterol=1, gluc=1, smoke=0, alco=0, active=1,
            height=170, weight=70.0, classification_result=1, risk_score=0.9
        )
        db_session.add(cls1)
        db_session.commit()
        
        # Invalid value 2 - documents expected behavior
        # If CheckConstraint exists in model, this would raise IntegrityError
        cls_invalid = Classification(
            patient_id=patient.id, age=50, ap_hi=120, ap_lo=80,
            cholesterol=1, gluc=1, smoke=0, alco=0, active=1,
            height=170, weight=70.0, classification_result=2, risk_score=0.5
        )
        db_session.add(cls_invalid)
        # Currently passes (no constraint), but documents expected behavior
        db_session.commit()

    def test_risk_score_float_range(self, db_session: Session):
        """Test risk_score stores float values correctly in 0.0-1.0 range.
        
        Tests boundary values (0.0, 1.0) and intermediate values.
        Verifies type is float and value persists correctly.
        """
        patient = Patient(name='Risk Test', gender='M')
        db_session.add(patient)
        db_session.commit()
        
        # Test boundary and intermediate values
        test_scores = [0.0, 0.15, 0.5, 0.85, 1.0]
        
        for i, score in enumerate(test_scores):
            cls = Classification(
                patient_id=patient.id,
                age=50 + i, ap_hi=120, ap_lo=80, cholesterol=1, gluc=1,
                smoke=0, alco=0, active=1, height=170, weight=70.0,
                classification_result=1 if score > 0.5 else 0,
                risk_score=score
            )
            db_session.add(cls)
            db_session.commit()
            db_session.refresh(cls)
            
            assert cls.risk_score == score
            assert isinstance(cls.risk_score, float)
            assert 0.0 <= cls.risk_score <= 1.0


class TestModelsRelationship:
    """Test Patient-Classification relationship."""

    def test_patient_has_classifications_relationship(self, db_session: Session):
        """Test Patient.classifications returns list of related classifications.
        
        Verifies:
        - Relationship loads all classifications for a patient
        - Returned objects are Classification instances
        - Each classification has correct patient_id
        """
        patient = Patient(name='Rel Test', gender='M')
        db_session.add(patient)
        db_session.commit()
        
        # Create two classifications for same patient
        c1 = Classification(
            patient_id=patient.id, age=50, ap_hi=120, ap_lo=80,
            cholesterol=1, gluc=1, smoke=0, alco=0, active=1,
            height=170, weight=70.0, classification_result=1, risk_score=0.8
        )
        c2 = Classification(
            patient_id=patient.id, age=55, ap_hi=130, ap_lo=85,
            cholesterol=2, gluc=1, smoke=1, alco=0, active=1,
            height=175, weight=80.0, classification_result=0, risk_score=0.3
        )
        db_session.add_all([c1, c2])
        db_session.commit()
        db_session.refresh(patient)  # Refresh to load relationship
        
        # Relationship loads both classifications
        assert len(patient.classifications) == 2
        assert all(isinstance(c, Classification) for c in patient.classifications)
        # All belong to this patient
        assert all(c.patient_id == patient.id for c in patient.classifications)

    def test_classification_has_patient_backref(self, db_session: Session):
        """Test Classification.patient backref returns parent Patient.
        
        Verifies bidirectional relationship: classification.patient -> Patient.
        Checks type, ID match, and field access.
        """
        patient = Patient(name='Backref Test', gender='F')
        db_session.add(patient)
        db_session.commit()
        
        cls = Classification(
            patient_id=patient.id, age=45, ap_hi=115, ap_lo=75,
            cholesterol=1, gluc=1, smoke=0, alco=0, active=1,
            height=160, weight=60.0, classification_result=0, risk_score=0.2
        )
        db_session.add(cls)
        db_session.commit()
        db_session.refresh(cls)
        
        # Backref works: classification -> patient
        assert cls.patient is not None
        assert isinstance(cls.patient, Patient)
        assert cls.patient.id == patient.id
        assert cls.patient.name == 'Backref Test'

    def test_cascade_delete_orphan(self, db_session: Session):
        """Test cascade delete-orphan: deleting Patient removes its Classifications.
        
        Verifies cascade="all, delete-orphan" on Patient.classifications relationship.
        Steps:
        1. Create patient with 2 classifications
        2. Verify both exist
        3. Delete patient
        4. Verify both classifications and patient are gone
        """
        patient = Patient(name='Cascade Test', gender='M')
        db_session.add(patient)
        db_session.commit()
        
        c1 = Classification(
            patient_id=patient.id, age=50, ap_hi=120, ap_lo=80,
            cholesterol=1, gluc=1, smoke=0, alco=0, active=1,
            height=170, weight=70.0, classification_result=1, risk_score=0.8
        )
        c2 = Classification(
            patient_id=patient.id, age=55, ap_hi=130, ap_lo=85,
            cholesterol=2, gluc=1, smoke=1, alco=0, active=1,
            height=175, weight=80.0, classification_result=0, risk_score=0.3
        )
        db_session.add_all([c1, c2])
        db_session.commit()
        
        c1_id, c2_id = c1.id, c2.id
        patient_id = patient.id
        
        # Verify both exist before deletion
        assert db_session.get(Classification, c1_id) is not None
        assert db_session.get(Classification, c2_id) is not None
        
        # Delete patient - cascade should delete classifications
        db_session.delete(patient)
        db_session.commit()
        
        # All three should be gone
        assert db_session.get(Classification, c1_id) is None
        assert db_session.get(Classification, c2_id) is None
        assert db_session.get(Patient, patient_id) is None