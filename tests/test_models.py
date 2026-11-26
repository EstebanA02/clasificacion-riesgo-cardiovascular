"""Pruebas de los modelos de base de datos (Patient y Classification)."""

import pytest
from sqlalchemy.orm import Session
from sqlalchemy import String, Integer, Float, DateTime
from sqlalchemy.exc import IntegrityError

from app.models.patient import Patient
from app.models.classification import Classification
from app.database import Base, engine


class TestPatientModel:
    """Modelo Patient (paciente)."""

    def test_patient_model_exists(self, db_session: Session, sample_patient_data):
        """Patient se instancia, persiste y se recupera.

        Verifica el ciclo completo: crear -> commit -> refresh -> consultar,
        usando el fixture sample_patient_data para datos consistentes.
        """
        # Arrange: crear el paciente a partir del fixture
        patient = Patient(**sample_patient_data)

        # Act: persistir en la base de datos
        db_session.add(patient)
        db_session.commit()
        db_session.refresh(patient)  # carga el id generado y los defaults

        # Assert: verificar persistencia y valores de los campos
        assert patient.id is not None  # PK autogenerada
        assert patient.name == 'Juan Perez'
        assert patient.gender == 'M'

        # Verificar que la lectura por PK funciona
        found = db_session.get(Patient, patient.id)
        assert found is not None
        assert found.name == 'Juan Perez'

    def test_patient_has_required_fields(self):
        """Patient tiene sus 3 campos con tipos correctos.

        Valida el metadata de la tabla sin necesidad de conexión a BD:
        nombres de columnas, tipos, nullability y primary key.
        """
        # Inspeccionar el metadata de la tabla
        cols = {c.name: c for c in Patient.__table__.columns}

        # id: primary key, entero
        assert 'id' in cols and cols['id'].primary_key
        # name: string, no nullable, máximo 100 caracteres
        assert 'name' in cols and isinstance(cols['name'].type, String) and not cols['name'].nullable
        # gender: string(1), no nullable, 'M' o 'F'
        assert 'gender' in cols and isinstance(cols['gender'].type, String) and not cols['gender'].nullable

    def test_patient_create_and_query(self, db_session: Session):
        """Creación y consulta manual de un Patient (sin fixture).

        Demuestra el flujo CRUD básico: instanciar -> add -> commit ->
        refresh -> consultar, con datos distintos a los del fixture.
        """
        # Arrange: crear paciente manualmente
        patient = Patient(name='Ana', gender='F')

        # Act: persistir
        db_session.add(patient)
        db_session.commit()
        db_session.refresh(patient)  # carga la PK generada

        # Assert: verificar campos y generación de PK
        assert patient.id is not None
        assert patient.name == 'Ana'
        assert patient.gender == 'F'

        # Verificar que la consulta por PK devuelve los mismos datos
        found = db_session.get(Patient, patient.id)
        assert found is not None
        assert found.name == 'Ana'

    def test_patient_gender_constraint(self, db_session: Session):
        """La columna gender rechaza valores inválidos a nivel BD.

        Intenta insertar gender='X' con el CHECK constraint
        ck_patient_gender; como resultado se genera un IntegrityError,
        se valida que se produce ese error y se hace rollback.
        """
        # Arrange: paciente con género inválido
        patient = Patient(name='Ana', gender='X')  # 'X' no está en ('M', 'F')
        db_session.add(patient)

        # Act & Assert: commit debe lanzar IntegrityError
        with pytest.raises(IntegrityError):
            db_session.commit()

        # Cleanup: rollback de la transacción fallida para no contaminar la sesión
        db_session.rollback()


class TestClassificationModel:
    """Modelo Classification (registro de clasificaciones)."""

    def test_classification_model_exists(self):
        """Classification se puede importar (prueba de humo)."""
        assert Classification is not None

    def test_classification_has_required_fields(self):
        """Classification tiene la estructura de columnas esperada.

        Valida el metadata de la tabla completo:
        - 15 columnas (1 PK + 1 FK + 10 rasgos clínicos + 2 salidas
          + classification_date; el género vive en Patient, no aquí)
        - PK, FK con índice y no nullable
        - 9 rasgos enteros, 1 flotante (weight)
        - campos de salida con tipos correctos
        - classification_date con default (Python o servidor)
        """
        # Inspeccionar el metadata de la tabla
        cols = {c.name: c for c in Classification.__table__.columns}
        indexed_cols = {col.name for idx in Classification.__table__.indexes for col in idx.columns}

        # Total de columnas (1 PK + 1 FK + 11 rasgos + 2 salidas + 1 fecha)
        assert len(cols) == 15

        # Primary key
        assert 'id' in cols and cols['id'].primary_key

        # FK hacia Patient con RESTRICT, indexada y no nullable
        assert ('patient_id'
                in cols
                and bool(cols['patient_id'].foreign_keys)
                and not cols['patient_id'].nullable
                and 'patient_id' in indexed_cols)

        # 9 rasgos clínicos enteros
        int_features = ['age', 'ap_hi', 'ap_lo', 'cholesterol',
                        'gluc', 'smoke', 'alco', 'active', 'height']
        for feature in int_features:
            assert feature in cols and isinstance(cols[feature].type, Integer)

        # weight es float (kg con decimales)
        assert 'weight' in cols and isinstance(cols['weight'].type, Float)

        # Campos de salida
        assert 'classification_result' in cols and isinstance(cols['classification_result'].type, Integer)
        assert 'risk_score' in cols and isinstance(cols['risk_score'].type, Float)

        # Marca temporal con default
        date_col = cols.get('classification_date')
        assert date_col is not None and isinstance(date_col.type, DateTime)
        # El default puede ser Python-side (default) o server-side (server_default)
        assert date_col.default is not None or date_col.server_default is not None

    def test_classification_foreign_key_to_patient(self, db_session: Session):
        """Classification referencia a Patient por FK con comportamiento RESTRICT.

        Dos escenarios:
        1. FK válida: el paciente existe -> la inserción se logra
        2. FK inválida: patient_id inexistente -> se lanza IntegrityError

        Requiere SQLite PRAGMA foreign_keys=ON (habilitado en conftest.py).
        """
        # 1. FK válida: crear paciente y luego clasificación con ese patient_id
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
        db_session.commit()  # debe pasar con FK válida

        assert cls.id is not None

        # 2. FK inválida: patient_id que no existe
        bad_cls = Classification(
            patient_id=99999,  # paciente inexistente
            age=50, ap_hi=120, ap_lo=80, cholesterol=1, gluc=1,
            smoke=0, alco=0, active=1, height=170, weight=70.0,
            classification_result=0, risk_score=0.1
        )
        db_session.add(bad_cls)

        # Debe lanzar IntegrityError por la FK RESTRICT
        with pytest.raises(IntegrityError):
            db_session.commit()

        db_session.rollback()

    def test_classification_boolean_fields_stored_as_int(self, db_session: Session):
        """smoke, alco, active se guardan como enteros 0/1 en BD.

        El esquema usa Mapped[int] para estos campos (no Mapped[bool]).
        Se pasan bool de Python (True/False) y quedan almacenados como
        1/0 en BD: se verifica la conversión y el tipo.
        """
        patient = Patient(name='Bool Test', gender='M')
        db_session.add(patient)
        db_session.commit()

        # Pasar valores bool de Python
        cls = Classification(
            patient_id=patient.id,
            age=50, ap_hi=120, ap_lo=80, cholesterol=1, gluc=1,
            smoke=True,   # bool True de Python
            alco=False,   # bool False de Python
            active=True,  # bool True de Python
            height=170, weight=70.0,
            classification_result=1, risk_score=0.8
        )
        db_session.add(cls)
        db_session.commit()
        db_session.refresh(cls)

        # En BD se guardan como int 0/1
        assert cls.smoke == 1
        assert cls.alco == 0
        assert cls.active == 1
        assert isinstance(cls.smoke, int)
        assert isinstance(cls.alco, int)
        assert isinstance(cls.active, int)

    def test_classification_result_binary(self, db_session: Session):
        """classification_result acepta 0 (no_riesgo) y 1 (riesgo).

        Valida que ambos valores válidos (0, 1) se aceptan. Nota: el
        valor 2 actualmente se acepta porque no existe un
        CheckConstraint sobre la columna; este test documenta el
        comportamiento esperado, y la restricción debería agregarse al
        modelo si el director del semillero lo aprueba.
        """
        patient = Patient(name='Binary Test', gender='M')
        db_session.add(patient)
        db_session.commit()

        # Valor válido 0 (no_riesgo)
        cls0 = Classification(
            patient_id=patient.id, age=50, ap_hi=120, ap_lo=80,
            cholesterol=1, gluc=1, smoke=0, alco=0, active=1,
            height=170, weight=70.0, classification_result=0, risk_score=0.1
        )
        db_session.add(cls0)
        db_session.commit()

        # Valor válido 1 (riesgo)
        cls1 = Classification(
            patient_id=patient.id, age=50, ap_hi=120, ap_lo=80,
            cholesterol=1, gluc=1, smoke=0, alco=0, active=1,
            height=170, weight=70.0, classification_result=1, risk_score=0.9
        )
        db_session.add(cls1)
        db_session.commit()

        # Valor inválido 2 — documenta el comportamiento actual esperado
        cls_invalid = Classification(
            patient_id=patient.id, age=50, ap_hi=120, ap_lo=80,
            cholesterol=1, gluc=1, smoke=0, alco=0, active=1,
            height=170, weight=70.0, classification_result=2, risk_score=0.5
        )
        db_session.add(cls_invalid)
        # Pasa sin restricción en la columna (comportamiento documentado)
        db_session.commit()

    def test_risk_score_float_range(self, db_session: Session):
        """risk_score guarda floats en el rango 0.0-1.0.

        Prueba valores de frontera (0.0, 1.0) e intermedios; verifica
        tipo float y persistencia correcta del valor.
        """
        patient = Patient(name='Risk Test', gender='M')
        db_session.add(patient)
        db_session.commit()

        # Valores de frontera e intermedios
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
    """Relación Patient-Classification (1:N)."""

    def test_patient_has_classifications_relationship(self, db_session: Session):
        """patient.classifications devuelve la lista de clasificaciones.

        Verifica:
        - la relación carga todas las clasificaciones del paciente
        - los objetos devueltos son instancias de Classification
        - cada clasificación tiene el patient_id correcto
        """
        patient = Patient(name='Rel Test', gender='M')
        db_session.add(patient)
        db_session.commit()

        # Crear dos clasificaciones para el mismo paciente
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
        db_session.refresh(patient)  # refresh para cargar la relación

        # La relación carga ambas clasificaciones
        assert len(patient.classifications) == 2
        assert all(isinstance(c, Classification) for c in patient.classifications)
        # Todas pertenecen a este paciente
        assert all(c.patient_id == patient.id for c in patient.classifications)

    def test_classification_has_patient_backref(self, db_session: Session):
        """classification.patient devuelve el Patient padre.

        Verifica la relación bidireccional:
        classification.patient -> Patient; comprueba tipo, coincidencia
        de id y acceso a campos.
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

        # El backref funciona: classification -> patient
        assert cls.patient is not None
        assert isinstance(cls.patient, Patient)
        assert cls.patient.id == patient.id
        assert cls.patient.name == 'Backref Test'

    def test_cascade_delete_orphan(self, db_session: Session):
        """Cascade delete-orphan: al eliminar un Patient se borran sus Classification.

        Verifica cascade="all, delete-orphan" en la relación
        Patient.classifications. Pasos:
        1. Crear un paciente con 2 clasificaciones
        2. Verificar que ambas existen
        3. Eliminar el paciente
        4. Verificar que las 2 clasificaciones y el paciente ya no están
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

        # Verificar que ambas existen antes de la eliminación
        assert db_session.get(Classification, c1_id) is not None
        assert db_session.get(Classification, c2_id) is not None

        # Eliminar el paciente: el cascade debe borrar las clasificaciones
        db_session.delete(patient)
        db_session.commit()

        # Las tres filas ya no deben existir
        assert db_session.get(Classification, c1_id) is None
        assert db_session.get(Classification, c2_id) is None
        assert db_session.get(Patient, patient_id) is None
