# Design: core-modules-v1 — Cardiovascular Risk Classification (Academic Scope)

## Technical Approach

Layered FastAPI application with strict separation: **API routers** (thin) → **Domain services** (business logic) → **SQLAlchemy models** → **Pydantic v2 schemas**. ML model loaded once at startup via FastAPI lifespan into `app.state.ml_service` and accessed through `MLService` singleton. Jinja2 + Bootstrap 5 (CDN) templates with vanilla JavaScript only — no HTMX, no Alpine.js. SQLite with `Base.metadata.create_all()` on startup (no Alembic). Training via offline `train.py` script.

**Testing Strategy: Test-After per Phase** — tests are written after each implementation phase completes, not upfront (TDD not required for academic scope). Each phase has explicit test commands in Implementation Notes below.

---

## Architecture Decisions

| Decision | Choice | Alternatives Considered | Rationale |
|----------|--------|------------------------|-----------|
| **ML Model Serialization** | `sklearn Pipeline` (ColumnTransformer + estimator) via `joblib` | ONNX, pickle, custom JSON | Native sklearn pipeline preserves preprocessing; `joblib` handles numpy arrays efficiently |
| **Model Loading** | FastAPI lifespan → `app.state.ml_service` + `Depends` | Global variable, module-level singleton | Lifespan ensures startup failure if missing; `Depends` enables testing with overrides; thread-safe |
| **Model Selection** | Recall optimization on validation set (clinical priority) | Accuracy, F1, AUC | Recall prioritizes detecting true cardiovascular risk cases |
| **Database Init** | `Base.metadata.create_all(engine)` in `main.py` startup | Alembic migrations | Academic scope — no migration complexity; greenfield project |
| **Pagination** | Simple offset/limit (`page`, `page_size`) | Keyset/cursor pagination | Simplicity for academic scope; dataset small enough |
| **Frontend** | Jinja2 + Bootstrap 5 CDN + vanilla JS | HTMX, Alpine.js, React | Constraint: vanilla JS only; no build step |
| **Validation** | Pydantic v2 on API + vanilla JS on forms | Server-only, WTForms | Dual validation: immediate UX feedback + server enforcement |
| **Terminology** | `classification_result` ("riesgo"/"no_riesgo"), `risk_score` (0.0-1.0) | `prediction`, `probability`, `risk_level` | Locked per academic requirements — never use "predicción"/"predecir" |

---

## Data Flow

### Request Flow: Patient Creation

```text
GET /pacientes/nuevo
      │
      ▼
templates/patients/form.html (Bootstrap 5 + vanilla JS)
      │
      ▼  User submits form
vanilla JS fetch POST /pacientes (JSON)
      │
      ▼
PatientCreate schema validation (name, gender)
      │
      ├── 422 ──→ JSON error response
      │
      ▼ 201
PatientService.create() ──→ DB INSERT (Patient)
      │
      ▼
Redirect to /pacientes/{id} (detail view with classification button)
```

### Request Flow: Manual Classification

```text
User clicks "Clasificar" button on patient detail
      │
      ▼
Modal/form with 11 clinical fields (Bootstrap 5 + vanilla JS)
      │
      ▼  User submits
vanilla JS fetch POST /clasificar {patient_id, 11 features}
      │
      ▼
ClassificationInput schema validation (11 features + cross-field)
      │
      ├── 422 ──→ JSON error response
      │
      ▼ 201
MLService.classify(features) ──→ (classification_result, risk_score)
      │
      ▼
ClassificationService.create_classification() ──→ DB INSERT (Classification)
      │
      ▼
JSON response with classification_result, risk_score, classification_date
      │
      ▼
UI updates: shows result badge + risk score
```

### Request Flow: History Pagination

```text
GET /pacientes/{id}/clasificaciones?page=1&page_size=10
      │
      ▼
ClassificationService.list_paginated(patient_id, page, page_size)
      │
      ├── ORDER BY classification_date DESC
      ├── OFFSET (page-1)*page_size
      └── LIMIT page_size
      │
      ▼
JSON: { items[], total, page, page_size, total_pages }
      │
      ▼
templates/history/list.html (Bootstrap table + vanilla JS pagination)
```

---

## File Tree (Complete `src/app/` Structure)

```
src/app/
├── __init__.py
├── main.py                 # FastAPI app, lifespan, create_all, exception handlers
├── config.py               # Settings (pydantic-settings): DATABASE_URL, MODEL_PATH
├── database.py             # engine, sessionmaker, Base
├── models/
│   ├── __init__.py
│   ├── patient.py          # Patient (id, name, gender)
│   └── classification.py   # Classification (id, patient_id, 11 features, result, risk_score, date)
├── schemas/
│   ├── __init__.py
│   ├── patient.py          # PatientCreate, PatientRead
│   └── classification.py   # ClassificationInput, ClassificationOutput
├── api/
│   ├── __init__.py
│   ├── patients.py         # POST/GET /pacientes, GET /pacientes/nuevo (form page)
│   ├── classification.py   # POST /clasificar, GET /pacientes/{id}/clasificar (form page)
│   ├── history.py          # GET /pacientes/{id}/clasificaciones, GET /pacientes/{id}/historial
│   └── reportes.py         # GET /pacientes/{id}/reporte
├── services/
│   ├── __init__.py
│   ├── patient_service.py  # create(), get()
│   ├── classification_service.py  # create_classification(), list_paginated()
│   └── history_service.py  # get_by_patient() for reports
├── ml/
│   ├── __init__.py
│   ├── inference.py        # MLService: lifespan load, classify(features) -> (label, risk_score)
│   └── schemas.py          # ClassificationInput (11 features), ClassificationOutput
├── templates/
│   ├── base.html           # Layout with Bootstrap 5 CDN
│   ├── patients/
│   │   ├── form.html       # Patient creation form (vanilla JS validation)
│   │   └── detail.html     # Patient detail + classification trigger button
│   ├── history/
│   │   └── list.html       # Classification history table + vanilla JS pagination
│   └── reportes/
│       └── paciente.html   # Patient report: summary cards + Bootstrap 5 table
├── static/
│   ├── css/style.css       # Minimal custom styles
│   └── js/main.js          # Vanilla JS: form validation, fetch calls, pagination
train.py                    # ROOT: offline training (LR, RF, XGB, recall selection, joblib)
models/
  └── model.joblib          # gitignored, generated by train.py
data/
  └── cardio.csv            # training dataset (versionado, 2.94MB)
tests/
├── conftest.py             # Fixtures: client, db_session, ml_model, sample_patient
├── test_ml.py
├── test_patients.py
├── test_classification.py
├── test_history.py
└── test_reportes.py
```

---

## Key Classes & Functions

### ML Pipeline (`src/app/ml/inference.py`)

```python
class MLService:
    """Singleton inference service loaded at startup."""
    def __init__(self, model_path: Path):
        self.pipeline: Pipeline = joblib.load(model_path)

    def classify(self, features: dict) -> tuple[str, float]:
        # 1. Validate input against ClassificationInput schema
        # 2. Convert to DataFrame with correct column order
        # 3. probability = pipeline.predict_proba(df)[:, 1]
        # 4. label = "riesgo" if probability >= threshold else "no_riesgo"
        #    (threshold embedded in pipeline or config)
        # 5. Return (label, float(probability))

    @classmethod
    def from_app_state(cls, request: Request) -> "MLService":
        return request.app.state.ml_service
```

### Training Script (`train.py`)

```python
def train():
    # 1. Load data/cardio.csv
    # 2. Define 3 sklearn Pipelines:
    #    - LogisticRegression + StandardScaler
    #    - RandomForestClassifier (no scaling needed)
    #    - XGBClassifier + StandardScaler
    # 3. Train/validation split (stratified)
    # 4. Fit each pipeline, compute recall on validation
    # 5. Select best by recall
    # 6. Save best pipeline to models/model.joblib via joblib.dump()
    # 7. Print metrics for all 3 models
```

### Patient Schema (`src/app/schemas/patient.py`)

```python
class PatientCreate(BaseModel):
    name: Annotated[str, Field(min_length=1, max_length=100)]
    gender: Literal["M", "F"]

class PatientRead(BaseModel):
    id: int
    name: str
    gender: Literal["M", "F"]
    model_config = ConfigDict(from_attributes=True)
```

### Classification Schema (`src/app/schemas/classification.py`)

```python
class ClassificationInput(BaseModel):
    age: Annotated[int, Field(ge=18, le=100)]
    ap_hi: Annotated[int, Field(gt=0, le=300)]
    ap_lo: Annotated[int, Field(gt=0, le=200)]
    cholesterol: Literal[1, 2, 3]
    gluc: Literal[1, 2, 3]
    smoke: bool
    alco: bool
    active: bool
    height: Annotated[float, Field(gt=0, le=250)]
    weight: Annotated[float, Field(gt=0, le=300)]
    gender: Literal["M", "F"]

    @model_validator(mode="after")
    def validate_bp(self) -> "ClassificationInput":
        if self.ap_hi <= self.ap_lo:
            raise ValueError("Presión sistólica debe ser > diastólica")
        return self

class ClassificationOutput(BaseModel):
    id: int
    patient_id: int
    classification_result: Literal["riesgo", "no_riesgo"]
    risk_score: float
    classification_date: datetime
    model_config = ConfigDict(from_attributes=True)
```

### Classification Model (`src/app/models/classification.py`)

```python
class Classification(Base):
    __tablename__ = "classifications"

    id: Mapped[int] = mapped_column(primary_key=True)
    patient_id: Mapped[int] = mapped_column(
        ForeignKey("patients.id", ondelete="RESTRICT"), index=True
    )
    # 11 clinical features (copied from form at classification time)
    age: Mapped[int]
    ap_hi: Mapped[int]
    ap_lo: Mapped[int]
    cholesterol: Mapped[int]  # 1/2/3
    gluc: Mapped[int]         # 1/2/3
    smoke: Mapped[int]        # 0/1
    alco: Mapped[int]         # 0/1
    active: Mapped[int]       # 0/1
    height: Mapped[int]
    weight: Mapped[float]
    # 3 output fields
    classification_result: Mapped[int]  # 0/1 (map to "no_riesgo"/"riesgo")
    risk_score: Mapped[float]           # 0.0-1.0
    classification_date: Mapped[datetime] = mapped_column(default=func.now())

    patient: Mapped["Patient"] = relationship(back_populates="classifications")
```

### Report Endpoint (`src/app/api/reportes.py`)

```python
@router.get("/pacientes/{patient_id}/reporte")
async def paciente_reporte(
    patient_id: int,
    request: Request,
    db: AsyncSession = Depends(get_db)
):
    patient = await patient_service.get_patient(db, patient_id)
    if not patient:
        raise HTTPException(404, "Paciente no encontrado")
    classifications = await history_service.get_by_patient(db, patient_id)
    return templates.TemplateResponse(
        "reportes/paciente.html",
        {"request": request, "patient": patient, "classifications": classifications}
    )
```

### Report Template (`src/app/templates/reportes/paciente.html`)

- **Extends**: `base.html`
- **Header**: Patient name, gender
- **Table** (Bootstrap 5): Fecha, Clasificación (0/1), Risk Score (valor continuo tal cual lo da el modelo)
- **Empty state**: "Sin clasificaciones registradas" when no classifications exist
- **Vanilla JS**: None needed (static HTML render)
- **No categorías de riesgo intermedias** (Alto/Medio/Bajo) ni umbrales inventados — solo `classification_result` binario 0/1 y `risk_score` continuo

### History Service Addition (`src/app/services/history_service.py`)

```python
class HistoryService:
    def get_by_patient(self, db: AsyncSession, patient_id: int) -> list[Classification]:
        stmt = (
            select(Classification)
            .where(Classification.patient_id == patient_id)
            .order_by(Classification.classification_date.desc())
        )
        result = await db.execute(stmt)
        return result.scalars().all()
```

### Services

```python
# services/patient_service.py
class PatientService:
    def create(self, data: PatientCreate) -> Patient:
        patient = Patient(name=data.name, gender=data.gender)
        self.session.add(patient)
        self.session.commit()
        self.session.refresh(patient)
        return patient

    def get(self, patient_id: int) -> Patient | None:
        return self.session.get(Patient, patient_id)

# services/classification_service.py
class ClassificationService:
    def create_classification(
        self, patient_id: int, features: ClassificationInput
    ) -> Classification:
        patient = self.session.get(Patient, patient_id)
        if not patient:
            raise NotFoundError("Patient not found")

        ml_service = get_ml_service()
        result, risk_score = ml_service.classify(features.model_dump())

        classification = Classification(
            patient_id=patient_id,
            age=features.age,
            ap_hi=features.ap_hi,
            ap_lo=features.ap_lo,
            cholesterol=features.cholesterol,
            gluc=features.gluc,
            smoke=int(features.smoke),
            alco=int(features.alco),
            active=int(features.active),
            height=int(features.height),
            weight=features.weight,
            classification_result=1 if result == "riesgo" else 0,
            risk_score=risk_score,
        )
        self.session.add(classification)
        self.session.commit()
        self.session.refresh(classification)
        return classification

    def list_paginated(
        self, patient_id: int, page: int = 1, page_size: int = 10
    ) -> tuple[list[Classification], int]:
        stmt = (
            select(Classification)
            .where(Classification.patient_id == patient_id)
            .order_by(Classification.classification_date.desc())
            .offset((page - 1) * page_size)
            .limit(page_size)
        )
        items = self.session.execute(stmt).scalars().all()

        total = self.session.execute(
            select(func.count(Classification.id)).where(
                Classification.patient_id == patient_id
            )
        ).scalar()

        return items, total
```

---

## Testing Strategy

**Test-After per Phase** — tests are implemented after each phase completes. Commands to run:

| Phase | After Completing | Test Command | Validates |
|-------|------------------|--------------|-----------|
| **Phase 1** | ML pipeline + training | `pytest tests/test_ml.py` | Recall validation on holdout set |
| **Phase 2** | Patient API (CRUD) | `pytest tests/test_patients.py` | CRUD contracts, validation, 404 handling |
| **Phase 3** | Classification API (ML + DB) | `pytest tests/test_classification.py` | ML inference integration, 11 features stored, risk_score mapping |
| **Phase 4** | History + Reporte | `pytest tests/test_history.py`<br>`pytest tests/test_reportes.py` | Pagination, HTML rendering, summary cards, empty state |

---

| Layer | What to Test | Approach |
|-------|--------------|----------|
| **Unit** | `MLService.classify()` — valid/invalid inputs, probability → label mapping | Mock pipeline; parametrize fixtures for threshold boundaries |
| **Unit** | `PatientCreate` validators — name length, gender enum | Pydantic validation tests |
| **Unit** | `ClassificationInput` validators — ranges, BP cross-field, enums | Parametrized tests for each constraint |
| **Integration** | POST `/pacientes` → DB persistence → 201 + Location | TestClient with test DB; verify JSON response |
| **Integration** | POST `/clasificar` → ML inference → DB persistence | Mock MLService; verify 11 features stored + result |
| **Integration** | GET `/pacientes/{id}/clasificaciones` pagination | Seed 25 rows; verify page=1 returns 10, page=2 returns 10, page=3 returns 5 |
| **Integration** | Patient not found → 404 on classification | POST with invalid patient_id |
| **Integration** | GET `/pacientes/{id}/reporte` HTML rendering | Verify template context, summary cards, table rows, empty state |
| **ML** | `train.py` produces model.joblib with recall ≥ baseline | Run on CI; assert metric on holdout |

---

## Migration / Rollout

**No data migration required** — greenfield project.

**Rollout plan:**
1. **Phase 1**: `train.py` + `ml/` + `config.py` + `database.py` → model artifact + DB schema
2. **Phase 2**: Patient API + schemas + services + templates → ingestion working
3. **Phase 3**: Classification API + ML service integration → classification working
4. **Phase 4**: History API + pagination + templates → history working
5. **Phase 5**: Reporte module (`api/reportes.py`, `templates/reportes/`, `services/history_service.py`) → reports working
6. **Phase 6**: Tests + uv lock + documentation

---

## Open Questions

- [ ] **Training data schema**: Confirm `data/cardio.csv` columns match 11 features expected by inference
- [ ] **Threshold value**: What recall-optimized threshold from training? (persist in model artifact or config)
- [ ] **Boolean storage**: Spec says smoke/alco/active are 0/1 in DB but bool in schema — confirm conversion logic
- [ ] **Height type**: Spec says `height: Mapped[int]` but schema uses `float` — align

---

## Next Step

Ready for **sdd-tasks** phase — break design into implementation tasks with acceptance criteria.