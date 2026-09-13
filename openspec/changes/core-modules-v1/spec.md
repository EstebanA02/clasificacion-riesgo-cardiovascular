# Delta for ML Pipeline

## ADDED Requirements

### Requirement: REQ-ML-01 — Offline Training Script (train.py)

The system SHALL provide an offline training script `train.py` that loads `data/cardio.csv`, trains three models (Logistic Regression, Random Forest, XGBoost) with scikit-learn Pipelines including preprocessing, selects the best model by **recall** metric on validation set, and serializes the best fitted pipeline with joblib to `models/model.joblib`.

#### Scenario: Successful training produces valid model artifact

- GIVEN `data/cardio.csv` exists with required columns
- WHEN `train.py` executes
- THEN `models/model.joblib` is created
- AND the serialized pipeline achieves highest recall among the three algorithms on holdout validation

#### Scenario: Training fails gracefully on missing data

- GIVEN `data/cardio.csv` is missing or malformed
- WHEN `train.py` executes
- THEN script exits with non-zero code and clear error message

### Requirement: REQ-ML-02 — FastAPI Lifespan Model Loading

The system SHALL load the serialized model exactly once at application startup via FastAPI lifespan events and store it in `app.state.model` for reuse across requests.

#### Scenario: Model loaded at startup

- GIVEN FastAPI application starts
- WHEN lifespan startup event runs
- THEN `app.state.model` contains the deserialized pipeline from `models/model.joblib`

#### Scenario: Startup fails if model artifact missing

- GIVEN `models/model.joblib` does not exist
- WHEN application starts
- THEN startup fails with explicit error indicating missing model

### Requirement: REQ-ML-03 — Inference Function

The system SHALL provide a `classify(patient_features: dict) -> tuple[str, float]` function that uses the loaded pipeline to return (classification_result, risk_score) where classification_result ∈ {"riesgo", "no_riesgo"} and risk_score ∈ [0.0, 1.0].

#### Scenario: Valid features return classification tuple

- GIVEN valid patient feature dictionary with 11 clinical features
- WHEN `classify(features)` is called
- THEN returns tuple of (classification_result: "riesgo"/"no_riesgo", risk_score: 0.0-1.0)

#### Scenario: Invalid features raise validation error

- GIVEN feature dictionary missing required fields or with invalid types
- WHEN `classify(features)` is called
- THEN raises ValueError with descriptive message

---

# Delta for Patient Ingestion

## ADDED Requirements

### Requirement: REQ-PI-01 — PatientCreate Schema (Name + Gender Only)

The system SHALL define a Pydantic v2 model `PatientCreate` with only two fields: `name` (str, min_length=1, max_length=100) and `gender` (Literal["M", "F"]). Clinical variables are NOT in this schema — they belong to classifications only.

#### Scenario: Valid patient data passes validation

- GIVEN form data with name="Juan Pérez", gender="M"
- WHEN `PatientCreate(**data)` is instantiated
- THEN validation succeeds

#### Scenario: Empty name rejects

- GIVEN form data with name="", gender="M"
- WHEN `PatientCreate(**data)` is instantiated
- THEN ValidationError raised with name constraint message

#### Scenario: Invalid gender rejects

- GIVEN form data with name="Juan", gender="X"
- WHEN `PatientCreate(**data)` is instantiated
- THEN ValidationError with gender constraint message

### Requirement: REQ-PI-02 — POST /pacientes Endpoint

The system SHALL expose POST `/pacientes` that creates a patient from validated `PatientCreate`, returns 201 Created with `Location: /pacientes/{id}` header on success, and returns 422 Unprocessable Entity with JSON error details on failure.

#### Scenario: Valid request creates patient and returns 201

- GIVEN valid PatientCreate payload
- WHEN POST `/pacientes` with JSON body
- THEN response 201 with Location header pointing to new patient

#### Scenario: Invalid data returns 422 with JSON errors

- GIVEN payload with empty name
- WHEN POST `/pacientes` with JSON body
- THEN response 422 with JSON error details

### Requirement: REQ-PI-03 — GET /pacientes/{id} Endpoint

The system SHALL expose GET `/pacientes/{id}` returning patient data (id, name, gender) without clinical fields.

#### Scenario: Existing patient returns data

- GIVEN patient exists
- WHEN GET `/pacientes/{id}`
- THEN response 200 with patient object (id, name, gender)

#### Scenario: Non-existent patient returns 404

- GIVEN patient ID does not exist
- WHEN GET `/pacientes/{id}`
- THEN response 404

### Requirement: REQ-PI-04 — Jinja2 + Bootstrap 5 Form with Vanilla JS Validation

The system SHALL render a patient form using Jinja2 templates with Bootstrap 5 (CDN) and vanilla JavaScript for client-side validation. No HTMX, no Alpine.js.

#### Scenario: Form renders with Bootstrap 5 styling

- GIVEN GET `/pacientes/nuevo` (or equivalent page route)
- WHEN template renders
- THEN form includes name input, gender select, submit button with Bootstrap 5 classes

#### Scenario: Vanilla JS validates before submit

- GIVEN form with empty name field
- WHEN user clicks submit
- THEN vanilla JS prevents submit and shows inline error message

#### Scenario: Valid form submits via fetch to POST /pacientes

- GIVEN form with valid name and gender
- WHEN user clicks submit
- THEN vanilla JS sends fetch POST to `/pacientes`, handles 201/422 responses

---

# Delta for Classification

## ADDED Requirements

### Requirement: REQ-CL-01 — ClassificationInput Schema (11 Clinical Features)

The system SHALL define a Pydantic v2 model `ClassificationInput` with exactly 11 clinical features: age (int), ap_hi (int), ap_lo (int), cholesterol (Literal[1,2,3]), gluc (Literal[1,2,3]), smoke (bool), alco (bool), active (bool), height (float), weight (float), gender (Literal["M","F"]). All fields required.

#### Scenario: Valid clinical data passes validation

- GIVEN all 11 fields with valid values
- WHEN `ClassificationInput(**data)` is instantiated
- THEN validation succeeds

#### Scenario: Systolic ≤ diastolic rejects

- GIVEN ap_hi=80, ap_lo=120
- WHEN `ClassificationInput(**data)` is instantiated
- THEN ValidationError raised with message "Presión sistólica debe ser > diastólica"

#### Scenario: Age out of range rejects

- GIVEN age=15 or age=101
- WHEN `ClassificationInput(**data)` is instantiated
- THEN ValidationError with age constraint message

#### Scenario: Invalid cholesterol/gluc values reject

- GIVEN cholesterol=4 or gluc=0
- WHEN `ClassificationInput(**data)` is instantiated
- THEN ValidationError with enum constraint message

### Requirement: REQ-CL-02 — POST /clasificar Endpoint (Manual Trigger)

The system SHALL expose POST `/clasificar` that accepts `patient_id` and `ClassificationInput`, calls `MLService.classify()`, stores result in `classifications` table with fields: patient_id, all 11 features, classification_result, risk_score, classification_date, and returns the created classification.

#### Scenario: Valid request classifies and stores result

- GIVEN existing patient_id and valid 11 clinical features
- WHEN POST `/clasificar` with JSON body
- THEN response 201 with classification object including classification_result, risk_score, classification_date
- AND row inserted in classifications table with all 11 features + 3 output fields

#### Scenario: Non-existent patient returns 404

- GIVEN non-existent patient_id
- WHEN POST `/clasificar`
- THEN response 404

#### Scenario: Invalid clinical data returns 422

- GIVEN payload with ap_hi ≤ ap_lo
- WHEN POST `/clasificar`
- THEN response 422 with JSON error details

### Requirement: REQ-CL-03 — Classification Output Uses Approved Terminology

The system SHALL use "clasificación" / "clasificar" terminology exclusively — never "predicción" / "predecir". The output field is `risk_score` (float 0-1), NOT risk_probability, NOT risk_level.

#### Scenario: API response uses classification_result and risk_score

- GIVEN classification created
- WHEN GET classification or POST /clasificar response
- THEN JSON contains `classification_result` ("riesgo"/"no_riesgo") and `risk_score` (float)
- AND does NOT contain "prediccion", "risk_probability", "risk_level"

#### Scenario: Database stores classification_result and risk_score

- GIVEN classification row created
- WHEN querying classifications table
- THEN columns are `classification_result` and `risk_score`
- AND no risk_level or risk_probability columns exist

---

# Delta for History

## ADDED Requirements

### Requirement: REQ-HS-01 — GET /pacientes/{id}/clasificaciones with Pagination

The system SHALL expose GET `/pacientes/{id}/clasificaciones` returning paginated list of classifications for a patient, ordered by `classification_date DESC`, with simple offset/limit pagination (page, page_size).

#### Scenario: First page returns most recent classifications

- GIVEN patient with 25 classifications exists
- WHEN GET `/pacientes/{id}/clasificaciones?page=1&page_size=10`
- THEN response 200 with 10 most recent classifications (by classification_date DESC)

#### Scenario: Subsequent page returns next batch

- GIVEN patient with 25 classifications
- WHEN GET `/pacientes/{id}/clasificaciones?page=2&page_size=10`
- THEN response 200 with next 10 classifications

#### Scenario: Empty result for patient with no classifications

- GIVEN patient exists with zero classifications
- WHEN GET `/pacientes/{id}/clasificaciones`
- THEN response 200 with empty list

### Requirement: REQ-HS-02 — Simple List View (Jinja2 + Bootstrap)

The system SHALL render a simple HTML list view of classifications using Jinja2 template with Bootstrap 5 (CDN), showing: classification_date, classification_result, risk_score. No HTMX, no complex interactions.

#### Scenario: History page renders classification list

- GIVEN GET `/pacientes/{id}/historial` (or equivalent page route)
- WHEN template renders with classifications data
- THEN table/list shows each classification with date, result, risk_score
- AND Bootstrap 5 styling applied

---

# Delta for Reporte (Module 5)

## ADDED Requirements

### Requirement: REQ-RP-01 — GET /pacientes/{id}/reporte Endpoint (HTML Only)

The system SHALL expose GET `/pacientes/{id}/reporte` that returns a Jinja2-rendered HTML report using template `templates/reportes/paciente.html`. The report SHALL include:
- Patient info: name, gender
- Classifications table: date, classification_result, risk_score (ordered by classification_date DESC)
- Summary: total classifications count, risk distribution (count of "riesgo" vs "no_riesgo")

**PDF generation is explicitly excluded.** No WeasyPrint, no PDF libraries.

#### Scenario: Patient with classifications returns 200 with HTML report

- GIVEN patient exists with one or more classifications
- WHEN GET `/pacientes/{id}/reporte`
- THEN response 200 with Content-Type text/html
- AND HTML contains patient name and gender
- AND HTML contains table with all classifications (date, classification_result, risk_score)
- AND HTML contains summary section with total count and risk distribution

#### Scenario: Patient without classifications returns 200 with "Sin clasificaciones"

- GIVEN patient exists with zero classifications
- WHEN GET `/pacientes/{id}/reporte`
- THEN response 200 with Content-Type text/html
- AND HTML contains "Sin clasificaciones" message
- AND summary shows total: 0, riesgo: 0, no_riesgo: 0

#### Scenario: Invalid patient_id returns 404

- GIVEN patient_id does not exist
- WHEN GET `/pacientes/{id}/reporte`
- THEN response 404

### Requirement: REQ-RP-02 — Report Template (Bootstrap 5, Extends base.html)

The system SHALL implement the report template at `templates/reportes/paciente.html` that:
- Extends `templates/base.html`
- Uses Bootstrap 5 (CDN) classes for styling
- Uses vanilla JavaScript only (no HTMX, no Alpine.js)
- Uses terminology "clasificación" / "clasificar" exclusively
- Uses field name `risk_score` (never risk_probability, risk_level)

#### Scenario: Template renders valid HTML structure

- GIVEN template rendered with patient and classifications data
- WHEN HTML output inspected
- THEN extends base.html
- THEN uses Bootstrap 5 table classes (table, table-striped, table-hover)
- THEN includes summary cards/alerts with risk distribution
- THEN no PDF-related links or buttons present

---

# Delta for Test Strategy

## ADDED Requirements

### Requirement: REQ-TEST-01 — Test-After Strategy (Per Module / PR)

The system SHALL follow a **Test-After** approach: tests are written **AFTER** each module/PR implementation, not before (not strict TDD).

Test types required per module:
- **Integration tests** — verify API contracts (status codes, response shapes, headers)
- **ML validation tests** — verify model recall ≥ 0.85 on holdout set
- **Edge case tests** — invalid inputs, missing data, boundary conditions

Execution:
- Run `pytest` at the end of each PR before merge
- All tests must pass for PR to be merged
- No separate test phase; tests are part of the module deliverable

#### Scenario: ML Pipeline module (Module 1) includes tests

- GIVEN train.py and inference.py implemented
- WHEN tests added for Module 1
- THEN integration tests cover lifespan model loading and classify() function
- THEN ML validation test asserts recall ≥ 0.85 on holdout
- THEN edge case tests cover missing model file, invalid features
- THEN `pytest` passes

#### Scenario: Patient Ingestion module (Module 2) includes tests

- GIVEN POST/GET /pacientes and form implemented
- WHEN tests added for Module 2
- THEN integration tests cover 201/422/404 responses for API endpoints
- THEN edge case tests cover empty name, invalid gender, non-existent patient
- THEN form rendering and JS validation behavior tested
- THEN `pytest` passes

#### Scenario: Classification module (Module 3) includes tests

- GIVEN POST /clasificar implemented
- WHEN tests added for Module 3
- THEN integration tests cover 201/404/422 for classification endpoint
- THEN ML validation: classification_result and risk_score correctness
- THEN edge cases: systolic ≤ diastolic, age bounds, enum values
- THEN `pytest` passes

#### Scenario: History module (Module 4) includes tests

- GIVEN GET /pacientes/{id}/clasificaciones and list view implemented
- WHEN tests added for Module 4
- THEN integration tests cover pagination (page 1, page 2, empty)
- THEN edge cases: non-existent patient, empty result
- THEN list view renders correctly
- THEN `pytest` passes

#### Scenario: Reporte module (Module 5) includes tests

- GIVEN GET /pacientes/{id}/reporte and template implemented
- WHEN tests added for Module 5
- THEN integration tests cover 200 (with data), 200 (sin clasificaciones), 404
- THEN HTML content assertions: patient info, classifications table, summary
- THEN terminology check: "clasificación"/"clasificar", risk_score
- THEN `pytest` passes

---

# Stack Constraints (Enforced)

The following are NOT included and SHALL NOT be added:
- ❌ HTMX
- ❌ Alembic
- ❌ WeasyPrint / PDF generation
- ❌ PSI drift calculation
- ❌ Playwright / E2E tests
- ❌ ruff / mypy / pytest-cov / openpyxl

### Approved Stack Only:
- FastAPI, Pydantic v2, SQLAlchemy 2.x, SQLite
- Jinja2 + Bootstrap 5 (CDN), vanilla JS only
- scikit-learn, XGBoost, joblib
- pytest + httpx
- uv only

### Database Schema — FIXED:
```
patients: id (PK), name, gender          -- ONLY 3 fields
classifications: 
  id (PK), patient_id (FK), 
  age, ap_hi, ap_lo, cholesterol, gluc, smoke, alco, active, height, weight,  -- 11 clinical features
  classification_result, risk_score, classification_date                     -- 3 output fields
```

### Terminology Lock:
- "clasificación" / "clasificar" ONLY — never "predicción" / "predecir"
- Field: `risk_score` (float 0-1)

### Algorithms:
- 3 models: Logistic Regression, Random Forest, XGBoost
- Selection metric: **recall** (clinical priority)
- Best model selected → serialized as `model.joblib`

### 5 Modules (Task Organization):
1. **ML Pipeline** — train.py (3 algorithms, recall selection, joblib), inference.py (lifespan load, classify())
2. **Patient Ingestion** — POST/GET /pacientes (name, gender only), Jinja2 + Bootstrap forms, vanilla JS validation
3. **Classification** — POST /clasificar (manual trigger), uses ML service, stores in classifications table
4. **History** — GET /pacientes/{id}/clasificaciones, pagination, simple list view
5. **Reporte** — GET /pacientes/{id}/reporte (HTML only, Jinja2 + Bootstrap 5, extends base.html, vanilla JS)