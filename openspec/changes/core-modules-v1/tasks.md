# Tasks: core-modules-v1 — Cardiovascular Risk Classification

## Phase Structure (6 Phases)

| Phase | Focus | Tasks | Mentoring Notes |
|-------|-------|-------|-----------------|
| 1 | Foundation + ML | 6 | Scaffold: all files. You: `train.py`, `inference.py`. Together: `pytest tests/test_ml.py` |
| 2 | Patient Ingestion | 5 | Scaffold: schemas, API, service, templates. You: API logic + form. Together: `pytest tests/test_patients.py` |
| 3 | Classification | 5 | Scaffold: schemas, API, service, form. You: service + form. Together: `pytest tests/test_classification.py` |
| 4 | History | 4 | Scaffold: schemas, API, service, template. You: API + template. Together: `pytest tests/test_history.py` |
| 5 | Reporte | 2 | Scaffold: API + template. You: template. Together: `pytest tests/test_reportes.py` |
| 6 | Tests + Docs | 2 | Run full `pytest`, update README, verify all contracts |

**Total: ~24 tasks**

---

## PR Split

| PR | Phases | Base Branch |
|----|--------|-------------|
| PR 1 | Phase 1 | `main` |
| PR 2 | Phase 2 | `PR 1` |
| PR 3 | Phase 3 | `PR 2` |
| PR 4 | Phase 4 + 5 | `PR 3` |

---

## Mentoring Pattern (Per Task)

```markdown
## Task X.Y: [Title]
**Scaffold (yo)**: Creo archivo X con imports, type hints, docstrings, función vacía + test con `pass`
**Implementas (tú)**: Rellenas lógica en función principal
**Revisamos (juntos)**: Code review en vivo, feedback
**Test (juntos)**: Escribimos test de integración al final del PR
```

**Acceptance Criteria**: Each PR must pass `pytest` for its test file(s) before merge.

---

## Phase 1: Foundation + ML Pipeline (6 Tasks)

### Task 1.1: Project Scaffold + Config
**Scaffold (yo)**: Create `src/app/` structure, `config.py` (Settings with `DATABASE_URL`, `MODEL_PATH`), `database.py` (engine, sessionmaker, Base), `main.py` (FastAPI app, lifespan stub, `create_all`), `pyproject.toml` (uv deps: fastapi, pydantic, pydantic-settings, sqlalchemy, uvicorn, scikit-learn, xgboost, joblib, pytest, httpx, jinja2)
**Implementas (tú)**: Wire settings, ensure `MODEL_PATH` points to `models/model.joblib`, add `create_all` in lifespan startup
**Revisamos (juntos)**: Verify imports, type hints, no circular deps, config loads from env
**Test (juntos)**: `pytest tests/test_config.py` (create if needed) — validates settings load

### Task 1.2: Database Models
**Scaffold (yo)**: Create `src/app/models/patient.py` (Patient: id, name, gender), `src/app/models/classification.py` (Classification: id, patient_id FK, 11 clinical features, classification_result 0/1, risk_score, classification_date)
**Implementas (tú)**: Add relationships, indexes, `__repr__`, ensure boolean fields stored as 0/1 int
**Revisamos (juntos)**: Verify FK cascade, column types match spec (height=int, weight=float), terminology lock
**Test (juntos)**: `pytest tests/test_models.py` — create tables, insert, query back

### Task 1.3: ML Schemas
**Scaffold (yo)**: Create `src/app/ml/schemas.py` with `ClassificationInput` (11 features + cross-field validator for ap_hi > ap_lo, age 18-100 **en AÑOS**, enums for cholesterol/gluc) and `ClassificationOutput`
**Implementas (tú)**: Add all validators, ensure Literal enums match spec exactly. **IMPORTANTE**: El validador de `age` debe esperar y validar AÑOS (18-100), no días — porque el usuario ingresa edad en años desde el formulario.
**Revisamos (juntos)**: Test edge cases: systolic ≤ diastolic, age bounds (18-100 años), enum values
**Test (juntos)**: `pytest tests/test_ml_schemas.py` — parametrize all validation scenarios

### Task 1.4: Training Script (train.py)
**Scaffold (yo)**: Create root `train.py` with imports, type hints, docstrings, empty `train()` function, `if __name__ == "__main__": train()`, argparse for `--data-path`, `--model-path`, `--report-path`, `--cv-folds`
**Implementas (tú)**: 
- Load `data/cardio_train.csv` con **separador `;`** (`pd.read_csv(..., sep=';')`)
- **Descartar columna `id`** (índice del dataset, no es feature)
- **Convertir `age` de días a años**: `age_years = age_days // 365` ANTES de entrenar
- **Filtrar filas fisiológicamente imposibles ANTES de entrenar** (documentado con hallazgos EDA):
  - `ap_hi < 70` o `ap_hi > 250`
  - `ap_lo < 40` o `ap_lo > 200`
  - `ap_hi <= ap_lo`
  - `height < 120` o `height > 220`
  - `weight < 30` o `weight > 200`
  - Total esperado: ~1.8% del dataset (~1226 + 60 registros)
  - **Hallazgos EDA documentados en comentario**:
    - `ap_hi` min -150, max 16020 (errores de captura)
    - `height` min 55 (error de captura)
    - `weight` min 10 (error de captura)
    - Rangos de presión más amplios que límites clínicos AHA/ACC 2025
- **StratifiedKFold(n_splits=5, shuffle=True, random_state=42)** sobre TODO el dataset (parametrizable con `--cv-folds`, default 5)
- Mantener los 3 Pipelines existentes sin cambios
- Evaluar cada pipeline con `sklearn.model_selection.cross_validate` con `scoring={"accuracy","precision","recall","f1"}`. Guardar media y desviación estándar de cada métrica por modelo
- Seleccionar el modelo con mayor MEDIA de recall
- Reentrenar (fit) el pipeline ganador con el 100 % de los datos y guardarlo con `joblib.dump` en `models/model.joblib` (sigue siendo un Pipeline con `predict_proba`; el contrato de MLService no cambia)
- Imprimir una tabla con las 4 métricas (media ± desv.) de los 3 modelos, marcando el seleccionado
- Guardar el mismo resumen en `models/train_report.json`: métricas por modelo, nombre del seleccionado, MIN_RECALL (constante = 0.85 definida al inicio del archivo) y "meets_min_recall" (bool)
- Si el recall medio del ganador es < MIN_RECALL: NO lanzar error. Imprimir una advertencia con el valor obtenido y la brecha, y continuar guardando el modelo
**Revisamos (juntos)**: Verify recall is selection metric, pipeline includes preprocessing, model saved correctly, age convertido, id descartado, filtros aplicados
**Test (juntos)**: Run `python train.py` (requires `data/cardio_train.csv`), verify `models/model.joblib` and `models/train_report.json` created, then `pytest tests/test_ml.py` (ML validation: recall medio en validación cruzada; si es < 0.85, advertencia, no fallo)

### Task 1.5: Inference Service (MLService)
**Scaffold (yo)**: Create `src/app/ml/inference.py` with `MLService` class: `__init__(model_path: Path)`, `classify(features: dict) -> tuple[str, float]` with empty body + `pass`, `from_app_state(cls, request: Request) -> MLService` classmethod
**Implementas (tú)**:
- Load pipeline in `__init__` via `joblib.load`
- In `classify`: validate input against `ClassificationInput`, convert to DataFrame, **mapeo de género: 'F'→1, 'M'→2** (dataset usa 1=Mujer, 2=Hombre; BD guarda 'M'/'F'; conversión SOLO aquí), `predict_proba[:,1]`, map to "riesgo"/"no_riesgo" using threshold (embedded or config), return (label, float(probability))
- **Documentar en inference.py**: comentario explicando por qué la conversión de género ocurre aquí y no en otro módulo
**Revisamos (juntos)**: Verify single load at startup, threshold handling, error messages for invalid features, mapeo género correcto
**Test (juntos)**: `pytest tests/test_ml.py` — integration: lifespan load, classify() returns correct tuple, edge cases

### Task 1.6: FastAPI Lifespan + Main App
**Scaffold (yo)**: Update `main.py` — add lifespan context manager, instantiate `MLService` from `settings.MODEL_PATH`, store in `app.state.ml_service`, exception handlers for 404/422
**Implementas (tú)**: Wire lifespan startup/shutdown, ensure startup fails if model missing, add routers (empty placeholders for patients, classification, history, reportes)
**Revisamos (juntos)**: Test startup with/without model, verify `app.state.ml_service` accessible via `Depends`
**Test (juntos)**: `pytest tests/test_ml.py` — full integration: startup loads model, classify works via DI

---

## Phase 2: Patient Ingestion (5 Tasks)

### Task 2.1: Patient Schemas
**Scaffold (yo)**: Create `src/app/schemas/patient.py` with `PatientCreate` (name: str min1 max100, gender: Literal["M","F"]) and `PatientRead` (id, name, gender, from_attributes=True)
**Implementas (tú)**: Add any additional validators if needed
**Revisamos (juntos)**: Confirm clinical fields NOT in schema
**Test (juntos)**: `pytest tests/test_patients.py` — schema validation tests

### Task 2.2: Patient Service
**Scaffold (yo)**: Create `src/app/services/patient_service.py` with `PatientService` class: `create(data: PatientCreate) -> Patient`, `get(patient_id: int) -> Patient | None`
**Implementas (tú)**: Implement DB insert with commit/refresh, query by PK
**Revisamos (juntos)**: Verify session handling, no leaks
**Test (juntos)**: `pytest tests/test_patients.py` — service layer tests with test DB

### Task 2.3: Patient API Routes
**Scaffold (yo)**: Create `src/app/api/patients.py` with router, `POST /pacientes`, `GET /pacientes/{id}`, `GET /pacientes/nuevo` (form page), dependency for `PatientService`
**Implementas (tú)**:
- POST: validate, service.create(), return 201 with Location header
- GET by id: service.get(), 404 if not found
- GET /nuevo: render `templates/patients/form.html`
**Revisamos (juntos)**: Verify status codes, headers, error responses match spec
**Test (juntos)**: `pytest tests/test_patients.py` — integration: 201/422/404 contracts

### Task 2.4: Patient Form Template + Vanilla JS
**Scaffold (yo)**: Create `templates/base.html` (Bootstrap 5 CDN, blocks for content/scripts), `templates/patients/form.html` (extends base, form with name input, gender select, submit, Bootstrap classes), `static/js/main.js` (empty, placeholder for validation)
**Implementas (tú)**:
- `form.html`: vanilla JS prevent submit on empty name, show inline error, fetch POST to `/pacientes` on valid, handle 201 redirect, 422 show errors
- `main.js`: reusable validation helpers
**Revisamos (juntos)**: No HTMX/Alpine, Bootstrap 5 classes only, JS vanilla
**Test (juntos)**: `pytest tests/test_patients.py` — template renders, JS validation logic (can test via static analysis or simple integration)

### Task 2.5: Patient Detail Template
**Scaffold (yo)**: Create `templates/patients/detail.html` (extends base, shows patient name/gender, "Clasificar" button linking to classification form)
**Implementas (tú)**: Add button/link to `/pacientes/{id}/clasificar` (to be created in Phase 3)
**Revisamos (juntos)**: Consistent styling, correct URL
**Test (juntos)**: `pytest tests/test_patients.py` — template integration

---

## Phase 3: Classification (5 Tasks)

### Task 3.1: Classification Schemas (API)
**Scaffold (yo)**: Create `src/app/schemas/classification.py` with `ClassificationInput` (re-export or alias from ml/schemas), `ClassificationOutput` (id, patient_id, classification_result, risk_score, classification_date, from_attributes=True)
**Implementas (tú)**: Ensure terminology lock — `classification_result`, `risk_score` only
**Revisamos (juntos)**: Cross-field validator works, enums match
**Test (juntos)**: `pytest tests/test_classification.py` — schema validation

### Task 3.2: Classification Service
**Scaffold (yo)**: Create `src/app/services/classification_service.py` with `ClassificationService`: `create_classification(patient_id: int, features: ClassificationInput) -> Classification`, `list_paginated(patient_id: int, page: int, page_size: int) -> tuple[list, int]`
**Implementas (tú)**:
- `create_classification`: verify patient exists, call `MLService.classify()`, build Classification ORM with all 11 features + 3 outputs, commit
- `list_paginated`: ORDER BY classification_date DESC, offset/limit, count total
**Revisamos (juntos)**: Verify all 11 features persisted, boolean→int conversion, risk_score float
**Test (juntos)**: `pytest tests/test_classification.py` — service tests with mocked MLService

### Task 3.3: Classification API Routes
**Scaffold (yo)**: Create `src/app/api/classification.py` with router, `POST /clasificar`, `GET /pacientes/{id}/clasificar` (form page), deps for ClassificationService, MLService
**Implementas (tú)**:
- POST: validate patient exists (404), validate input (422), service.create_classification(), return 201 with ClassificationOutput
- GET /clasificar: render `templates/classification/form.html` with patient context
**Revisamos (juntos)**: 404 for missing patient, 422 for invalid clinical data, 201 response shape
**Test (juntos)**: `pytest tests/test_classification.py` — integration: 201/404/422, ML inference integration

### Task 3.4: Classification Form Template + Vanilla JS
**Scaffold (yo)**: Create `templates/classification/form.html` (extends base, 11 field form: age, ap_hi, ap_lo, cholesterol select, gluc select, smoke/alco/active checkboxes, height, weight, gender hidden from patient, submit)
**Implementas (tú)**: Vanilla JS validation (required, ranges, systolic > diastolic), fetch POST to `/clasificar`, handle responses, show result badge + risk score
**Revisamos (juntos)**: Bootstrap 5 only, no HTMX, all 11 fields present, terminology correct
**Test (juntos)**: `pytest tests/test_classification.py` — form renders, JS validation

### Task 3.5: Wire Classification into Patient Detail
**Scaffold (yo)**: Update `templates/patients/detail.html` — add classification result display area (empty initially), link/button to classification form
**Implementas (tú)**: After classification POST, update UI with result badge ("riesgo"/"no_riesgo") and risk score
**Revisamos (juntos)**: Flow from patient detail → classification form → result display
**Test (juntos)**: `pytest tests/test_classification.py` — end-to-end flow

---

## Phase 4: History (4 Tasks)

### Task 4.1: History API Routes
**Scaffold (yo)**: Create `src/app/api/history.py` with router, `GET /pacientes/{id}/clasificaciones` (pagination params page, page_size), `GET /pacientes/{id}/historial` (list view page), deps for ClassificationService
**Implementas (tú)**:
- JSON endpoint: service.list_paginated(), return {items[], total, page, page_size, total_pages}
- HTML endpoint: fetch all (or paginated), render `templates/history/list.html`
**Revisamos (juntos)**: Pagination math correct, ordering DESC, empty result handling
**Test (juntos)**: `pytest tests/test_history.py` — pagination: page1=10, page2=10, page3=5, empty=0

### Task 4.2: History Service (Extended)
**Scaffold (yo)**: Already exists from Phase 3 (`list_paginated`). Add `get_by_patient(patient_id: int) -> list[Classification]` for report use
**Implementas (tú)**: Implement `get_by_patient` with ORDER BY classification_date DESC
**Revisamos (juntos)**: Reuse existing query logic
**Test (juntos)**: `pytest tests/test_history.py` — service test

### Task 4.3: History List Template + Vanilla JS Pagination
**Scaffold (yo)**: Create `templates/history/list.html` (extends base, Bootstrap table with columns: Fecha, Clasificación, Risk Score, empty state row)
**Implementas (tú)**: Vanilla JS for pagination controls (prev/next/page numbers), fetch JSON endpoint, update table rows
**Revisamos (juntos)**: Bootstrap table classes, JS handles empty state, no HTMX
**Test (juntos)**: `pytest tests/test_history.py` — template renders, pagination works

### Task 4.4: Patient Detail → History Link
**Scaffold (yo)**: Update `templates/patients/detail.html` — add "Ver historial" link to `/pacientes/{id}/historial`
**Implementas (tú)**: Ensure navigation works
**Revisamos (juntos)**: Consistent UX
**Test (juntos)**: `pytest tests/test_history.py` — integration

---

## Phase 5: Reporte (2 Tasks)

### Task 5.1: Report API Endpoint
**Scaffold (yo)**: Create `src/app/api/reportes.py` with router, `GET /pacientes/{patient_id}/reporte`, deps for PatientService, HistoryService
**Implementas (tú)**:
- Get patient (404 if not found)
- Get all classifications via `history_service.get_by_patient()`
- Render `templates/reportes/paciente.html` with context: request, patient, classifications
**Revisamos (juntos)**: 404 handling, context passed correctly, Content-Type text/html
**Test (juntos)**: `pytest tests/test_reportes.py` — 200 with data, 200 sin clasificaciones, 404

### Task 5.2: Report Template
**Scaffold (yo)**: Create `templates/reportes/paciente.html` (extends base.html, blocks for content)
**Implementas (tú)**:
- Header: patient name, gender
- Table (Bootstrap 5: table, table-striped, table-hover): Fecha, Clasificación (0/1), Risk Score (valor continuo tal cual lo da el modelo)
- Empty state: "Sin clasificaciones registradas" when list empty
- Terminology: "clasificación"/"clasificar" only, `risk_score` field name
- No PDF links/buttons
- No umbrales ni categorías de riesgo intermedias (solo classification_result binario 0/1 y risk_score continuo)
**Revisamos (juntos)**: Extends base.html, Bootstrap 5 classes, vanilla JS only (none needed for static render), terminology lock
**Test (juntos)**: `pytest tests/test_reportes.py` — HTML content assertions, table rows with classification_result + risk_score, empty state, terminology check, ausencia de categorías Alto/Medio/Bajo

---

## Phase 6: Tests + Docs (2 Tasks)

### Task 6.1: Full Test Suite + Coverage
**Scaffold (yo)**: Ensure all test files exist: `tests/test_ml.py`, `tests/test_patients.py`, `tests/test_classification.py`, `tests/test_history.py`, `tests/test_reportes.py`, `tests/conftest.py` (fixtures: client, db_session, ml_model, sample_patient)
**Implementas (tú)**: 
- Fill any missing integration tests per spec scenarios
- Run full `pytest` — all must pass
- Verify ML recall medio en validación cruzada (test slow, solo advertencia si < 0.85)
**Revisamos (juntos)**: Test coverage per module, no flaky tests, CI-ready
**Test (juntos)**: `pytest -v` — full suite green

### Task 6.2: Documentation + Lock Files
**Scaffold (yo)**: Create `README.md` with project overview, stack, run commands, API endpoints summary, terminology lock note
**Implementas (tú)**:
- Update `README.md`: install (uv sync), train (uv run train.py), run (uv run uvicorn src.app.main:app), test (uv run pytest)
- Run `uv lock` to generate `uv.lock`
- Verify all contracts from spec are met
**Revisamos (juntos)**: README accurate, lock file committed, all acceptance criteria satisfied
**Test (juntos)**: Fresh clone → uv sync → train.py → uvicorn → pytest all pass (sin pasos manuales de descarga de datos, cardio_train.csv versionado)

---

## Acceptance Criteria Checklist (Per PR)

| PR | Required `pytest` Pass |
|----|------------------------|
| PR 1 (Phase 1) | `pytest tests/test_ml.py` |
| PR 2 (Phase 2) | `pytest tests/test_patients.py` |
| PR 3 (Phase 3) | `pytest tests/test_classification.py` |
| PR 4 (Phase 4+5) | `pytest tests/test_history.py tests/test_reportes.py` |
| **Final** | `pytest` (all) |

---

## File Tree Reference (from Design)

```
src/app/
├── __init__.py
├── main.py
├── config.py
├── database.py
├── models/
│   ├── __init__.py
│   ├── patient.py
│   └── classification.py
├── schemas/
│   ├── __init__.py
│   ├── patient.py
│   └── classification.py
├── api/
│   ├── __init__.py
│   ├── patients.py
│   ├── classification.py
│   ├── history.py
│   └── reportes.py
├── services/
│   ├── __init__.py
│   ├── patient_service.py
│   ├── classification_service.py
│   └── history_service.py
├── ml/
│   ├── __init__.py
│   ├── inference.py
│   └── schemas.py
├── templates/
│   ├── base.html
│   ├── patients/
│   │   ├── form.html
│   │   └── detail.html
│   ├── history/
│   │   └── list.html
│   └── reportes/
│       └── paciente.html
├── static/
│   ├── css/style.css
│   └── js/main.js
train.py
models/model.joblib (gitignored)
data/cardio_train.csv
tests/
├── conftest.py
├── test_ml.py
├── test_patients.py
├── test_classification.py
├── test_history.py
└── test_reportes.py
```