# Exploration: Patient Data Ingestion Module

## Current State
The semillero project is a **greenfield project** — only the project structure (`src/app/__init__.py`) and OpenSpec configuration exist. No implementation code has been written yet.

The existing FastAPI reference project (`C:\Users\Esteban\Documents\Code\FastAPI`) demonstrates:
- SQLAlchemy Core (Table-based) with MySQL
- Basic Pydantic schemas without validation
- Simple CRUD endpoints with raw SQLAlchemy connection
- No clinical validation, no frontend templates, no HTMX

**This reference is insufficient** for our clinical requirements — we need SQLAlchemy ORM (not Core), Pydantic v2 with clinical validators, Jinja2/Bootstrap forms, and proper validation error handling.

---

## Affected Areas (to be created)
| Path | Purpose |
|------|---------|
| `src/app/schemas/patient.py` | Pydantic models with clinical validation |
| `src/app/models/patient.py` | SQLAlchemy ORM models matching fixed schema |
| `src/app/api/patients.py` | FastAPI endpoints: POST/GET patients |
| `src/app/templates/patients/form.html` | Jinja2 + Bootstrap form |
| `src/app/templates/patients/detail.html` | Patient detail view |
| `src/app/templates/base.html` | Base template with Bootstrap CDN |
| `src/app/main.py` | FastAPI app factory, router inclusion |
| `src/app/config/database.py` | SQLite + SQLAlchemy session management |
| `tests/test_patients.py` | TDD tests (unit + integration) |

---

## Approaches Compared

### 1. Patient Schema: Pydantic Models

| Approach | Description | Pros | Cons | Effort |
|----------|-------------|------|------|--------|
| **A: Single model with Field validators** | One `PatientCreate` model with `@field_validator` for ranges/enums | Simple, all validation in one place | Mixes API schema with domain logic | Low |
| **B: Separate schemas per layer** | `PatientCreate` (API) → `PatientInDB` (ORM) → `PatientResponse` (output) | Clean separation, OpenAPI docs accurate | More files, mapping boilerplate | Medium |
| **C: Pydantic v2 `model_validator` + `Annotated`** | Use `Annotated[float, Field(ge=0, le=300)]` + custom validators | Type-safe, self-documenting, composable | Requires Pydantic v2 knowledge | Medium |

**Recommendation: C** — Pydantic v2's `Annotated` + `Field` constraints + `@model_validator(mode='after')` for cross-field clinical rules (e.g., `ap_hi > ap_lo`). Aligns with modern best practices and the project's `Python >=3.13` requirement.

---

### 2. Clinical Validations (Ranges & Enums)

| Field | Type | Constraints | Source |
|-------|------|-------------|--------|
| `age` | int | 18–100 | Clinical |
| `gender` | enum | `M` / `F` | DB schema |
| `ap_hi` (SBP) | int | 70–250 mmHg | Clinical |
| `ap_lo` (DBP) | int | 40–150 mmHg | Clinical |
| `cholesterol` | enum | 1 (normal), 2 (above), 3 (well above) | Dataset |
| `gluc` | enum | 1 (normal), 2 (above), 3 (well above) | Dataset |
| `smoke` | bool | 0/1 | Dataset |
| `alco` | bool | 0/1 | Dataset |
| `active` | bool | 0/1 | Dataset |
| `height` | float | 100–250 cm | Clinical |
| `weight` | float | 30–200 kg | Clinical |
| `name` | str | 1–100 chars | General |

**Cross-field rule**: `ap_hi > ap_lo` (systolic > diastolic) — enforced via `@model_validator(mode='after')`.

---

### 3. FastAPI Endpoints

| Approach | Description | Pros | Cons | Effort |
|----------|-------------|------|------|--------|
| **A: Raw SQLAlchemy Core** (like reference) | Direct `conn.execute()` | Simple, fast | No ORM benefits, manual mapping | Low |
| **B: SQLAlchemy ORM + Session** | `session.add()`, `session.commit()` | Relationships, type safety, testable | Slight overhead | Medium |
| **C: ORM + Dependency Injection** | `get_db` dependency, `PatientRepository` | Testable, decoupled, follows FastAPI patterns | More structure | Medium |

**Recommendation: C** — Use `AsyncSession` with dependency injection (`get_db`), repository pattern for testability. Required for strict TDD and integration testing with `httpx`.

**Endpoints:**
- `POST /pacientes` — Create patient + classification data (returns 201 + `Location` header)
- `GET /pacientes/{id}` — Retrieve patient with classifications (1:N)
- `GET /pacientes` — List with pagination (future)

**Error handling:** Custom exception handlers for `RequestValidationError` → return structured JSON + 422, compatible with HTMX `hx-target` error rendering.

---

### 4. Frontend: Jinja2 + Bootstrap + HTMX

| Approach | Description | Pros | Cons | Effort |
|----------|-------------|------|------|--------|
| **A: Full page reloads** | Traditional form POST → redirect | Simple, no JS | Poor UX, no progressive enhancement | Low |
| **B: HTMX for form submission** | `hx-post`, `hx-target`, `hx-swap` | Partial updates, validation errors inline, no full reload | Learning curve | Medium |
| **C: Alpine.js + HTMX** | Reactive UI for conditional fields | Rich interactions | Adds dependency | High |

**Recommendation: B** — HTMX only. Matches "JS mínimo" requirement. Form submits to `POST /pacientes`, returns partial HTML (success toast or validation errors). Bootstrap 5 via CDN per config.

**Template structure:**
- `base.html` — Bootstrap CDN, HTMX CDN, common layout
- `patients/form.html` — Clinical form with `hx-post="/pacientes"`, `hx-target="#form-result"`
- `patients/detail.html` — Read-only view with classification history
- `partials/_errors.html` — Reusable validation error display
- `partials/_toast.html` — Success/error notifications

---

### 5. Database: SQLAlchemy ORM + Migrations

| Approach | Description | Pros | Cons | Effort |
|----------|-------------|------|------|--------|
| **A: `meta.create_all()`** (reference) | Auto-create on startup | Zero config | No versioning, dangerous in prod | Low |
| **B: Alembic migrations** | Versioned schema changes | Production-safe, reviewable | Extra setup, learning curve | Medium |
| **C: SQLAlchemy ORM + Alembic** | ORM models + generated migrations | Best of both | Most setup | Medium |

**Recommendation: C** — Define ORM models (`Patient`, `Classification`) with relationships, use Alembic for migrations. SQLite for dev, but migration strategy must work for future PostgreSQL.

**Models (matching fixed schema):**
```python
class Patient(Base):
    __tablename__ = "patients"
    id = Column(Integer, primary_key=True)
    name = Column(String(100), nullable=False)
    gender = Column(String(1), nullable=False)  # 'M' | 'F'
    classifications = relationship("Classification", back_populates="patient")

class Classification(Base):
    __tablename__ = "classifications"
    id = Column(Integer, primary_key=True)
    patient_id = Column(Integer, ForeignKey("patients.id"), nullable=False)
    age = Column(Integer, nullable=False)
    ap_hi = Column(Integer, nullable=False)
    ap_lo = Column(Integer, nullable=False)
    cholesterol = Column(Integer, nullable=False)  # 1,2,3
    gluc = Column(Integer, nullable=False)  # 1,2,3
    smoke = Column(Boolean, nullable=False)
    alco = Column(Boolean, nullable=False)
    active = Column(Boolean, nullable=False)
    height = Column(Float, nullable=False)
    weight = Column(Float, nullable=False)
    classification_result = Column(String(50))  # set by ML module
    risk_score = Column(Float)
    classification_date = Column(DateTime, default=datetime.utcnow)
    patient = relationship("Patient", back_populates="classifications")
```

---

### 6. Integration with Classification Module

| Approach | Description | Pros | Cons | Effort |
|----------|-------------|------|------|--------|
| **A: Synchronous call in endpoint** | `POST /pacientes` → create patient → call ML → save classification | Simple | Blocks request, couples modules | Low |
| **B: Background task** | `POST /pacientes` → create patient → return 202 → background task classifies | Non-blocking | Complexity, need status polling | Medium |
| **C: Separate endpoint** | `POST /pacientes` → patient only; `POST /clasificaciones` → classification | Decoupled, RESTful | Two requests from frontend | Medium |

**Recommendation: C** — Separate concerns. Patient ingestion module only handles patient + raw clinical data storage. Classification module (separate change) reads patient data, runs inference, writes to `classifications`. This respects the 4-module architecture (ingreso, clasificación, almacenamiento/historial, reporte).

**Link:** `classification.patient_id` → `patient.id` (1:N). Frontend calls classification endpoint after patient creation, or classification runs async via background worker.

---

## Recommendation Summary

| Area | Decision |
|------|----------|
| Pydantic | Approach C: `Annotated` + `Field` + `@model_validator` |
| Clinical validation | Enum fields for categorical, range constraints, cross-field `ap_hi > ap_lo` |
| FastAPI | Approach C: ORM + DI + Repository pattern |
| Frontend | HTMX + Bootstrap 5 CDN, partial form submission |
| Database | SQLAlchemy ORM + Alembic migrations |
| Classification integration | Separate endpoint/module (decoupled) |

---

## Risks

1. **Schema drift** — Fixed DB schema per AGENTS.md; any model change requires Alembic migration + team approval
2. **Clinical validation gaps** — Missing cross-field rules (e.g., BMI from height/weight) could allow invalid data
3. **HTMX error rendering** — 422 validation errors must return HTML partials, not JSON, for seamless UX
4. **SQLite vs PostgreSQL** — SQLite doesn't enforce all constraints; CI must test against both
5. **Terminology enforcement** — Code review must catch "predicción"/"predecir" — consider adding a linter rule

---

## Decisions Needed

1. **Project package name** — `src/app` vs `src/cardio_risk` vs `src/semillero`? (affects import paths)
2. **Patient name field** — Required per schema, but is it PII? Should we hash/pseudonymize?
3. **Classification trigger** — Auto-classify on patient creation (background) or explicit clinician action?
4. **Pagination** — `GET /pacientes` needs pagination design (offset vs cursor)
5. **HTMX vs Alpine** — Confirm HTMX-only is sufficient for conditional field visibility (e.g., pediatric vs adult ranges)

---

## Ready for Proposal

**Yes.** The exploration is complete. The orchestrator should:
1. Confirm the project package name
2. Clarify classification trigger (auto vs manual)
3. Proceed to **Propose** phase with the above recommendations as the baseline approach

The next step is to create an SDD proposal for "Patient Data Ingestion Module" covering schemas, endpoints, templates, and database models.