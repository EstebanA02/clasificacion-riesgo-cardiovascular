## Proyecto
Sistema de software (FastAPI + ML) para clasificación de riesgo cardiovascular. Proyecto de semillero de investigación UNIMINUTO. El modelo clasifica similitud de perfil con casos diagnosticados, NO predice enfermedad futura.

## Stack
- Backend: Python, FastAPI, Pydantic, SQLAlchemy, SQLite
- Frontend: HTML, CSS, Bootstrap (CDN), JS mínimo, Jinja2
- ML: scikit-learn, pandas, joblib, xgboost
- Testing: pytest, httpx
- Gestor de paquetes: uv (usar `uv run`, `uv add`, nunca pip)

## Estructura de carpetas
- src/<paquete>/
- api/ # routers FastAPI
- models/ # modelos SQLAlchemy
- schemas/ # esquemas Pydantic
- ml/ # carga del modelo serializado, inferencia
- templates/ # Jinja2
- data/ # dataset (no versionar el CSV grande)
- notebooks/ # EDA y entrenamiento
- models/ # modelo .joblib serializado
- tests/


## Esquema de base de datos (NO cambiar nombres sin confirmar)

**Tabla `patients`**
- id (PK)
- name
- gender

**Tabla `classifications`** (1:N con patients)
- id (PK)
- patient_id (FK -> patients.id)
- age
- ap_hi, ap_lo
- cholesterol, gluc
- smoke, alco, active
- height, weight
- classification_result
- risk_score
- classification_date

## Terminología obligatoria
- "clasificación" / "clasificar" — NUNCA "predicción" / "predecir"
- Tabla y componente: `classifications`, no `predictions`
- Campo: `risk_score`, no `prediction_score`
- El componente del modelo se llama "Modelo de clasificación"

## Algoritmos y métrica
Comparar: Regresión Logística, Random Forest, XGBoost.
Métrica de selección: **recall** (prioridad sobre accuracy/precisión/F1, por costo clínico de falsos negativos).

## Arquitectura
Cliente-servidor en 3 capas: Frontend (Jinja2/Bootstrap) → Backend (FastAPI, valida con Pydantic) → Modelo de clasificación (joblib, cargado una vez en memoria) → SQLite (SQLAlchemy ORM).

## Actor y módulos
Único actor: profesional de salud.
4 módulos: ingreso de datos, clasificación, almacenamiento/historial, reporte/visualización.

## Comandos
- Instalar deps: `uv sync`
- Correr backend: `uv run uvicorn <paquete>.main:app --reload`
- Tests: `uv run pytest`
- Notebook: `uv run jupyter lab`

## Reglas de trabajo
- No introducir nuevas tablas, campos o nombres sin avisar antes.
- No agregar dependencias fuera del stack definido sin confirmar.
- Mantener docstrings/comentarios en español.
- Todo cambio de esquema o arquitectura debe quedar justificable frente al documento del proyecto (ya aprobado/en revisión por el director).

## Git / GitHub
- Commits en formato convencional (feat/fix/docs/test/chore)
- Trabajo en ramas por feature, merge a main vía PR
- Cada tarea del Kanban (GitHub Projects) debe vincularse a un Issue
- CI en GitHub Actions corre `uv sync` + `uv run pytest` en cada push/PR
- No romper CI: si un PR falla pytest, no se mergea