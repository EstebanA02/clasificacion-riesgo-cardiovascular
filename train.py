"""Script de entrenamiento del modelo de clasificación de riesgo cardiovascular.

Proyecto: semillero UNIMINUTO — clasificación por similitud de perfil,
con validación cruzada estratificada y selección por recall medio.

Artefactos generados:
- models/model.joblib: pipeline ganador reentrenado con el 100% de los datos
- models/train_report.json: métricas de validación cruzada + modelo seleccionado

Uso:
    uv run python train.py
    uv run python train.py --cv-folds 10
"""

import argparse
import json
import warnings
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

# Orden de columnas que espera MLService al inferir
FEATURE_ORDER = [
    "age", "gender", "height", "weight", "ap_hi", "ap_lo",
    "cholesterol", "gluc", "smoke", "alco", "active",
]

# Rutas por defecto de los artefactos
DEFAULT_DATA_PATH = Path("data/cardio_train.csv")
DEFAULT_MODEL_PATH = Path("models/model.joblib")
DEFAULT_REPORT_PATH = Path("models/train_report.json")

# Métrica de selección y mínimo aceptable
SCORING = ["accuracy", "precision", "recall", "f1"]
MIN_RECALL = 0.85
RANDOM_STATE = 42

def load_data(data_path: Path = DEFAULT_DATA_PATH) -> pd.DataFrame:
    """BLOQUE 1 y 2 — Carga y limpieza del dataset.

    Debe devolver un DataFrame limpio, con columnas en FEATURE_ORDER.

    Que escribirás tú:
    1. pd.read_csv con sep=';' (el dataset usa punto y coma como separador)
    2. Descartar columna 'id' (es solo un índice, no una feature clínica; si
       se deja, el modelo aprendería ruido)
    3. Convertir age de días a años (age // 365); el rango real es 29-64
    4. Descartar filas fisiológicamente imposibles (hallazgos del EDA):
       - ap_hi fuera de [70, 250]   (en el dataset: min -150, max 16020,
         errores de captura)
       - ap_lo fuera de [40, 200]
       - ap_hi <= ap_lo (la sistólica debe superar la diastólica)
       - height fuera de [120, 220] (en el dataset: min 55, error de captura)
       - weight fuera de [30, 200]  (en el dataset: min 10, error de captura)
       Total esperado: ~1.98% del dataset. Esas filas NUNCA llegan al
       entrenamiento: se filtran ANTES de la validación cruzada.

    Contrato: el DataFrame devuelto conserva las 11 features (columnas en
    FEATURE_ORDER) y la columna target 'cardio'.

    Atención: el dataset usa gender numérico (1=Mujer, 2=Hombre).
    NO conviertas a 'M'/'F' aquí — ese trabajo vive SOLO en MLService.classify().
    """
    
    df = pd.read_csv(data_path, sep=';')
    df = df.drop(columns=['id'])
    df['age'] = df['age']//365
    
    rows_before = len(df)
    
    df_filtered = df[
        (df["ap_hi"] >= 70)
        & (df["ap_hi"] <= 250)
        & (df["ap_lo"] >= 40)
        & (df["ap_lo"] <= 200)
        & (df["ap_hi"] > df["ap_lo"])
        & (df["height"] >= 120)
        & (df["height"] <= 220)
        & (df["weight"] >= 30)
        & (df["weight"] <= 200)
    ]
      
    required_columns = FEATURE_ORDER + ['cardio']
    df = df_filtered[required_columns].copy()
    
    descartadas = rows_before - len(df)
    porcentaje = descartadas / rows_before * 100
    print(
    f"Filas descartadas por filtros fisiológicos: {descartadas} ({porcentaje:.2f}%)")
    
    return df

def build_pipelines() -> dict[str, Pipeline]:
    """BLOQUE 3 — Crea los 3 pipelines a comparar.

    Que escribirás tú:
    - logistic_regression: StandardScaler + LogisticRegression
    - random_forest: RandomForestClassifier
    - xgboost: XGBClassifier (+ StandardScaler si lo justificamos juntos:
      los árboles no siempre necesitan escalar)

    Fija random_state=42 en cada modelo donde aplique.
    """
    
    pipelines = {
        'logistic_regression': Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', LogisticRegression(random_state=RANDOM_STATE))            
        ]),
        
        'random_forest': Pipeline([
            ('classifier', RandomForestClassifier(random_state=RANDOM_STATE))
        ]),
        
        'xgboost': Pipeline([
            ('classifier', XGBClassifier(random_state=RANDOM_STATE))
        ])
    }
    
    return pipelines

def cross_validate_pipelines(
    df: pd.DataFrame, cv_folds: int,
) -> dict[str, dict[str, dict[str, float]]]:
    """BLOQUE 4 — Validación cruzada estratificada.

    Que escribirás tú:
    1. Separar X (features en FEATURE_ORDER) de y (columna target 'cardio'),
       usando el df limpio que devuelve load_data
    2. StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=RANDOM_STATE)
    3. Por cada pipeline: cross_validate(..., scoring=SCORING)
    4. Devolver dict {modelo: {métrica: {"mean": ..., "std": ...}}}

    ¿Por qué estratificado? — Para que cada fold conserve la misma proporción
    de casos con/sin diagnóstico (balance de clases) que el dataset completo.
    """
    
    pipelines = build_pipelines()
    
    X = df[FEATURE_ORDER]
    y = df['cardio']
    
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=RANDOM_STATE)
    
    cv_results = {}
    
    for name, pipeline in pipelines.items():
        cv_scores = cross_validate(pipeline, X, y, cv=cv, scoring=SCORING)
        
        model_metrics = {}
        for metric in SCORING:
            scores_array = cv_scores[f'test_{metric}']
            
            model_metrics[metric] = {
            'mean': float(scores_array.mean()),
            'std': float(scores_array.std()),
            }
    
        cv_results[name] = model_metrics
    
    return cv_results
    
    

def select_and_fit(
    pipelines: dict[str, Pipeline],
    cv_results: dict,
    df: pd.DataFrame,
) -> Pipeline:
    """BLOQUE 5 — Selección y reentrenamiento.

    Que escribirás tú:
    1. Elegir el pipeline con mayor recall medio (mean) de cv_results
    2. Reentrenar (fit) ese pipeline con el 100% del dataset
    3. Devolver el pipeline ya ajustado
    """
    best_name = max(cv_results, key=lambda n: cv_results[n]['recall']['mean'])
    best_pipeline = pipelines[best_name]
    
    X = df[FEATURE_ORDER]
    y = df['cardio']
    
    best_pipeline.fit(X,y)
    
    return best_pipeline, best_name
    

def save_artifacts(
    best_pipeline: Pipeline,
    model_path: Path,
    report_path: Path,
    cv_results: dict,
    best_name: str,
) -> None:
    """BLOQUE 6 — Guardado y reporte.

    Que escribirás tú:
    1. joblib.dump del pipeline ganador en model_path
    2. json.dump del reporte en report_path:
       - métricas por modelo (mean/std)
       - selected_model: nombre del ganador
       - min_recall: MIN_RECALL
       - meets_min_recall: bool
    3. Si el recall medio del ganador < MIN_RECALL:
       advertencia con warnings.warn (NO error) e información de la brecha,
       y continuar guardando de todas formas
    """

    Path(model_path).parent.mkdir(parents=True, exist_ok=True)
    Path(report_path).parent.mkdir(parents=True, exist_ok=True)
    
    joblib.dump(best_pipeline, model_path)
    
    winning_recall = cv_results[best_name]['recall']['mean']
    meets_min_recall = bool(winning_recall >= MIN_RECALL)
    
    if not meets_min_recall:
        gap = MIN_RECALL - winning_recall
        
        warnings.warn(
            f"ALERTA DE CALIDAD: El recall medio del modelo ganador "
            f"('{best_name}' con {winning_recall:.4f}) esta por debajo del minimo "
            f"esperado ({MIN_RECALL}). Brecha: {gap:.4f}. "
            f"Se guardaran los artefactos de todas formas.",
            UserWarning
        )

    report = {
        'selected_model': best_name,
        'min_recall': MIN_RECALL,
        'meets_min_recall': meets_min_recall,
        'metrics': cv_results
    }
    
    with open(report_path, 'w', encoding= 'utf-8') as f:
        json.dump(report, f, indent= 4, ensure_ascii=False)
        
def print_metrics_table(cv_results: dict, best_name: str) -> None:
    """Imprime la tabla de metricas por modelo, marcando el seleccionado.

    Formato ASCII sin tildes: la consola de Windows (cp1252) corrompe
    acentos y simbolos unicode.
    """
    ANCHO = 22
    print()
    print(f"{'Modelo':<{ANCHO}} {'Metrica':<10} {'Media':>7} {'Desv.':>7}")
    print("-" * (ANCHO + 30))
    for name, metrics in cv_results.items():
        marca = "  << SELECCIONADO" if name == best_name else ""
        for metric, valores in metrics.items():
            print(f"{name:<{ANCHO}} {metric:<10} {valores['mean']:7.4f} "
                  f"{valores['std']:7.4f}{marca}")
            marca = ""
        print()

def train(
    data_path: Path = DEFAULT_DATA_PATH,
    model_path: Path = DEFAULT_MODEL_PATH,
    report_path: Path = DEFAULT_REPORT_PATH,
    cv_folds: int = 5,
) -> None:
    """Orquesta el proceso completo (bloque 1 a 6)."""
    df = load_data(data_path)
    pipelines = build_pipelines()
    cv_results = cross_validate_pipelines(df, cv_folds)
    best_pipeline, best_name = select_and_fit(pipelines, cv_results, df)
    print_metrics_table(cv_results, best_name)
    save_artifacts(best_pipeline, model_path, report_path, cv_results, best_name)

def parse_args() -> argparse.Namespace:
    """Configura los argumentos de línea de comandos (parte del scaffold)."""
    parser = argparse.ArgumentParser(
        description="Entrenamiento del modelo con validación cruzada estratificada"
    )
    parser.add_argument("--data-path", type=Path, default=DEFAULT_DATA_PATH)
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--report-path", type=Path, default=DEFAULT_REPORT_PATH)
    parser.add_argument("--cv-folds", type=int, default=5)
    return parser.parse_args()

def main() -> None:
    """Punto de entrada del script."""
    args = parse_args()
    train(
        data_path=args.data_path,
        model_path=args.model_path,
        report_path=args.report_path,
        cv_folds=args.cv_folds,
    )

if __name__ == "__main__":
    main()
