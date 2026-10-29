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
    
    df = load_data(data_path)
    pipelines = build_pipelines()
    cv_results = cross_validate_pipelines(df, cv_folds)
    best_pipeline, best_name = select_and_fit(pipelines, cv_results, df)
    print_metrics_table(cv_results, best_name)
    save_artifacts(best_pipeline, model_path, report_path, cv_results, best_name)

def parse_args() -> argparse.Namespace:
    
    parser = argparse.ArgumentParser(
        description="Entrenamiento del modelo con validación cruzada estratificada"
    )
    parser.add_argument("--data-path", type=Path, default=DEFAULT_DATA_PATH)
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--report-path", type=Path, default=DEFAULT_REPORT_PATH)
    parser.add_argument("--cv-folds", type=int, default=5)
    return parser.parse_args()

def main() -> None:
    
    args = parse_args()
    train(
        data_path=args.data_path,
        model_path=args.model_path,
        report_path=args.report_path,
        cv_folds=args.cv_folds,
    )

if __name__ == "__main__":
    main()
