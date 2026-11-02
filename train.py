"""Script de entrenamiento del modelo de clasificación de riesgo cardiovascular.

Proyecto: semillero UNIMINUTO — clasificación por similitud de perfil,
con validación cruzada estratificada, ajuste de umbral de decisión (plan B del
documento académico) y selección por precisión a recall >= MIN_RECALL.

Artefactos generados:
- models/model.joblib: dict {"pipeline": <Pipeline reentrenado con el 100% de
  los datos>, "threshold": <umbral ajustado>}
- models/train_report.json: métricas de CV a umbral 0.5 (media ± desviación)
  más threshold, precision_at_min_recall y recall_at_threshold por modelo

Uso:
    uv run python train.py
    uv run python train.py --cv-folds 10
"""

import argparse
import json
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_curve
from sklearn.model_selection import StratifiedKFold, cross_validate, cross_val_predict
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
    """Carga y limpia el dataset para entrenamiento.

    Lee el CSV (separador ';'), descarta 'id', convierte age de dias a anios
    y filtra filas fisiologicamente imposibles (hallazgos EDA: ap_hi entre
    -150 y 16020, height min 55, weight min 10; ~1.98% del dataset se descarta).

    Contrato: devuelve las 11 features en FEATURE_ORDER + target 'cardio'.
    El dataset usa gender numerico (1=Mujer, 2=Hombre); la conversion a
    'M'/'F' vive unicamente en MLService.classify().
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
    f"Filas descartadas por filtros fisiologicos: {descartadas} ({porcentaje:.2f}%)")
    
    return df

def build_pipelines() -> dict[str, Pipeline]:
    """Crea los 3 pipelines candidatos para comparacion.

    Solo logistic_regression lleva StandardScaler: LR ajusta coeficientes con
    descenso de gradiente y sufre el rango dispar de escalas de las features;
    los arboles dividen por umbrales y son invariantes a escala.
    Todos con random_state=42 para reproducibilidad.
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

def threshold_for_recall(y_true, probs) -> tuple[float, float, float]:
    """Devuelve (threshold, precision, recall) del umbral mas alto con recall >= MIN_RECALL.

    Alinea precision y recall con los thresholds descartando el ultimo
    elemento (precision_recall_curve entrega precision=1 y recall=0 finales
    que no corresponden a ningun umbral). Si ningun punto alcanza
    MIN_RECALL: warnings.warn y se usa el threshold de mayor recall, sin
    lanzar error.
    """
    precision, recall, thresholds = precision_recall_curve(y_true, probs)
    precision = precision[:-1]
    recall = recall[:-1]

    pos = np.flatnonzero(recall >= MIN_RECALL)
    if pos.size:
        index = int(pos[np.argmax(thresholds[pos])])
    else:
        index = int(np.argmax(recall))
        warnings.warn(
            f"Ningun umbral alcanza recall >= {MIN_RECALL}; se usa el umbral "
            f"de mayor recall ({recall[index]:.4f}).",
            UserWarning,
        )

    return float(thresholds[index]), float(precision[index]), float(recall[index])

def cross_validate_pipelines(
    df: pd.DataFrame, cv_folds: int,
) -> dict[str, dict]:
    """Validacion cruzada estratificada de los 3 pipelines + ajuste de umbral.

    StratifiedKFold conserva la proporcion del target en cada fold. Las 4
    metricas de SCORING se reportan a umbral 0.5 (media +/- desviacion);
    scoring debe ser lista, no set. Ademas, con probabilidades
    OUT-OF-FOLD (cross_val_predict sobre el mismo cv) se calcula por
    modelo: threshold (plan B), precision_at_min_recall y
    recall_at_threshold.
    Devuelve {modelo: {metrica: {"mean": float, "std": float},
    "threshold": float, "precision_at_min_recall": float,
    "recall_at_threshold": float}}.
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

        
        probs_oof = cross_val_predict(pipeline, X, y, cv=cv,
                                      method="predict_proba")[:, 1]
        threshold, precision, recall = threshold_for_recall(y, probs_oof)
        model_metrics['threshold'] = threshold
        model_metrics['precision_at_min_recall'] = precision
        model_metrics['recall_at_threshold'] = recall

        cv_results[name] = model_metrics

    return cv_results
    
    

def select_and_fit(
    pipelines: dict[str, Pipeline],
    cv_results: dict,
    df: pd.DataFrame,
) -> tuple[Pipeline, str, float]:
    """Selecciona el ganador por mayor precision a recall >= MIN_RECALL.

    Criterio (plan B): recall como restriccion, precision como desempate.
    La CV solo SELECCIONA: los fold-models vieron ~80% de los datos. El
    pipeline se reentrena aqui con el 100% del dataset; el threshold,
    calculado SOLO con probabilidades out-of-fold, se devuelve sin
    recalcular con el modelo reentrenado.
    Devuelve (pipeline_ajustado, nombre_ganador, threshold).
    """
    best_name = max(
        cv_results, key=lambda n: cv_results[n]['precision_at_min_recall']
    )
    best_pipeline = pipelines[best_name]

    X = df[FEATURE_ORDER]
    y = df['cardio']

    best_pipeline.fit(X, y)

    threshold = float(cv_results[best_name]['threshold'])

    return best_pipeline, best_name, threshold
    

def save_artifacts(
    best_pipeline: Pipeline,
    threshold: float,
    model_path: Path,
    report_path: Path,
    cv_results: dict,
    best_name: str,
) -> None:
    """Guarda el ganador (joblib) y el reporte academico (JSON).

    En model_path se guarda un dict {"pipeline": <Pipeline>,
    "threshold": float} con el pipeline reentrenado y su umbral ajustado.
    El reporte incluye selected_model, metrics de CV a umbral 0.5,
    threshold por modelo, min_recall y meets_min_recall (con
    recall_at_threshold). Crea los directorios destino si no existen.
    Si el ganador no alcanza MIN_RECALL: warnings.warn con valor y brecha,
    sin interrumpir el guardado.
    """
    Path(model_path).parent.mkdir(parents=True, exist_ok=True)
    Path(report_path).parent.mkdir(parents=True, exist_ok=True)

    joblib.dump({"pipeline": best_pipeline, "threshold": threshold}, model_path)

    winning_recall = cv_results[best_name]['recall_at_threshold']
    meets_min_recall = bool(winning_recall >= MIN_RECALL)

    if not meets_min_recall:
        gap = MIN_RECALL - winning_recall

        warnings.warn(
            f"ALERTA DE CALIDAD: El recall del modelo ganador "
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
        for metric in SCORING:  
            valores = metrics[metric]
            print(f"{name:<{ANCHO}} {metric:<10} {valores['mean']:7.4f} "
                  f"{valores['std']:7.4f}{marca}")
            marca = ""
        print()

    
    print(f"{'Modelo':<{ANCHO}} {'Umbral':>7} {'P@0.85':>7} {'R@th':>7}")
    print("-" * (ANCHO + 30))
    for name, metrics in cv_results.items():
        marca = "  << SELECCIONADO" if name == best_name else ""
        print(f"{name:<{ANCHO}} {metrics['threshold']:7.4f} "
              f"{metrics['precision_at_min_recall']:7.4f} "
              f"{metrics['recall_at_threshold']:7.4f}{marca}")
    print()

def train(
    data_path: Path = DEFAULT_DATA_PATH,
    model_path: Path = DEFAULT_MODEL_PATH,
    report_path: Path = DEFAULT_REPORT_PATH,
    cv_folds: int = 5,
) -> None:
    """Orquesta el proceso completo: carga, CV + ajuste de umbral,
    seleccion, tabla y guardado."""
    df = load_data(data_path)
    pipelines = build_pipelines()
    cv_results = cross_validate_pipelines(df, cv_folds)
    best_pipeline, best_name, threshold = select_and_fit(pipelines, cv_results, df)
    print_metrics_table(cv_results, best_name)
    save_artifacts(best_pipeline, threshold, model_path, report_path,
                   cv_results, best_name)

def parse_args() -> argparse.Namespace:
    """Configura los argumentos CLI del script (parte del scaffold)."""
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
