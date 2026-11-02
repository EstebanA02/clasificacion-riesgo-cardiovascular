"""Pruebas del proceso de entrenamiento (Task 1.4).

Cubren: generacion de artefactos, estructura del joblib
{"pipeline", "threshold"}, contenido del reporte JSON, criterio de
seleccion por precision a recall >= MIN_RECALL, y filtros de load_data.

Los tests que requieren entrenar corren train() a rutas temporales con
cv_folds=3 (marcados slow, ~30s) y NUNCA tocan models/model.joblib real.
Los tests de MLService.classify y lifespan se escriben en Tasks 1.5 y 1.6.
"""

import json
import sys
import warnings
from pathlib import Path

import joblib
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from train import DEFAULT_DATA_PATH, FEATURE_ORDER, MIN_RECALL, load_data, train

MODEL_KEYS = {"pipeline", "threshold"}
METRICS = ("accuracy", "precision", "recall", "f1")

@pytest.fixture(scope="module")
def trained(tmp_path_factory):
    """Ejecuta train() a rutas temporales (cv_folds=3) una vez por modulo."""
    out = tmp_path_factory.mktemp("ml_artifacts")
    model_path = out / "model.joblib"
    report_path = out / "train_report.json"
    train(data_path=DEFAULT_DATA_PATH, model_path=model_path,
          report_path=report_path, cv_folds=3)
    return model_path, report_path

def test_load_data_filters(tmp_path):
    """load_data descarta filas fuera de rango, no incluye 'id' y age queda en anios."""
    base = dict(gender=1, height=170, weight=70, ap_hi=120, ap_lo=80,
                cholesterol=1, gluc=1, smoke=0, alco=0, active=1, cardio=0)
    sin_gender = {k: v for k, v in base.items() if k != "gender"}
    sin_ap_hi = {k: v for k, v in base.items() if k != "ap_hi"}
    sin_weight = {k: v for k, v in base.items() if k != "weight"}
    sin_presiones = {k: v for k, v in base.items()
                     if k not in ("ap_hi", "ap_lo")}
    filas = [
        dict(id=1, age=35 * 365, **base),                 # valida
        dict(id=2, age=42 * 365, gender=2, **sin_gender), # valida
        dict(id=3, age=50 * 365, ap_hi=12000, **sin_ap_hi),   # ap_hi fuera de rango
        dict(id=4, age=31 * 365, weight=10, **sin_weight),    # weight fuera de rango
        dict(id=5, age=33 * 365, ap_hi=75, ap_lo=90, **sin_presiones),  # ap_hi <= ap_lo
    ]
    csv_path = tmp_path / "mini.csv"
    pd.DataFrame(filas).to_csv(csv_path, sep=";", index=False)

    resultado = load_data(csv_path)

    assert "id" not in resultado.columns
    assert list(resultado.columns) == FEATURE_ORDER + ["cardio"]
    assert len(resultado) == 2
    assert set(resultado["age"]) == {35, 42}  # dias -> anios

@pytest.mark.slow
def test_train_generates_artifacts(trained):
    """train() genera model.joblib y train_report.json en la ruta temporal."""
    model_path, report_path = trained
    assert model_path.exists()
    assert report_path.exists()

@pytest.mark.slow
def test_artifact_structure(trained):
    """El joblib es {"pipeline": Pipeline predict_proba, "threshold": float en (0,1)}."""
    model_path, _ = trained
    artifact = joblib.load(model_path)
    assert set(artifact.keys()) == MODEL_KEYS
    assert hasattr(artifact["pipeline"], "predict_proba")
    threshold = artifact["threshold"]
    assert isinstance(threshold, float)
    assert 0.0 < threshold < 1.0

@pytest.mark.slow
def test_report_content(trained):
    """El reporte tiene selected_model, min_recall, meets_min_recall y metrics completas."""
    _, report_path = trained
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert isinstance(report["selected_model"], str)
    assert report["min_recall"] == MIN_RECALL
    assert isinstance(report["meets_min_recall"], bool)
    assert set(report["metrics"].keys()) == {
        "logistic_regression", "random_forest", "xgboost",
    }
    for model_metrics in report["metrics"].values():
        for metric in METRICS:
            assert 0.0 <= model_metrics[metric]["mean"] <= 1.0
            assert 0.0 <= model_metrics[metric]["std"] <= 1.0
        assert isinstance(model_metrics["threshold"], float)
        assert 0.0 < model_metrics["threshold"] < 1.0
        assert isinstance(model_metrics["precision_at_min_recall"], float)
        assert isinstance(model_metrics["recall_at_threshold"], float)

@pytest.mark.slow
def test_selected_model_is_best_precision(trained):
    """El seleccionado es el de mayor precision_at_min_recall (recall como restriccion)."""
    _, report_path = trained
    report = json.loads(report_path.read_text(encoding="utf-8"))
    metrics = report["metrics"]
    mejor = max(metrics, key=lambda n: metrics[n]["precision_at_min_recall"])
    assert report["selected_model"] == mejor

@pytest.mark.slow
def test_recall_at_threshold(trained):
    """recall_at_threshold del ganador debe ser >= MIN_RECALL; si no, advertencia (no fallo)."""
    _, report_path = trained
    report = json.loads(report_path.read_text(encoding="utf-8"))
    winner = report["metrics"][report["selected_model"]]
    recall = winner["recall_at_threshold"]
    if recall < report["min_recall"]:
        warnings.warn(
            f"recall_at_threshold del ganador ({recall:.4f}) < MIN_RECALL "
            f"({report['min_recall']}) — el modelo no alcanza el minimo declarado"
        )
