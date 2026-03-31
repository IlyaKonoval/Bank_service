import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from pipeline.evaluation import ModelEvaluator


@pytest.fixture
def trained_model():
    np.random.seed(42)
    n = 300
    X = np.random.randn(n, 5)
    y = (X[:, 0] + X[:, 1] > 0).astype(int)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = LogisticRegression(max_iter=200, random_state=42)
    model.fit(X_scaled[:200], y[:200])

    return model, X_scaled[200:], y[200:]


@pytest.fixture
def evaluator(tmp_path):
    return ModelEvaluator(output_dir=str(tmp_path))


class TestModelEvaluator:

    def test_full_report_returns_metrics(self, trained_model, evaluator):
        model, X_test, y_test = trained_model
        metrics = evaluator.full_report(model, X_test, y_test, model_name="test_lr")
        assert "accuracy" in metrics
        assert "precision" in metrics
        assert "recall" in metrics
        assert "f1" in metrics
        assert "roc_auc" in metrics
        assert "pr_auc" in metrics
        assert 0 <= metrics["accuracy"] <= 1

    def test_full_report_saves_files(self, trained_model, evaluator):
        model, X_test, y_test = trained_model
        evaluator.full_report(model, X_test, y_test, model_name="test_lr")

        output_dir = Path(evaluator.output_dir)
        assert (output_dir / "test_lr_metrics.json").exists()
        assert (output_dir / "test_lr_roc_curve.png").exists()
        assert (output_dir / "test_lr_pr_curve.png").exists()
        assert (output_dir / "test_lr_confusion_matrix.png").exists()

    def test_metrics_json_valid(self, trained_model, evaluator):
        model, X_test, y_test = trained_model
        evaluator.full_report(model, X_test, y_test, model_name="test_lr")

        with open(Path(evaluator.output_dir) / "test_lr_metrics.json") as f:
            data = json.load(f)
        assert data["model_name"] == "test_lr"
        assert isinstance(data["f1"], float)

    def test_compare_models(self, evaluator):
        results = {
            "ModelA": {"f1": 0.8, "roc_auc": 0.85},
            "ModelB": {"f1": 0.75, "roc_auc": 0.9},
            "ModelC": {"f1": 0.6, "roc_auc": 0.7},
        }
        df = evaluator.compare_models(results)
        assert isinstance(df, pd.DataFrame)
        assert df.iloc[0]["model"] == "ModelA"
        assert (Path(evaluator.output_dir) / "model_comparison.png").exists()

    def test_find_optimal_threshold(self, trained_model, evaluator):
        model, X_test, y_test = trained_model
        threshold = evaluator.find_optimal_threshold(model, X_test, y_test)
        assert 0.1 <= threshold <= 0.9
