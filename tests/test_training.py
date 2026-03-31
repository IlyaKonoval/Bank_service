import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

from pipeline.training import (
    ModelTrainer,
    get_base_models,
    build_stacking_ensemble,
    build_voting_ensemble,
)


@pytest.fixture
def train_val_data():
    import numpy as np
    np.random.seed(42)
    n = 200

    X = pd.DataFrame({
        "AGE": np.random.randint(20, 65, n),
        "GENDER": np.random.randint(0, 2, n),
        "CHILD_TOTAL": np.random.randint(0, 5, n),
        "DEPENDANTS": np.random.randint(0, 4, n),
        "PERSONAL_INCOME": np.random.uniform(10000, 100000, n),
        "SOCSTATUS_WORK_FL": np.random.randint(0, 2, n),
        "SOCSTATUS_PENS_FL": np.random.randint(0, 2, n),
        "LOAN_NUM_TOTAL": np.random.randint(1, 8, n),
        "LOAN_NUM_CLOSED": np.random.uniform(0, 5, n).round(),
    })
    X["LOAN_NUM_CLOSED"] = X[["LOAN_NUM_CLOSED", "LOAN_NUM_TOTAL"]].min(axis=1)

    y = pd.Series(np.random.binomial(1, 0.15, n))

    split = int(n * 0.75)
    return X[:split], X[split:], y[:split], y[split:]


class TestGetBaseModels:

    def test_returns_six_models(self):
        models = get_base_models()
        assert len(models) == 6

    def test_all_have_fit_and_predict(self):
        models = get_base_models()
        for name, model in models.items():
            assert hasattr(model, "fit")
            assert hasattr(model, "predict")
            assert hasattr(model, "predict_proba")


class TestModelTrainer:

    def test_benchmark_returns_results(self, train_val_data):
        X_train, X_val, y_train, y_val = train_val_data
        trainer = ModelTrainer(use_smote=True)
        results = trainer.benchmark(X_train, y_train, X_val, y_val)
        assert len(results) == 6
        for name, data in results.items():
            assert "model" in data
            assert "f1" in data
            assert "roc_auc" in data
            assert 0 <= data["roc_auc"] <= 1

    def test_benchmark_without_smote(self, train_val_data):
        X_train, X_val, y_train, y_val = train_val_data
        trainer = ModelTrainer(use_smote=False)
        results = trainer.benchmark(X_train, y_train, X_val, y_val)
        assert len(results) == 6

    def test_feature_engineer_is_fitted_after_benchmark(self, train_val_data):
        X_train, X_val, y_train, y_val = train_val_data
        trainer = ModelTrainer(use_smote=True)
        trainer.benchmark(X_train, y_train, X_val, y_val)
        assert trainer.feature_engineer._is_fitted


class TestEnsembles:

    def test_stacking_builds(self):
        models = {
            "lr": LogisticRegression(max_iter=100),
            "lr2": LogisticRegression(max_iter=200),
        }
        ensemble = build_stacking_ensemble(models)
        assert hasattr(ensemble, "fit")

    def test_voting_builds(self):
        models = {
            "lr": LogisticRegression(max_iter=100),
            "lr2": LogisticRegression(max_iter=200),
        }
        ensemble = build_voting_ensemble(models)
        assert hasattr(ensemble, "fit")
