import logging
from typing import Any

import optuna
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.ensemble import (
    GradientBoostingClassifier,
    RandomForestClassifier,
    StackingClassifier,
    VotingClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_score

from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier

from pipeline.features import FeatureEngineer

optuna.logging.set_verbosity(optuna.logging.WARNING)
logger = logging.getLogger(__name__)


def get_base_models() -> dict[str, Any]:
    return {
        "LogisticRegression": LogisticRegression(
            max_iter=1000, random_state=42, class_weight="balanced"
        ),
        "RandomForest": RandomForestClassifier(
            n_estimators=300, max_depth=10, random_state=42, class_weight="balanced",
            n_jobs=-1,
        ),
        "GradientBoosting": GradientBoostingClassifier(
            n_estimators=200, max_depth=5, learning_rate=0.1, random_state=42,
        ),
        "XGBoost": XGBClassifier(
            n_estimators=300, max_depth=6, learning_rate=0.1,
            scale_pos_weight=6, random_state=42, eval_metric="logloss", verbosity=0,
        ),
        "LightGBM": LGBMClassifier(
            n_estimators=300, max_depth=6, learning_rate=0.1,
            is_unbalance=True, random_state=42, verbose=-1,
        ),
        "CatBoost": CatBoostClassifier(
            iterations=300, depth=6, learning_rate=0.1,
            auto_class_weights="Balanced", random_seed=42, verbose=0,
        ),
    }


def _xgb_objective(trial, X, y, cv, use_smote=True):
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 100, 800),
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
        "gamma": trial.suggest_float("gamma", 0.0, 5.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
        "scale_pos_weight": trial.suggest_float("scale_pos_weight", 3, 10),
    }
    model = XGBClassifier(**params, random_state=42, eval_metric="logloss", verbosity=0)
    if use_smote:
        pipeline = ImbPipeline([("smote", SMOTE(random_state=42)), ("model", model)])
    else:
        pipeline = model
    scores = cross_val_score(pipeline, X, y, cv=cv, scoring="f1", n_jobs=-1)
    return scores.mean()


def _lgbm_objective(trial, X, y, cv, use_smote=True):
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 100, 800),
        "max_depth": trial.suggest_int("max_depth", 3, 12),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 20, 150),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "min_child_samples": trial.suggest_int("min_child_samples", 5, 50),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
    }
    model = LGBMClassifier(**params, is_unbalance=True, random_state=42, verbose=-1)
    if use_smote:
        pipeline = ImbPipeline([("smote", SMOTE(random_state=42)), ("model", model)])
    else:
        pipeline = model
    scores = cross_val_score(pipeline, X, y, cv=cv, scoring="f1", n_jobs=-1)
    return scores.mean()


def _catboost_objective(trial, X, y, cv, use_smote=True):
    params = {
        "iterations": trial.suggest_int("iterations", 100, 800),
        "depth": trial.suggest_int("depth", 3, 10),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1e-8, 10.0, log=True),
        "bagging_temperature": trial.suggest_float("bagging_temperature", 0.0, 10.0),
        "random_strength": trial.suggest_float("random_strength", 0.0, 10.0),
    }
    model = CatBoostClassifier(
        **params, auto_class_weights="Balanced", random_seed=42, verbose=0,
    )
    if use_smote:
        pipeline = ImbPipeline([("smote", SMOTE(random_state=42)), ("model", model)])
    else:
        pipeline = model
    scores = cross_val_score(pipeline, X, y, cv=cv, scoring="f1", n_jobs=-1)
    return scores.mean()


OPTUNA_OBJECTIVES = {
    "XGBoost": _xgb_objective,
    "LightGBM": _lgbm_objective,
    "CatBoost": _catboost_objective,
}


def optimize_model(
    model_name: str, X, y, n_trials: int = 50, use_smote: bool = True,
) -> dict:
    if model_name not in OPTUNA_OBJECTIVES:
        raise ValueError(f"Optimization not supported for {model_name}")

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    objective = OPTUNA_OBJECTIVES[model_name]

    study = optuna.create_study(direction="maximize", study_name=model_name)
    study.optimize(
        lambda trial: objective(trial, X, y, cv, use_smote=use_smote),
        n_trials=n_trials,
    )

    logger.info(f"{model_name} best F1: {study.best_value:.4f}")
    logger.info(f"{model_name} best params: {study.best_params}")

    return study.best_params


def build_optimized_model(model_name: str, best_params: dict) -> Any:
    if model_name == "XGBoost":
        return XGBClassifier(
            **best_params, random_state=42, eval_metric="logloss", verbosity=0,
        )
    elif model_name == "LightGBM":
        return LGBMClassifier(
            **best_params, is_unbalance=True, random_state=42, verbose=-1,
        )
    elif model_name == "CatBoost":
        return CatBoostClassifier(
            **best_params, auto_class_weights="Balanced", random_seed=42, verbose=0,
        )
    raise ValueError(f"Unknown model: {model_name}")


def build_stacking_ensemble(base_estimators: dict[str, Any]) -> StackingClassifier:
    estimators = [(name, model) for name, model in base_estimators.items()]
    return StackingClassifier(
        estimators=estimators,
        final_estimator=LogisticRegression(max_iter=1000, random_state=42),
        cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
        n_jobs=-1,
        passthrough=False,
    )


def build_voting_ensemble(base_estimators: dict[str, Any]) -> VotingClassifier:
    estimators = [(name, model) for name, model in base_estimators.items()]
    return VotingClassifier(estimators=estimators, voting="soft", n_jobs=-1)


class ModelTrainer:

    def __init__(self, use_smote: bool = True):
        self.use_smote = use_smote
        self.feature_engineer = FeatureEngineer()
        self.results = {}

    def benchmark(self, X_train, y_train, X_val, y_val) -> dict[str, dict]:
        X_train_fe = self.feature_engineer.fit_transform(X_train)
        X_val_fe = self.feature_engineer.transform(X_val)

        if self.use_smote:
            smote = SMOTE(random_state=42)
            X_train_resampled, y_train_resampled = smote.fit_resample(X_train_fe, y_train)
        else:
            X_train_resampled, y_train_resampled = X_train_fe, y_train

        models = get_base_models()
        results = {}

        for name, model in models.items():
            logger.info(f"Training {name}...")
            model.fit(X_train_resampled, y_train_resampled)

            y_pred = model.predict(X_val_fe)
            y_proba = model.predict_proba(X_val_fe)[:, 1]

            f1 = f1_score(y_val, y_pred)
            roc_auc = roc_auc_score(y_val, y_proba)

            results[name] = {"model": model, "f1": f1, "roc_auc": roc_auc}
            logger.info(f"  {name}: F1={f1:.4f}, ROC-AUC={roc_auc:.4f}")

        self.results = results
        return results

    def optimize_top_models(
        self, X_train, y_train, top_n: int = 3, n_trials: int = 50
    ) -> dict[str, Any]:
        # SMOTE is applied INSIDE each CV fold via imblearn Pipeline
        # to prevent data leakage between train/validation splits
        X_train_fe = self.feature_engineer.transform(X_train)

        optimizable = {
            name: data
            for name, data in self.results.items()
            if name in OPTUNA_OBJECTIVES
        }
        sorted_models = sorted(optimizable.items(), key=lambda x: x[1]["f1"], reverse=True)
        top_models = sorted_models[:top_n]

        optimized = {}
        for name, data in top_models:
            logger.info(f"Optimizing {name} ({n_trials} trials)...")
            best_params = optimize_model(
                name, X_train_fe, y_train,
                n_trials=n_trials, use_smote=self.use_smote,
            )
            optimized_model = build_optimized_model(name, best_params)
            optimized[name] = optimized_model

        return optimized

    def train_final_ensemble(
        self,
        optimized_models: dict[str, Any],
        X_train,
        y_train,
        ensemble_type: str = "stacking",
    ) -> Any:
        X_train_fe = self.feature_engineer.transform(X_train)

        if self.use_smote:
            smote = SMOTE(random_state=42)
            X_resampled, y_resampled = smote.fit_resample(X_train_fe, y_train)
        else:
            X_resampled, y_resampled = X_train_fe, y_train

        if ensemble_type == "stacking":
            ensemble = build_stacking_ensemble(optimized_models)
        else:
            ensemble = build_voting_ensemble(optimized_models)

        logger.info(f"Training {ensemble_type} ensemble...")
        ensemble.fit(X_resampled, y_resampled)
        return ensemble
