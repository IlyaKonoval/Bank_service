import logging
import pickle
import sys
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from pipeline.training import ModelTrainer
from pipeline.evaluation import ModelEvaluator

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("artifacts/training.log", encoding="utf-8"),
    ],
)
logger = logging.getLogger(__name__)

ARTIFACTS_DIR = Path("artifacts")
ARTIFACTS_DIR.mkdir(exist_ok=True)


def load_data(path: str = "processed_data.csv") -> pd.DataFrame:
    return pd.read_csv(path)


def split_data(df: pd.DataFrame):
    X = df.drop("TARGET", axis=1)
    y = df["TARGET"]

    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
    )

    logger.info(f"Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")
    logger.info(f"Target ratio — train: {y_train.mean():.3f}, val: {y_val.mean():.3f}, test: {y_test.mean():.3f}")

    return X_train, X_val, X_test, y_train, y_val, y_test


def main():
    logger.info("=" * 60)
    logger.info("STARTING TRAINING PIPELINE")
    logger.info("=" * 60)

    df = load_data()
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(df)

    # === 1. Benchmark ===
    logger.info("\n--- PHASE 1: Benchmark ---")
    trainer = ModelTrainer(use_smote=True)
    benchmark_results = trainer.benchmark(X_train, y_train, X_val, y_val)

    evaluator = ModelEvaluator(output_dir="artifacts")
    comparison_df = evaluator.compare_models(benchmark_results)
    logger.info(f"\nBenchmark results:\n{comparison_df.to_string()}")

    # === 2. Optuna optimization ===
    logger.info("\n--- PHASE 2: Hyperparameter Optimization ---")
    optimized_models = trainer.optimize_top_models(
        X_train, y_train, top_n=3, n_trials=50
    )

    # === 3. Train ensemble ===
    logger.info("\n--- PHASE 3: Ensemble Training ---")
    stacking = trainer.train_final_ensemble(
        optimized_models, X_train, y_train, ensemble_type="stacking"
    )
    voting = trainer.train_final_ensemble(
        optimized_models, X_train, y_train, ensemble_type="voting"
    )

    # === 4. Evaluate all on test set ===
    logger.info("\n--- PHASE 4: Test Set Evaluation ---")
    X_test_fe = trainer.feature_engineer.transform(X_test)
    feature_names = trainer.feature_engineer.get_feature_names_out()

    all_final_models = {
        **{name: model for name, model in optimized_models.items()},
        "Stacking": stacking,
        "Voting": voting,
    }

    # Train optimized individual models on SMOTE data
    from imblearn.over_sampling import SMOTE
    X_train_fe = trainer.feature_engineer.transform(X_train)
    smote = SMOTE(random_state=42)
    X_smote, y_smote = smote.fit_resample(X_train_fe, y_train)

    test_results = {}
    for name, model in all_final_models.items():
        if name not in ("Stacking", "Voting"):
            model.fit(X_smote, y_smote)
        metrics = evaluator.full_report(model, X_test_fe, y_test, model_name=name)
        test_results[name] = metrics

    # === 5. SHAP ===
    logger.info("\n--- PHASE 5: SHAP Analysis ---")
    best_model_name = max(test_results, key=lambda k: test_results[k]["f1"])
    best_model = all_final_models[best_model_name]

    if best_model_name not in ("Stacking", "Voting"):
        evaluator.shap_analysis(best_model, X_test_fe, feature_names, model_name=best_model_name)

    # === 6. Find optimal threshold ===
    X_val_fe = trainer.feature_engineer.transform(X_val)
    optimal_threshold = evaluator.find_optimal_threshold(best_model, X_val_fe, y_val)

    # === 7. Save artifacts ===
    logger.info("\n--- PHASE 6: Saving Artifacts ---")

    artifact = {
        "model": best_model,
        "feature_engineer": trainer.feature_engineer,
        "feature_names": feature_names,
        "optimal_threshold": optimal_threshold,
        "best_model_name": best_model_name,
        "test_metrics": test_results[best_model_name],
        "all_results": {name: {k: v for k, v in m.items() if k != "model_name"} for name, m in test_results.items()},
    }

    with open(ARTIFACTS_DIR / "model.pkl", "wb") as f:
        pickle.dump(artifact, f, protocol=pickle.HIGHEST_PROTOCOL)

    logger.info(f"Best model: {best_model_name}")
    logger.info(f"Test F1: {test_results[best_model_name]['f1']:.4f}")
    logger.info(f"Test ROC-AUC: {test_results[best_model_name]['roc_auc']:.4f}")
    logger.info(f"Optimal threshold: {optimal_threshold:.2f}")
    logger.info(f"Artifacts saved to {ARTIFACTS_DIR}/")
    logger.info("PIPELINE COMPLETE")


if __name__ == "__main__":
    main()
