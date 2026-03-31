import json
import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

logger = logging.getLogger(__name__)


class ModelEvaluator:

    def __init__(self, output_dir: str = "artifacts"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

    def full_report(self, model, X_test, y_test, model_name: str = "model") -> dict:
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]

        metrics = {
            "model_name": model_name,
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred),
            "recall": recall_score(y_test, y_pred),
            "f1": f1_score(y_test, y_pred),
            "roc_auc": roc_auc_score(y_test, y_proba),
            "pr_auc": average_precision_score(y_test, y_proba),
        }

        logger.info(f"\n{'='*50}")
        logger.info(f"Model: {model_name}")
        for k, v in metrics.items():
            if k != "model_name":
                logger.info(f"  {k}: {v:.4f}")
        logger.info(f"\n{classification_report(y_test, y_pred)}")

        self._plot_roc_curve(y_test, y_proba, model_name)
        self._plot_pr_curve(y_test, y_proba, model_name)
        self._plot_confusion_matrix(y_test, y_pred, model_name)

        with open(self.output_dir / f"{model_name}_metrics.json", "w") as f:
            json.dump(metrics, f, indent=2)

        return metrics

    def compare_models(self, results: dict[str, dict]) -> pd.DataFrame:
        rows = []
        for name, data in results.items():
            rows.append({"model": name, "f1": data["f1"], "roc_auc": data["roc_auc"]})

        df = pd.DataFrame(rows).sort_values("f1", ascending=False).reset_index(drop=True)

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(df)))

        axes[0].barh(df["model"], df["f1"], color=colors)
        axes[0].set_xlabel("F1 Score")
        axes[0].set_title("F1 Score")

        axes[1].barh(df["model"], df["roc_auc"], color=colors)
        axes[1].set_xlabel("ROC-AUC")
        axes[1].set_title("ROC-AUC")

        plt.tight_layout()
        plt.savefig(self.output_dir / "model_comparison.png", dpi=150, bbox_inches="tight")
        plt.close()

        return df

    def shap_analysis(self, model, X_test, feature_names: list[str], model_name: str = "model"):
        try:
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_test)

            if isinstance(shap_values, list):
                shap_values = shap_values[1]

            plt.figure(figsize=(12, 8))
            shap.summary_plot(
                shap_values, X_test,
                feature_names=feature_names,
                show=False,
                max_display=20,
            )
            plt.tight_layout()
            plt.savefig(
                self.output_dir / f"{model_name}_shap_summary.png",
                dpi=150, bbox_inches="tight",
            )
            plt.close()

            plt.figure(figsize=(12, 6))
            shap.summary_plot(
                shap_values, X_test,
                feature_names=feature_names,
                plot_type="bar",
                show=False,
                max_display=20,
            )
            plt.tight_layout()
            plt.savefig(
                self.output_dir / f"{model_name}_shap_importance.png",
                dpi=150, bbox_inches="tight",
            )
            plt.close()

            logger.info(f"SHAP analysis saved for {model_name}")
        except Exception as e:
            logger.warning(f"SHAP analysis failed for {model_name}: {e}")

    def find_optimal_threshold(self, model, X_val, y_val) -> float:
        y_proba = model.predict_proba(X_val)[:, 1]
        best_threshold = 0.5
        best_f1 = 0

        for threshold in np.arange(0.1, 0.9, 0.01):
            y_pred = (y_proba >= threshold).astype(int)
            score = f1_score(y_val, y_pred)
            if score > best_f1:
                best_f1 = score
                best_threshold = threshold

        logger.info(f"Optimal threshold: {best_threshold:.2f} (F1={best_f1:.4f})")
        return best_threshold

    def _plot_roc_curve(self, y_test, y_proba, model_name: str):
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        auc = roc_auc_score(y_test, y_proba)

        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, label=f"ROC (AUC = {auc:.3f})", linewidth=2)
        plt.plot([0, 1], [0, 1], "k--", alpha=0.5)
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title(f"ROC Curve — {model_name}")
        plt.legend()
        plt.grid(alpha=0.3)
        plt.savefig(
            self.output_dir / f"{model_name}_roc_curve.png",
            dpi=150, bbox_inches="tight",
        )
        plt.close()

    def _plot_pr_curve(self, y_test, y_proba, model_name: str):
        precision, recall, _ = precision_recall_curve(y_test, y_proba)
        ap = average_precision_score(y_test, y_proba)

        plt.figure(figsize=(8, 6))
        plt.plot(recall, precision, label=f"PR (AP = {ap:.3f})", linewidth=2)
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.title(f"Precision-Recall Curve — {model_name}")
        plt.legend()
        plt.grid(alpha=0.3)
        plt.savefig(
            self.output_dir / f"{model_name}_pr_curve.png",
            dpi=150, bbox_inches="tight",
        )
        plt.close()

    def _plot_confusion_matrix(self, y_test, y_pred, model_name: str):
        cm = confusion_matrix(y_test, y_pred)

        plt.figure(figsize=(6, 5))
        plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
        plt.title(f"Confusion Matrix — {model_name}")
        plt.colorbar()

        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                plt.text(j, i, str(cm[i, j]), ha="center", va="center",
                         color="white" if cm[i, j] > cm.max() / 2 else "black")

        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.xticks([0, 1], ["No Response", "Response"])
        plt.yticks([0, 1], ["No Response", "Response"])
        plt.tight_layout()
        plt.savefig(
            self.output_dir / f"{model_name}_confusion_matrix.png",
            dpi=150, bbox_inches="tight",
        )
        plt.close()
