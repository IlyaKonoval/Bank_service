import warnings

import pickle
from sklearn.exceptions import UndefinedMetricWarning

warnings.filterwarnings("ignore", category=UndefinedMetricWarning)

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score


class MarketingCampaignModel:
    """
    Класс для предсказания отклика на маркетинговую кампанию.
    """

    def __init__(self, model):
        """
        Инициализация класса.

        Args:
            model: Обученная модель.
        """
        self.best_model = model
        self.scaler = StandardScaler()

    def predict_proba(self, X):
        """
        Возвращает вероятности предсказаний для каждого класса.

        Args:
            X: Данные для предсказания.

        Returns:
            Массив вероятностей.
        """
        return self.best_model.predict_proba(X)

    def predict(self, client_data):
        """
        Предсказывает вероятность отклика для нового клиента.

        Args:
            client_data: DataFrame с данными клиента.

        Returns:
            float: Вероятность отклика.
        """
        client_data_scaled = self.scaler.transform(client_data)
        probability = self.best_model.predict_proba(client_data_scaled)[0, 1]
        return probability


if __name__ == "__main__":
    """
    Точка входа в программу.
    """
    data_path = "processed_data.csv"
    model_path = "model.pkl"

    # --- Загрузка данных ---
    df = pd.read_csv(data_path)

    # --- Предобработка данных ---
    X = df.drop("TARGET", axis=1)
    y = df["TARGET"]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Разделение на train/val/test (80/10/10)
    X_train_temp, X_test, y_train_temp, y_test = train_test_split(
        X_scaled, y, test_size=0.1, random_state=42
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_temp, y_train_temp, test_size=0.11111, random_state=42
    )

    # --- Обучение и выбор лучшей модели ---
    def find_best_threshold(model, X_val, y_val):
        """
        Находит лучший порог для бинаризации вероятностей.
        """
        val_predictions = model.predict_proba(X_val)[:, 1]

        best_threshold = 0
        best_precision = 0
        best_recall = 0

        for threshold in range(0, 101, 1):
            threshold /= 100

            val_predictions_binary = (val_predictions > threshold).astype(int)
            precision = precision_score(y_val, val_predictions_binary)
            recall = recall_score(y_val, val_predictions_binary)

            if recall >= 0.66 and precision > best_precision:
                best_threshold = threshold
                best_precision = precision
                best_recall = recall

        return best_threshold

    def train_and_evaluate_model(model_type, X_train, y_train, X_val, y_val, X_test, y_test):
        """
        Обучает и оценивает выбранную модель.
        """
        if model_type == "lr":
            model = LogisticRegression()
        elif model_type == "svm":
            model = SVC(probability=True)
        elif model_type == "dt":
            model = DecisionTreeClassifier()
        else:
            raise ValueError("Неизвестный тип модели.")

        model.fit(X_train, y_train)
        best_threshold = find_best_threshold(model, X_val, y_val)

        test_predictions = model.predict_proba(X_test)[:, 1]
        test_predictions_binary = (test_predictions > best_threshold).astype(int)

        precision = precision_score(y_test, test_predictions_binary)
        recall = recall_score(y_test, test_predictions_binary)
        f1 = f1_score(y_test, test_predictions_binary)
        accuracy = accuracy_score(y_test, test_predictions_binary)

        print(f"Результаты для модели {model_type}:")
        print(f"Precision: {precision}")
        print(f"Recall: {recall}")
        print(f"F1-мера: {f1}")
        print(f"Accuracy: {accuracy}\n")

        return {"model": model, "precision": precision, "recall": recall, "f1": f1}

    results = {}
    for model_type in ["lr", "svm", "dt"]:
        results[model_type] = train_and_evaluate_model(
            model_type, X_train, y_train, X_val, y_val, X_test, y_test
        )

    best_model_type = max(results, key=lambda k: results[k]["f1"])
    best_model = results[best_model_type]["model"]
    print(f"Лучшая модель: {best_model_type}")

    # --- Сохранение лучшей модели ---
    with open(model_path, "wb") as file:
        pickle.dump(best_model, file)
    print(f"Модель сохранена в файл: {model_path}")