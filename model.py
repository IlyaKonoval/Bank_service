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
    Класс для обучения и оценки моделей предсказания отклика на
    маркетинговую кампанию.
    """

    def __init__(self, data_path):
        """
        Инициализация класса.

        Args:
            data_path: Путь к файлу с данными.
        """
        self.scaler = StandardScaler()
        self.data_path = data_path
        self.df = None
        self.X_train = None
        self.X_val = None
        self.X_test = None
        self.y_train = None
        self.y_val = None
        self.y_test = None
        self.best_threshold = None
        self.best_model = None

    def load_data(self):
        """
        Загрузка данных из файла.
        """
        self.df = pd.read_csv(self.data_path)

    def preprocess_data(self):
        """
        Предобработка данных: разделение на признаки и целевую
        переменную, масштабирование признаков, разделение на
        обучающую, валидационную и тестовую выборки.
        """
        X = self.df.drop('TARGET', axis=1)
        y = self.df['TARGET']

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Разделение на train/val/test (80/10/10)
        X_train_temp, self.X_test, y_train_temp, self.y_test = train_test_split(
            X_scaled, y, test_size=0.1, random_state=42
        )
        self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(
            X_train_temp, y_train_temp, test_size=0.11111, random_state=42
        )


    def find_best_threshold(self, model, X_val, y_val):
        """
        Находит лучший порог для бинаризации вероятностей
        предсказаний на основе метрик precision и recall.

        Args:
            model: Обученная модель.
            X_val: Валидационные данные (признаки).
            y_val: Валидационные данные (целевая переменная).

        Returns:
            float: Лучший порог.
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

    def train_and_evaluate_model(self, model_type):
        """
        Обучает и оценивает выбранную модель.

        Args:
            model_type: Тип модели ('lr', 'svm', 'dt').
        """
        if model_type == 'lr':
            model = LogisticRegression()
        elif model_type == 'svm':
            model = SVC(probability=True)
        elif model_type == 'dt':
            model = DecisionTreeClassifier()
        else:
            raise ValueError("Неизвестный тип модели.")

        model.fit(self.X_train, self.y_train)
        self.best_threshold = self.find_best_threshold(model, self.X_val, self.y_val)

        test_predictions = model.predict_proba(self.X_test)[:, 1]
        test_predictions_binary = (test_predictions > self.best_threshold).astype(int)

        precision = precision_score(self.y_test, test_predictions_binary)
        recall = recall_score(self.y_test, test_predictions_binary)
        f1 = f1_score(self.y_test, test_predictions_binary)
        accuracy = accuracy_score(self.y_test, test_predictions_binary)

        print(f"Результаты для модели {model_type}:")
        print(f"Precision: {precision}")
        print(f"Recall: {recall}")
        print(f"F1-мера: {f1}")
        print(f"Accuracy: {accuracy}\n")

        return {'model': model, 'precision': precision, 'recall': recall, 'f1': f1}

    def find_best_model(self):
        """
        Обучает и сравнивает разные модели, выбирает
        лучшую по F1-мере.
        """
        results = {}
        for model_type in ['lr', 'svm', 'dt']:
            results[model_type] = self.train_and_evaluate_model(model_type)

        best_model_type = max(results, key=lambda k: results[k]['f1'])
        self.best_model = results[best_model_type]['model']
        print(f"Лучшая модель: {best_model_type}")

    def predict_proba(self, X):
        """
        Возвращает вероятности предсказаний для каждого класса.

        Args:
            X: Данные для предсказания.

        Returns:
            Массив вероятностей.
        """
        return self.best_model.predict_proba(X)

    def save_model(self, model_path):
        """
        Сохраняет лучшую модель в файл.

        Args:
            model_path: Путь к файлу для сохранения модели.
        """
        with open('model.pkl', 'wb') as file:
            pickle.dump(model, file)
    
    def predict(self, client_data):
        """
        Предсказывает вероятность отклика для нового клиента.

        Args:
            client_data: DataFrame с данными клиента (должны быть 
                         те же признаки, что и в обучающих данных).

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
    model = MarketingCampaignModel(data_path)
    model.load_data()
    model.preprocess_data()
    model.find_best_model()
    model.save_model(model_path)