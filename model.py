import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score


def load_data(data_path="processed_data.csv"):
    df = pd.read_csv(data_path)
    return df


def preprocess_data(df):
    X = df.drop("TARGET", axis=1)
    y = df["TARGET"]
    return X, y


def split_data(X, y):
    X_train_temp, X_test, y_train_temp, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_train_temp, y_train_temp, test_size=0.11111, random_state=42)
    return X_train, X_val, X_test, y_train, y_val, y_test

def fit_scaler(X):
    scaler = StandardScaler().fit(X)
    return scaler.mean_, scaler.scale_


def transform_scaler(X, mean_, std_):
    return (X - mean_) / std_


def find_best_threshold(model, X_val, y_val):
    val_predictions = model.predict_proba(X_val)[:, 1]
    best_threshold, best_precision, best_recall = 0, 0, 0
    for threshold in range(0, 101, 1):
        threshold /= 100
        val_predictions_binary = (val_predictions > threshold).astype(int)
        precision = precision_score(y_val, val_predictions_binary)
        recall = recall_score(y_val, val_predictions_binary)
        if recall >= 0.66 and precision > best_precision:
            best_threshold, best_precision, best_recall = threshold, precision, recall
    return best_threshold


def calculate_precision(y_true, y_pred):
    return precision_score(y_true, y_pred)


def calculate_recall(y_true, y_pred):
    return recall_score(y_true, y_pred)


def calculate_f1_score(y_true, y_pred):
    return f1_score(y_true, y_pred)


def calculate_accuracy(y_true, y_pred):
    return accuracy_score(y_true, y_pred)

def train_and_evaluate_model(model_type, X_train, y_train, X_val, y_val, X_test, y_test):
    if model_type == "lr":
        model = LogisticRegression(random_state=42)
    elif model_type == "svm":
        model = SVC(probability=True, random_state=42)
    elif model_type == "dt":
        model = DecisionTreeClassifier(random_state=42)
    else:
        raise ValueError("Unknown model type.")

    model.fit(X_train, y_train)
    best_threshold = find_best_threshold(model, X_val, y_val)
    test_predictions = model.predict_proba(X_test)[:, 1]
    test_predictions_binary = (test_predictions > best_threshold).astype(int)

    # Сохраняем порядок признаков
    feature_order = X_train.columns.tolist()

    precision = precision_score(y_test, test_predictions_binary)
    recall = recall_score(y_test, test_predictions_binary)
    f1 = f1_score(y_test, test_predictions_binary)
    accuracy = accuracy_score(y_test, test_predictions_binary)

    # Возвращаем весь словарь результатов
    return {
        "model": model,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "accuracy": accuracy,
        "feature_order": feature_order
    }


def save_model(model_data, model_path="model.pkl"):
    """Сохраняет словарь с данными модели (включая модель и порядок признаков)."""
    with open(model_path, "wb") as file:
        pickle.dump(model_data, file)

def load_model(model_path="model.pkl"):
    """Загружает модель из заданного пути."""
    with open(model_path, "rb") as file:
        return pickle.load(file)


if __name__ == "__main__":
    # Пример использования кода
    # Загрузка данных
    df = load_data()

    # Предобработка данных
    X, y = preprocess_data(df)

    # Разделение данных
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y)

    # Обучение и оценка модели
    model_type = "lr"
    results = train_and_evaluate_model(model_type, X_train, y_train, X_val, y_val, X_test, y_test)

    # Вывод результатов
    print("Precision:", results["precision"])
    print("Recall:", results["recall"])
    print("F1 Score:", results["f1"])
    print("Accuracy:", results["accuracy"])

    # Сохранение модели
    save_model(results)  # Передаем словарь results