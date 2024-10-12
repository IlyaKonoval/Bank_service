import streamlit as st
import altair as alt
import os
import pandas as pd
import pickle
import numpy as np
from model import (
    fit_scaler,
    transform_scaler,
    calculate_precision,
    calculate_recall,
    calculate_f1_score,
    calculate_accuracy,
)

# --- Функции ---

def get_universal_data_path():
    """
    Возвращает универсальный путь к папке 'data'.
    """
    env_data_path = os.environ.get("DATA_PATH")
    if env_data_path:
        return env_data_path
    return os.path.dirname(os.path.abspath(__file__))


def load_model(model_path):
    """Загружает модель и порядок признаков из заданного пути."""
    if os.path.exists(model_path):
        with open(model_path, 'rb') as file:
            loaded_data = pickle.load(file)
            return loaded_data.get("model"), loaded_data.get("feature_order")
    else:
        st.warning("Модель не найдена.")
        return None, None


def predict_for_client(client_data, loaded_model, mean_, std_, feature_order):
    """
    Выполняет предсказание для клиента на основе его данных.
    """
    client_data = client_data[feature_order]

    # Масштабируем данные
    client_data_scaled = transform_scaler(client_data, mean_, std_)

    # Выполняем предсказание
    prediction = loaded_model.predict_proba(client_data_scaled)[0, 1]
    return prediction


def main():
    """
    Основная функция приложения Streamlit.
    """

    # --- Заголовок приложения ---
    st.title("Анализ данных о клиентах банка и предсказание отклика")

    # --- Загрузка данных и модели ---
    data_path = get_universal_data_path()
    df = pd.read_csv(os.path.join(data_path, "processed_data.csv"))  # Загружаем данные

    model_path = os.path.join(data_path, "model.pkl")
    loaded_model, feature_order = load_model(model_path)

    # --- Предобработка данных (обучение scaler) ---
    X = df.drop("TARGET", axis=1)
    y = df["TARGET"]
    mean_, std_ = fit_scaler(X)  # Обучаем scaler

    # --- Боковая панель для выбора анализа ---
    analysis_type = st.sidebar.selectbox(
        "Выберите тип анализа:",
        [
            "Обзор данных",
            "Числовые признаки",
            "Категориальные признаки",
            "Корреляция",
            "Целевая переменная",
            "Результаты модели",
            "Предсказание для клиента",
        ],
    )

    # --- Различные типы анализа ---

    if analysis_type == "Обзор данных":
        st.header("Обзор данных")
        st.write(df.head())
        st.write(df.describe())

    elif analysis_type == "Числовые признаки":
        st.header("Анализ числовых признаков")
        numerical_features = [
            "AGE", "PERSONAL_INCOME", "CHILD_TOTAL", "DEPENDANTS", "LOAN_NUM_TOTAL", "LOAN_NUM_CLOSED"
        ]
        selected_feature = st.selectbox("Выберите признак:", numerical_features)
        st.bar_chart(df[selected_feature].value_counts().sort_index())

        chart = alt.Chart(df).mark_boxplot().encode(y=selected_feature)
        st.altair_chart(chart, use_container_width=True)

    elif analysis_type == "Категориальные признаки":
        st.header("Анализ категориальных признаков")
        categorical_features = ["GENDER", "SOCSTATUS_WORK_FL", "SOCSTATUS_PENS_FL"]
        selected_feature = st.selectbox("Выберите признак:", categorical_features)
        st.bar_chart(df[selected_feature].value_counts())

    elif analysis_type == "Корреляция":
        st.header("Анализ корреляции")
        correlation_method = st.radio("Выберите метод корреляции:", ["Пирсон", "Спирмен"])
        corr_mat = df.corr() if correlation_method == "Пирсон" else df.corr(method="spearman")
        st.write(f"Матрица корреляции ({correlation_method}):")
        chart = alt.Chart(corr_mat.reset_index()).mark_rect().encode(x="index:O", y="index:O", color="value:Q")
        st.altair_chart(chart, use_container_width=True)

    elif analysis_type == "Целевая переменная":
        st.header("Анализ целевой переменной (TARGET)")
        st.bar_chart(df["TARGET"].value_counts())
        target_rate = df["TARGET"].mean()
        st.write(f"Доля клиентов, давших отклик (TARGET=1): {target_rate:.2f}")

    elif analysis_type == "Результаты модели":
        st.header("Результаты модели")
        threshold = st.slider("Выберите порог:", 0.0, 1.0, 0.5, 0.01)
        X_test = df.drop("TARGET", axis=1)
        X_test_scaled = transform_scaler(X_test, mean_, std_)

        if loaded_model and hasattr(loaded_model, "predict_proba"):
            test_predictions = loaded_model.predict_proba(X_test_scaled)[:, 1]
            test_predictions_binary = (test_predictions > threshold).astype(int)

            precision = calculate_precision(df["TARGET"], test_predictions_binary)
            recall = calculate_recall(df["TARGET"], test_predictions_binary)
            f1 = calculate_f1_score(df["TARGET"], test_predictions_binary)
            accuracy = calculate_accuracy(df["TARGET"], test_predictions_binary)

            st.write(f"Precision: {precision:.3f}")
            st.write(f"Recall: {recall:.3f}")
            st.write(f"F1-мера: {f1:.3f}")
            st.write(f"Accuracy: {accuracy:.3f}")
        else:
            st.warning("Модель не загружена или не поддерживает predict_proba.")

    elif analysis_type == "Предсказание для клиента":
        st.header("Предсказание для клиента")
        with st.form(key="client_form"):
            age = st.number_input("Возраст:", min_value=0, max_value=100, value=30)
            socstatus_work_fl = st.selectbox("Работает:", [0, 1])
            socstatus_pens_fl = st.selectbox("Пенсионер:", [0, 1])
            gender = st.selectbox("Пол (1 - мужчина, 0 - женщина):", [0, 1])
            child_total = st.number_input("Количество детей:", min_value=0, value=0)
            dependants = st.number_input("Количество иждивенцев:", min_value=0, value=0)
            personal_income = st.number_input("Личный доход (руб.):", min_value=0, value=50000)
            loan_num_total = st.number_input("Количество ссуд:", min_value=0, value=0)
            loan_num_closed = st.number_input("Количество погашенных ссуд:", min_value=0, value=0)
            submit_button = st.form_submit_button(label="Сделать предсказание")

        if submit_button and loaded_model:
            client_data = pd.DataFrame({
                "AGE": [age],
                "SOCSTATUS_WORK_FL": [socstatus_work_fl],
                "SOCSTATUS_PENS_FL": [socstatus_pens_fl],
                "GENDER": [gender],
                "CHILD_TOTAL": [child_total],
                "DEPENDANTS": [dependants],
                "PERSONAL_INCOME": [personal_income],
                "LOAN_NUM_TOTAL": [loan_num_total],
                "LOAN_NUM_CLOSED": [loan_num_closed],
            })

            prediction = predict_for_client(client_data, loaded_model, mean_, std_, feature_order)
            st.write(f"Вероятность отклика на рекламу: {prediction:.3f}")

if __name__ == "__main__":
    main()