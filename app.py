import streamlit as st
import altair as alt
import os
import pandas as pd
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    accuracy_score,
)
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


# --- Функции ---


def get_universal_data_path():
    """
    Возвращает универсальный путь к папке 'data'.

    Функция сначала проверяет, установлена ли переменная окружения
    DATA_PATH. Если да, то используется этот путь.
    Иначе, используется путь к папке, где находится текущий файл.
    """

    # Получаем значение переменной окружения DATA_PATH
    env_data_path = os.environ.get("DATA_PATH")
    # Если переменная DATA_PATH установлена, возвращаем ее значение
    if env_data_path:
        return env_data_path
    # Иначе, возвращаем путь к папке, где находится текущий файл
    return os.path.dirname(os.path.abspath(__file__))


def main():
    """
    Основная функция приложения Streamlit.
    Здесь происходит загрузка данных, модели,
    выбор типа анализа и отображение результатов.
    """

    # --- Заголовок приложения ---
    st.title("Анализ данных о клиентах банка и предсказание отклика")

    # --- Загрузка данных и модели, создание scaler ---
    data_path = get_universal_data_path()  # Получаем путь к папке с данными
    df = pd.read_csv(
        os.path.join(data_path, "processed_data.csv")
    )  # Загружаем данные из CSV файла

    model_path = os.path.join(
        data_path, "model.pkl"
    )  # Получаем путь к файлу с моделью
    model = joblib.load(model_path)  # Загружаем модель

    scaler = (
        StandardScaler()
    )  # Создаем объект для масштабирования данных (StandardScaler)

    # --- Боковая панель для выбора анализа ---
    analysis_type = st.sidebar.selectbox(
        "Выберите тип анализа:",  # Текст над выпадающим списком
        [
            "Обзор данных",
            "Числовые признаки",
            "Категориальные признаки",
            "Корреляция",
            "Целевая переменная",
            "Результаты модели",
            "Предсказание для клиента",
        ],  # Список доступных опций
    )

    # --- Различные типы анализа ---

    # --- Обзор данных ---
    if analysis_type == "Обзор данных":
        st.header("Обзор данных")  # Заголовок раздела
        st.write(df.head())  # Вывод первых 5 строк DataFrame
        st.write(df.describe())  # Вывод описательной статистики

        # Дополнительная информация
        st.subheader("Дополнительные показатели:")
        st.write(f"- Средний возраст: {df['AGE'].mean():.2f}")  # Средний возраст
        st.write(
            f"- Средний доход: {df['PERSONAL_INCOME'].mean():.2f}"
        )  # Средний доход
        st.write(
            f"- Доля работающих: {(df['SOCSTATUS_WORK_FL'] == 1).sum() / len(df):.2f}"
        )  # Доля работающих
        st.write(
            f"- Доля пенсионеров: {(df['SOCSTATUS_PENS_FL'] == 1).sum() / len(df):.2f}"
        )  # Доля пенсионеров

    # --- Анализ числовых признаков ---
    elif analysis_type == "Числовые признаки":
        st.header("Анализ числовых признаков")
        numerical_features = [
            "AGE",
            "PERSONAL_INCOME",
            "CHILD_TOTAL",
            "DEPENDANTS",
            "LOAN_NUM_TOTAL",
            "LOAN_NUM_CLOSED",
        ]  # Список числовых признаков
        selected_feature = st.selectbox(
            "Выберите признак:", numerical_features
        )  # Выбор признака из списка

        # Гистограмма
        st.bar_chart(
            df[selected_feature].value_counts().sort_index()
        )  # Построение гистограммы

        # Boxplot (ящик с усами)
        st.write("Boxplot:")
        chart = (
            alt.Chart(df)
            .mark_boxplot()
            .encode(y=selected_feature)  # Построение boxplot с помощью Altair
        )
        st.altair_chart(chart, use_container_width=True)  # Отображение графика

    # --- Анализ категориальных признаков ---
    elif analysis_type == "Категориальные признаки":
        st.header("Анализ категориальных признаков")
        categorical_features = [
            "GENDER",
            "SOCSTATUS_WORK_FL",
            "SOCSTATUS_PENS_FL",
        ]  # Список категориальных признаков
        selected_feature = st.selectbox(
            "Выберите признак:", categorical_features
        )  # Выбор признака из списка

        # Столбчатая диаграмма
        st.bar_chart(
            df[selected_feature].value_counts()
        )  # Построение столбчатой диаграммы

    # --- Анализ корреляции ---
    elif analysis_type == "Корреляция":
        st.header("Анализ корреляции")
        correlation_method = st.radio(
            "Выберите метод корреляции:", ["Пирсон", "Спирмен"]
        )  # Выбор метода корреляции

        # Расчет матрицы корреляции
        if correlation_method == "Пирсон":
            corr_mat = df.corr()
        else:
            corr_mat = df.corr(method="spearman")

        # Heatmap (тепловая карта)
        st.write(f"Матрица корреляции ({correlation_method}):")
        chart = (
            alt.Chart(corr_mat.reset_index())
            .mark_rect()
            .encode(
                x="index:O", y="index:O", color="value:Q"
            )  # Построение heatmap с помощью Altair
        )
        st.altair_chart(chart, use_container_width=True)  # Отображение графика

    # --- Анализ целевой переменной ---
    elif analysis_type == "Целевая переменная":
        st.header("Анализ целевой переменной (TARGET)")

        # Столбчатая диаграмма распределения TARGET
        st.bar_chart(df["TARGET"].value_counts())

        # Доля клиентов с TARGET=1
        target_rate = df["TARGET"].mean()
        st.write(f"Доля клиентов, давших отклик (TARGET=1): {target_rate:.2f}")

        # Исследование связи с числовыми признаками
        st.subheader("Связь с числовыми признаками:")
        numerical_features = [
            "AGE",
            "PERSONAL_INCOME",
            "CHILD_TOTAL",
            "DEPENDANTS",
            "LOAN_NUM_TOTAL",
            "LOAN_NUM_CLOSED",
        ]  # Список числовых признаков
        selected_feature = st.selectbox(
            "Выберите числовой признак:", numerical_features
        )  # Выбор признака из списка

        # Гистограммы для разных значений TARGET
        for target_value in df["TARGET"].unique():
            st.write(f"**TARGET = {target_value}**")
            st.bar_chart(
                df[df["TARGET"] == target_value][selected_feature]
                .value_counts()
                .sort_index()
            )  # Построение гистограммы

    # --- Результаты модели ---
    elif analysis_type == "Результаты модели":
        st.header("Результаты модели")

        # Выбор порога
        threshold = st.slider(
            "Выберите порог:", 0.0, 1.0, 0.5, 0.01
        )  # Ползунок для выбора порога

        # Получение предсказаний на тестовых данных
        X_test = df.drop("TARGET", axis=1)  # Отделяем признаки от целевой переменной
        X_test_scaled = scaler.fit_transform(
            X_test
        )  # Масштабируем тестовые данные

        # Проверка, поддерживает ли модель predict_proba
        if isinstance(model, SVC) and not hasattr(model, "predict_proba"):
            st.warning(
                "Выбранная модель SVM не поддерживает predict_proba. "
                "Для визуализации метрик в зависимости от порога "
                "необходимо переобучить модель с параметром probability=True."
            )
        else:
            # Получение вероятностей предсказаний
            test_predictions = model.predict_proba(X_test_scaled)[:, 1]
            # Преобразование вероятностей в бинарные предсказания по порогу
            test_predictions_binary = (test_predictions > threshold).astype(int)

            # Расчет метрик
            precision = precision_score(df["TARGET"], test_predictions_binary)
            recall = recall_score(df["TARGET"], test_predictions_binary)
            f1 = f1_score(df["TARGET"], test_predictions_binary)
            accuracy = accuracy_score(df["TARGET"], test_predictions_binary)

            # Вывод метрик
            st.write(f"Precision: {precision:.3f}")
            st.write(f"Recall: {recall:.3f}")
            st.write(f"F1-мера: {f1:.3f}")
            st.write(f"Accuracy: {accuracy:.3f}")

    # --- Предсказание для клиента ---
    elif analysis_type == "Предсказание для клиента":
        st.header("Предсказание для клиента")

        # --- Ввод данных клиента ---
        age = st.number_input("Возраст:", min_value=0, max_value=100, value=30)
        socstatus_work_fl = st.selectbox("Работает:", [0, 1])
        socstatus_pens_fl = st.selectbox("Пенсионер:", [0, 1])
        gender = st.selectbox("Пол (1 - мужчина, 0 - женщина):", [0, 1])
        child_total = st.number_input("Количество детей:", min_value=0, value=0)
        dependants = st.number_input("Количество иждивенцев:", min_value=0, value=0)
        personal_income = st.number_input(
            "Личный доход (руб.):", min_value=0, value=50000
        )
        loan_num_total = st.number_input("Количество ссуд:", min_value=0, value=0)
        loan_num_closed = st.number_input(
            "Количество погашенных ссуд:", min_value=0, value=0
        )

        # --- Создание DataFrame с данными клиента ---
        client_data = pd.DataFrame(
            {
                "AGE": [age],
                "SOCSTATUS_WORK_FL": [socstatus_work_fl],
                "SOCSTATUS_PENS_FL": [socstatus_pens_fl],
                "GENDER": [gender],
                "CHILD_TOTAL": [child_total],
                "DEPENDANTS": [dependants],
                "PERSONAL_INCOME": [personal_income],
                "LOAN_NUM_TOTAL": [loan_num_total],
                "LOAN_NUM_CLOSED": [loan_num_closed],
            }
        )

        # --- Масштабирование данных клиента ---
        client_data_scaled = scaler.fit_transform(client_data)

        # --- Предсказание вероятности отклика ---
        prediction = model.predict_proba(client_data_scaled)[0, 1]

        # --- Вывод предсказания ---
        st.write(f"Вероятность отклика на рекламу: {prediction:.3f}")


# --- Запуск приложения ---
if __name__ == "__main__":
    main()