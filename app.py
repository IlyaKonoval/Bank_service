import streamlit as st
import altair as alt
import os
import pandas as pd

def get_universal_data_path():
    """Возвращает универсальный путь к папке 'data'. """

    # Проверяем, установлена ли переменная окружения DATA_PATH
    env_data_path = os.environ.get("DATA_PATH")
    if env_data_path:
        return env_data_path

    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")


def main():
    st.title("Анализ данных о клиентах банка")

    data_path = get_universal_data_path()

    df = pd.read_csv(os.path.join(data_path, "processed_data.csv"))

    # Выбор анализа
    analysis_type = st.sidebar.selectbox(
        "Выберите тип анализа:",
        [
            "Обзор данных",
            "Числовые признаки",
            "Категориальные признаки",
            "Корреляция",
            "Целевая переменная",
        ],
    )

    if analysis_type == "Обзор данных":
        st.header("Обзор данных")
        st.write(df.head())
        st.write(df.describe())

        # Дополнительная информация
        st.subheader("Дополнительные показатели:")
        st.write(f"- Средний возраст: {df['AGE'].mean():.2f}")
        st.write(f"- Средний доход: {df['PERSONAL_INCOME'].mean():.2f}")
        st.write(f"- Доля работающих: {(df['SOCSTATUS_WORK_FL'] == 1).sum() / len(df):.2f}")
        st.write(f"- Доля пенсионеров: {(df['SOCSTATUS_PENS_FL'] == 1).sum() / len(df):.2f}")

    elif analysis_type == "Числовые признаки":
        st.header("Анализ числовых признаков")
        numerical_features = [
            "AGE",
            "PERSONAL_INCOME",
            "CHILD_TOTAL",
            "DEPENDANTS",
            "LOAN_NUM_TOTAL",
            "LOAN_NUM_CLOSED",
        ]
        selected_feature = st.selectbox("Выберите признак:", numerical_features)

        # Гистограмма - используем st.bar_chart
        st.bar_chart(df[selected_feature].value_counts().sort_index())

        # Boxplot (ящик с усами) - используем Altair
        st.write("Boxplot:")
        chart = alt.Chart(df).mark_boxplot().encode(
            y=selected_feature
        )
        st.altair_chart(chart, use_container_width=True)

    elif analysis_type == "Категориальные признаки":
        st.header("Анализ категориальных признаков")
        categorical_features = ["GENDER", "SOCSTATUS_WORK_FL", "SOCSTATUS_PENS_FL"]
        selected_feature = st.selectbox("Выберите признак:", categorical_features)

        # Столбчатая диаграмма - используем st.bar_chart
        st.bar_chart(df[selected_feature].value_counts())

    elif analysis_type == "Корреляция":
        st.header("Анализ корреляции")
        correlation_method = st.radio(
            "Выберите метод корреляции:", ["Пирсон", "Спирмен"]
        )

        if correlation_method == "Пирсон":
            corr_mat = df.corr()
        else:
            corr_mat = df.corr(method="spearman")

        # Heatmap (тепловая карта) - используем Altair
        st.write(f"Матрица корреляции ({correlation_method}):")
        chart = alt.Chart(corr_mat.reset_index()).mark_rect().encode(
            x='index:O',
            y='index:O',
            color='value:Q'
        )
        st.altair_chart(chart, use_container_width=True)

    elif analysis_type == "Целевая переменная":
        st.header("Анализ целевой переменной (TARGET)")

        # Столбчатая диаграмма распределения TARGET - используем st.bar_chart
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
        ]
        selected_feature = st.selectbox("Выберите числовой признак:", numerical_features)

        # Гистограммы для разных значений TARGET (с помощью st.bar_chart)
        for target_value in df["TARGET"].unique():
            st.write(f"**TARGET = {target_value}**")
            st.bar_chart(
                df[df["TARGET"] == target_value][selected_feature]
                .value_counts()
                .sort_index()
            )


if __name__ == "__main__":
    main()