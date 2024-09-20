import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import os

from eda import DataProcessor

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
    processor = DataProcessor(data_path)
    df = processor.process_data()

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

        # Гистограмма
        fig, ax = plt.subplots()
        ax.hist(df[selected_feature], bins=20)
        ax.set_title(f"Распределение {selected_feature}")
        ax.set_xlabel(selected_feature)
        ax.set_ylabel("Частота")
        st.pyplot(fig)

        # Boxplot (ящик с усами)
        fig, ax = plt.subplots()
        ax.boxplot(df[selected_feature])
        ax.set_title(f"Boxplot {selected_feature}")
        ax.set_ylabel(selected_feature)
        st.pyplot(fig)

    elif analysis_type == "Категориальные признаки":
        st.header("Анализ категориальных признаков")
        categorical_features = ["GENDER", "SOCSTATUS_WORK_FL", "SOCSTATUS_PENS_FL"]
        selected_feature = st.selectbox("Выберите признак:", categorical_features)

        # Столбчатая диаграмма
        fig, ax = plt.subplots()
        sns.countplot(x=selected_feature, data=df)
        ax.set_title(f"Распределение {selected_feature}")
        ax.set_xlabel(selected_feature)
        ax.set_ylabel("Количество")
        st.pyplot(fig)

    elif analysis_type == "Корреляция":
        st.header("Анализ корреляции")
        correlation_method = st.radio(
            "Выберите метод корреляции:", ["Пирсон", "Спирмен"]
        )

        if correlation_method == "Пирсон":
            corr_mat = df.corr()
        else:
            corr_mat = df.corr(method="spearman")

        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(corr_mat, cmap="BuPu", annot=True, ax=ax)
        ax.set_title(f"Матрица корреляции ({correlation_method})")
        st.pyplot(fig)

    elif analysis_type == "Целевая переменная":
        st.header("Анализ целевой переменной (TARGET)")

        # Столбчатая диаграмма распределения TARGET
        fig, ax = plt.subplots()
        sns.countplot(x="TARGET", data=df)
        ax.set_title("Распределение целевой переменной (TARGET)")
        ax.set_xlabel("TARGET")
        ax.set_ylabel("Количество")
        st.pyplot(fig)

        # Доля клиентов с TARGET=1
        target_rate = df["TARGET"].mean()
        st.write(f"Доля клиентов, давших отклик (TARGET=1): {target_rate:.2f}")


if __name__ == "__main__":
    main()