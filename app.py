import streamlit as st
import altair as alt
import os
import pickle
import pandas as pd
from pathlib import Path


def get_base_path():
    env_path = os.environ.get("DATA_PATH")
    if env_path:
        return Path(env_path)
    return Path(os.path.dirname(os.path.abspath(__file__)))


def load_artifacts(artifacts_path):
    if artifacts_path.exists():
        with open(artifacts_path, "rb") as f:
            return pickle.load(f)
    return None


def predict_client(client_data, artifact):
    fe = artifact["feature_engineer"]
    model = artifact["model"]
    threshold = artifact["optimal_threshold"]

    client_fe = fe.transform(client_data)
    proba = model.predict_proba(client_fe)[0, 1]
    prediction = int(proba >= threshold)
    return proba, prediction


def main():
    st.set_page_config(page_title="Bank Marketing Prediction", layout="wide")
    st.title("Анализ данных клиентов банка и предсказание отклика")

    base_path = get_base_path()
    df = pd.read_csv(base_path / "processed_data.csv")

    artifacts_path = base_path / "artifacts" / "model.pkl"
    artifact = load_artifacts(artifacts_path)

    analysis_type = st.sidebar.selectbox(
        "Раздел:",
        [
            "Обзор данных",
            "Числовые признаки",
            "Категориальные признаки",
            "Корреляция",
            "Целевая переменная",
            "Сравнение моделей",
            "Результаты лучшей модели",
            "SHAP анализ",
            "Предсказание для клиента",
        ],
    )

    if analysis_type == "Обзор данных":
        st.header("Обзор данных")
        st.dataframe(df.head(20))
        st.subheader("Статистика")
        st.dataframe(df.describe())
        st.write(f"Размер датасета: {df.shape[0]} строк, {df.shape[1]} колонок")
        st.write(f"Доля отклика (TARGET=1): {df['TARGET'].mean():.2%}")

    elif analysis_type == "Числовые признаки":
        st.header("Анализ числовых признаков")
        numerical_features = [
            "AGE", "PERSONAL_INCOME", "CHILD_TOTAL", "DEPENDANTS",
            "LOAN_NUM_TOTAL", "LOAN_NUM_CLOSED",
        ]
        selected = st.selectbox("Признак:", numerical_features)

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Распределение")
            st.bar_chart(df[selected].value_counts().sort_index())
        with col2:
            st.subheader("Box Plot")
            chart = alt.Chart(df).mark_boxplot().encode(y=selected)
            st.altair_chart(chart, use_container_width=True)

        st.subheader("Распределение по TARGET")
        chart = alt.Chart(df).mark_bar(opacity=0.7).encode(
            x=alt.X(selected, bin=alt.Bin(maxbins=30)),
            y="count()",
            color="TARGET:N",
        )
        st.altair_chart(chart, use_container_width=True)

    elif analysis_type == "Категориальные признаки":
        st.header("Анализ категориальных признаков")
        categorical_features = ["GENDER", "SOCSTATUS_WORK_FL", "SOCSTATUS_PENS_FL"]
        selected = st.selectbox("Признак:", categorical_features)

        col1, col2 = st.columns(2)
        with col1:
            st.bar_chart(df[selected].value_counts())
        with col2:
            target_rate = df.groupby(selected)["TARGET"].mean()
            st.bar_chart(target_rate)
            st.caption("Доля отклика по группам")

    elif analysis_type == "Корреляция":
        st.header("Матрица корреляции")
        method = st.radio("Метод:", ["pearson", "spearman"])
        corr = df.corr(method=method)

        base = alt.Chart(
            corr.reset_index().melt("index")
        ).encode(
            x=alt.X("index:N", title=""),
            y=alt.Y("variable:N", title=""),
        )

        heatmap = base.mark_rect().encode(
            color=alt.Color("value:Q", scale=alt.Scale(scheme="blueorange", domain=[-1, 1]))
        )
        text = base.mark_text(fontSize=10).encode(
            text=alt.Text("value:Q", format=".2f"),
        )
        st.altair_chart(heatmap + text, use_container_width=True)

    elif analysis_type == "Целевая переменная":
        st.header("Распределение TARGET")
        counts = df["TARGET"].value_counts()
        st.bar_chart(counts)
        st.metric("Доля отклика", f"{df['TARGET'].mean():.2%}")
        st.metric("Всего клиентов", f"{len(df):,}")
        st.metric("Откликнулись", f"{counts.get(1, 0):,}")

    elif analysis_type == "Сравнение моделей":
        st.header("Сравнение моделей")
        if artifact and "all_results" in artifact:
            results_df = pd.DataFrame([
                {"Model": name, **metrics}
                for name, metrics in artifact["all_results"].items()
            ]).sort_values("f1", ascending=False)

            st.dataframe(results_df.style.highlight_max(
                subset=["f1", "roc_auc", "precision", "recall", "accuracy", "pr_auc"],
                color="#90EE90",
            ))

            chart = alt.Chart(results_df.melt("Model", var_name="Metric", value_name="Score")).mark_bar().encode(
                x=alt.X("Model:N", sort="-y"),
                y="Score:Q",
                color="Metric:N",
                column="Metric:N",
            ).properties(width=120, height=300)
            st.altair_chart(chart)

            comparison_img = base_path / "artifacts" / "model_comparison.png"
            if comparison_img.exists():
                st.image(str(comparison_img))
        else:
            st.warning("Запустите train.py для обучения моделей")

    elif analysis_type == "Результаты лучшей модели":
        st.header("Лучшая модель")
        if artifact:
            st.subheader(f"Модель: {artifact['best_model_name']}")

            metrics = artifact["test_metrics"]
            cols = st.columns(6)
            metric_names = ["accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc"]
            for i, name in enumerate(metric_names):
                cols[i].metric(name.upper(), f"{metrics[name]:.3f}")

            st.subheader(f"Оптимальный порог: {artifact['optimal_threshold']:.2f}")

            for suffix in ["_roc_curve.png", "_pr_curve.png", "_confusion_matrix.png"]:
                img_path = base_path / "artifacts" / f"{artifact['best_model_name']}{suffix}"
                if img_path.exists():
                    st.image(str(img_path))
        else:
            st.warning("Запустите train.py для обучения моделей")

    elif analysis_type == "SHAP анализ":
        st.header("SHAP — интерпретация модели")
        if artifact:
            for suffix in ["_shap_summary.png", "_shap_importance.png"]:
                img_path = base_path / "artifacts" / f"{artifact['best_model_name']}{suffix}"
                if img_path.exists():
                    st.image(str(img_path))
                else:
                    st.info(f"Файл {img_path.name} не найден")
        else:
            st.warning("Запустите train.py для обучения моделей")

    elif analysis_type == "Предсказание для клиента":
        st.header("Предсказание для клиента")
        if not artifact:
            st.warning("Запустите train.py для обучения моделей")
            return

        with st.form(key="client_form"):
            col1, col2, col3 = st.columns(3)
            with col1:
                age = st.number_input("Возраст:", min_value=18, max_value=80, value=35)
                gender = st.selectbox("Пол:", [("Мужской", 1), ("Женский", 0)], format_func=lambda x: x[0])
                child_total = st.number_input("Кол-во детей:", min_value=0, max_value=10, value=0)
            with col2:
                dependants = st.number_input("Кол-во иждивенцев:", min_value=0, max_value=10, value=0)
                personal_income = st.number_input("Доход (руб.):", min_value=5000, max_value=500000, value=50000)
                socstatus_work = st.selectbox("Работает:", [("Да", 1), ("Нет", 0)], format_func=lambda x: x[0])
            with col3:
                socstatus_pens = st.selectbox("Пенсионер:", [("Нет", 0), ("Да", 1)], format_func=lambda x: x[0])
                loan_total = st.number_input("Кол-во ссуд:", min_value=0, max_value=20, value=1)
                loan_closed = st.number_input("Погашенных ссуд:", min_value=0, max_value=20, value=0)

            submit = st.form_submit_button("Предсказать")

        if submit:
            if loan_closed > loan_total:
                st.error("Погашенных ссуд не может быть больше общего количества")
                return

            client_data = pd.DataFrame([{
                "AGE": age,
                "GENDER": gender[1],
                "CHILD_TOTAL": child_total,
                "DEPENDANTS": dependants,
                "PERSONAL_INCOME": float(personal_income),
                "SOCSTATUS_WORK_FL": socstatus_work[1],
                "SOCSTATUS_PENS_FL": socstatus_pens[1],
                "LOAN_NUM_TOTAL": loan_total,
                "LOAN_NUM_CLOSED": float(loan_closed),
            }])

            proba, prediction = predict_client(client_data, artifact)

            col1, col2 = st.columns(2)
            with col1:
                st.metric("Вероятность отклика", f"{proba:.1%}")
            with col2:
                if prediction == 1:
                    st.success("Клиент ОТКЛИКНЕТСЯ на кампанию")
                else:
                    st.error("Клиент НЕ откликнется на кампанию")

            st.progress(float(proba))


if __name__ == "__main__":
    main()
