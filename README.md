# Bank Marketing Prediction Service

[![CI](https://github.com/IlyaKonoval/Bank_service/actions/workflows/ci.yml/badge.svg)](https://github.com/IlyaKonoval/Bank_service/actions/workflows/ci.yml)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/downloads/)
[![Code style: ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)

End-to-end ML-сервис для предсказания отклика клиентов банка на маркетинговую кампанию: от разведочного анализа данных до продакшн-ready Streamlit-приложения с автоматическим CI/CD и Docker-контейнеризацией.

## Постановка задачи

**Бизнес-задача:** банку необходимо определить, какие клиенты с наибольшей вероятностью откликнутся на маркетинговую кампанию, чтобы оптимизировать расходы на рекламу и повысить конверсию.

**Техническая задача:** построить бинарный классификатор на несбалансированных данных (доля целевого класса ~14%), способный ранжировать клиентов по вероятности отклика, и обернуть его в интерактивный веб-интерфейс.

**Датасет:** 11 595 клиентов, 9 признаков (возраст, пол, доход, количество детей/иждивенцев, занятость, пенсионный статус, количество кредитов), целевая переменная TARGET (0/1).

## Архитектура проекта

```
Bank_service/
├── app.py                   # Streamlit-приложение (UI + предсказания)
├── train.py                 # Оркестратор обучающего пайплайна
├── pipeline/
│   ├── features.py          # Feature Engineering (sklearn-совместимый трансформер)
│   ├── training.py          # Бенчмарк, Optuna-оптимизация, ансамбли
│   └── evaluation.py        # Метрики, визуализации, SHAP-анализ
├── tests/
│   ├── test_features.py     # 11 тестов на feature engineering
│   ├── test_training.py     # 7 тестов на обучение и ансамбли
│   └── test_evaluation.py   # 5 тестов на метрики и отчёты
├── data/                    # Исходные данные (CSV)
├── artifacts/               # Обученные модели, графики, метрики
├── Makefile                 # Команды: make train / test / lint / run
├── Dockerfile               # Обучение + деплой в одном образе
├── docker-compose.yml       # Оркестрация контейнера
├── .github/workflows/ci.yml # CI: тесты + coverage + линтер + Docker build
└── requirements.txt         # Зависимости
```

## ML-пайплайн

Обучающий пайплайн (`train.py`) реализует 6-фазную архитектуру:

### Фаза 1 — Бенчмарк
Обучение 6 базовых моделей с SMOTE-балансировкой на train-выборке, оценка на validation:

| Модель             | F1     | ROC-AUC |
|--------------------|--------|---------|
| LogisticRegression | 0.3338 | 0.6879  |
| XGBoost            | 0.2929 | 0.6139  |
| RandomForest       | 0.2835 | 0.6430  |
| LightGBM           | 0.0769 | 0.6186  |
| CatBoost           | 0.0615 | 0.6426  |
| GradientBoosting   | 0.0571 | 0.6387  |

### Фаза 2 — Оптимизация гиперпараметров
Автоматическая оптимизация топ-3 моделей через **Optuna** (50 trials, Stratified 5-Fold CV с SMOTE **внутри** каждого фолда через `imblearn.Pipeline` для предотвращения data leakage):

| Модель   | Best CV F1 |
|----------|-----------|
| CatBoost | 0.3108    |
| LightGBM | 0.2951    |
| XGBoost  | 0.2934    |

### Фаза 3 — Ансамбли
Обучение **Stacking** (мета-классификатор: Logistic Regression) и **Voting** (soft voting) ансамблей из оптимизированных моделей.

### Фаза 4 — Финальная оценка на тесте

| Модель   | F1     | ROC-AUC | Precision | Recall |
|----------|--------|---------|-----------|--------|
| CatBoost | 0.2917 | 0.6337  | 0.1987    | 0.5488 |
| Voting   | 0.2909 | 0.6246  | 0.1928    | 0.5915 |
| XGBoost  | 0.2734 | 0.6069  | 0.1725    | 0.6585 |
| LightGBM | 0.2689 | 0.6243  | 0.1951    | 0.4329 |
| Stacking | 0.2384 | 0.5936  | 0.1984    | 0.2988 |

Лучшая модель — **CatBoost** (F1=0.29, ROC-AUC=0.63). Для каждой модели генерируются ROC-кривая, PR-кривая и матрица ошибок.

> Задача сложная из-за сильного дисбаланса классов (86/14%) и ограниченного набора признаков. Полученные метрики согласованы между CV и тестом, что подтверждает отсутствие data leakage в пайплайне.

### Фаза 5 — Интерпретируемость (SHAP)
TreeExplainer для лучшей модели: summary plot и feature importance.

### Фаза 6 — Подбор оптимального порога
Перебор порогов на validation-выборке по F1-мере (оптимальный порог: 0.53, F1=0.34).

## Feature Engineering

Sklearn-совместимый трансформер `FeatureEngineer` (fit/transform API):

- **LOAN_CLOSE_RATIO** — доля погашенных кредитов
- **OPEN_LOANS** — количество открытых кредитов
- **INCOME_PER_DEPENDANT** — доход на иждивенца
- **INCOME_LOG** — логарифм дохода (снижение влияния выбросов)
- **AGE_INCOME_INTERACTION** — взаимодействие возраста и дохода
- **LOAN_BURDEN** — кредитная нагрузка относительно дохода
- **AGE_GROUP** — бинаризация возраста (5 групп: young/adult/middle/senior/elder)
- **INCOME_GROUP** — бинаризация дохода (5 уровней)
- **StandardScaler** — нормализация всех признаков

## Streamlit-приложение

Интерактивный дашборд с 9 разделами:

1. **Обзор данных** — таблица, описательная статистика, доля отклика
2. **Числовые признаки** — гистограммы, boxplot, распределение по TARGET
3. **Категориальные признаки** — столбчатые диаграммы, доля отклика по группам
4. **Корреляция** — heatmap (Pearson/Spearman)
5. **Целевая переменная** — распределение, метрики
6. **Сравнение моделей** — таблица с подсветкой лучших, графики по метрикам
7. **Результаты лучшей модели** — ROC, PR, Confusion Matrix
8. **SHAP-анализ** — интерпретация предсказаний
9. **Предсказание для клиента** — форма ввода параметров, real-time предсказание

## Инфраструктура и DevOps

### CI/CD (GitHub Actions)
- Тесты с coverage на Python 3.11 и 3.12 с кэшированием зависимостей
- Линтинг (ruff)
- Сборка Docker-образа
- Отчёт покрытия в Codecov

### Docker
```bash
# Запуск через Docker Compose
docker-compose up --build

# Или напрямую
docker build -t bank-service .
docker run -p 8501:8501 bank-service
```

Dockerfile выполняет полный цикл: установка зависимостей, обучение модели, запуск Streamlit-сервера с healthcheck.

### Тестирование
```bash
make test       # Запуск тестов
make coverage   # Тесты с отчётом покрытия
make lint       # Линтер
make train      # Обучение моделей
make run        # Запуск Streamlit
```

**23 теста** покрывают:
- Feature engineering: создание признаков, масштабирование, обработка граничных случаев
- Training: бенчмарк 6 моделей, SMOTE, ансамбли
- Evaluation: метрики, генерация отчётов, подбор порога

## Стек технологий

| Категория        | Инструменты                                         |
|------------------|-----------------------------------------------------|
| ML-модели        | XGBoost, LightGBM, CatBoost, Scikit-learn           |
| Оптимизация      | Optuna (байесовская оптимизация гиперпараметров)     |
| Балансировка     | SMOTE (imbalanced-learn) через imblearn.Pipeline     |
| Интерпретация    | SHAP (TreeExplainer)                                 |
| Feature Eng.     | Pandas, NumPy, Scikit-learn Pipeline API             |
| Визуализация     | Altair, Matplotlib, Streamlit                        |
| Web UI           | Streamlit                                            |
| Контейнеризация  | Docker, Docker Compose                               |
| CI/CD            | GitHub Actions                                       |
| Качество кода    | pytest, pytest-cov, ruff                             |

## Быстрый старт

```bash
# 1. Клонировать репозиторий
git clone https://github.com/IlyaKonoval/Bank_service.git
cd Bank_service

# 2. Установить зависимости
pip install -r requirements.txt

# 3. Обучить модели
make train

# 4. Запустить приложение
make run
```

Приложение будет доступно по адресу `http://localhost:8501`.

## Ключевые инженерные решения

1. **Модульная архитектура пайплайна** — разделение на features/training/evaluation позволяет независимо тестировать и модифицировать каждый этап
2. **Sklearn-совместимый трансформер** — `FeatureEngineer` реализует `fit`/`transform` API, что обеспечивает совместимость с Pipeline и предотвращает data leakage
3. **SMOTE внутри CV через imblearn.Pipeline** — балансировка применяется только к train-фолду каждого сплита, что исключает утечку синтетических сэмплов в валидационный фолд
4. **Автоматический подбор моделей** — бенчмарк 6 алгоритмов + Optuna-оптимизация топ-3 + ансамбли (Stacking, Voting)
5. **Оптимизация порога** — подбор threshold по F1 на validation вместо дефолтного 0.5
6. **Docker-контейнеризация** — воспроизводимость: обучение и деплой в одном образе
7. **CI/CD** — автоматические тесты с coverage, линтинг и сборка Docker при каждом push/PR
