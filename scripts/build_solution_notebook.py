from __future__ import annotations

from pathlib import Path
from textwrap import dedent

import nbformat as nbf


def md(text: str):
    return nbf.v4.new_markdown_cell(dedent(text).strip() + "\n")


def code(text: str):
    return nbf.v4.new_code_cell(dedent(text).strip() + "\n")


path = Path("ml-env/work.ipynb")
nb = nbf.read(path, as_version=4)
base_cells = nb.cells[:23]

cells = [
    md(
        """
        <a id="author-solution"></a>
        # Авторское решение: прогнозирование спроса на велопрокат BikeSochi

        Ниже я оформляю решение как воспроизводимое ML-исследование: от проверки исходных файлов и baseline-модели до подбора нелинейных моделей, финального тестирования и сохранения рабочего pipeline.

        **Цель:** выбрать модель, которая лучше предоставленной линейной регрессии прогнозирует почасовой спрос `Rented Bike Count`.

        **Основная метрика:** `RMSE`, потому что именно ее нужно минимизировать в задаче. Дополнительно считаю `MAE` и `R2`: первая показывает среднюю ошибку в велосипедах, вторая - долю объясненной вариации спроса.

        **Ключевое ограничение:** `ds_s14_test_data.csv` не используется для подбора гиперпараметров и выбора модели. Новые модели выбираются только по 5-fold CV на `ds_s14_train_data.csv`; test применяется один раз в конце.
        """
    ),
    md(
        """
        <a id="navigation"></a>
        ## Навигация по исследованию

        1. [Методологическая рамка](#methodology)
        2. [Этап 1. Среда, константы и пути](#stage-1)
        3. [Этап 2. Загрузка данных и контракт схемы](#stage-2)
        4. [Этап 3. Baseline компании](#stage-3)
        5. [Этап 4. Первичный аудит и EDA](#stage-4)
        6. [Этап 5. Pipeline и feature engineering](#stage-5)
        7. [Этап 6. Optuna и 5-fold CV](#stage-6)
        8. [Этап 7. Финальная проверка на test](#stage-7)
        9. [Этап 8. Интерпретация и артефакты](#stage-8)
        10. [Финальные выводы](#final-conclusions)
        """
    ),
    md(
        """
        <a id="methodology"></a>
        ## Методологическая рамка

        В этой задаче легко получить внешне хороший результат и при этом испортить честность оценки. Поэтому заранее фиксирую правила эксперимента.

        - Все обучаемые преобразования (`imputer`, `scaler`, `encoder`, модель) живут внутри `Pipeline`.
        - Финальная test-выборка не участвует в Optuna, выборе модели или выборе признаков.
        - Для новых моделей используется единая `KFold(n_splits=5, shuffle=True, random_state=42)`.
        - В Optuna минимизируется RMSE. Так как `sklearn` для regression scoring возвращает отрицательные значения, в objective используется `-scores["test_rmse"].mean()`.
        - Baseline компании оценивается отдельно как предоставленная модель. Его test-метрики нужны для итогового сравнения, но не для настройки новых моделей.
        - Графики используются не декоративно: каждый блок EDA должен дать решение для предобработки, признаков или интерпретации.
        """
    ),
    md(
        """
        <a id="stage-1"></a>
        ## Этап 1. Среда, константы и пути

        Сначала фиксирую импорты, настройки отображения, seed и рабочие пути. Это защищает ноутбук от скрытого состояния: его можно перезапустить сверху вниз и получить те же фолды, те же trials Optuna и те же метрики.
        """
    ),
    code(
        r'''
        from __future__ import annotations

        import hashlib
        import importlib
        import json
        import platform
        import sys
        import time
        import warnings
        from pathlib import Path
        from typing import Any, Dict, List, Optional, Tuple

        import joblib
        import matplotlib.pyplot as plt
        import numpy as np
        import optuna
        import pandas as pd
        import seaborn as sns
        import sklearn
        from IPython.display import Markdown, display
        from sklearn.base import BaseEstimator, TransformerMixin
        from sklearn.compose import ColumnTransformer
        from sklearn.dummy import DummyRegressor
        from sklearn.inspection import permutation_importance
        from sklearn.impute import SimpleImputer
        from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
        from sklearn.model_selection import KFold, cross_validate
        from sklearn.neighbors import KNeighborsRegressor
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import OneHotEncoder, StandardScaler
        from sklearn.tree import DecisionTreeRegressor

        warnings.filterwarnings("ignore", category=FutureWarning)
        warnings.filterwarnings("ignore", category=UserWarning)
        optuna.logging.set_verbosity(optuna.logging.WARNING)

        RANDOM_STATE = 42
        CV_SPLITS = 5
        N_TRIALS_KNN = 35
        N_TRIALS_TREE = 45
        TARGET_ORIGINAL = "Rented Bike Count"
        TARGET = "rented_bike_count"

        def find_project_root(start: Optional[Path] = None) -> Path:
            start = (start or Path.cwd()).resolve()
            for candidate in [start, *start.parents]:
                has_project_files = all(
                    [
                        (candidate / "data" / "raw" / "ds_s14_train_data.csv").exists(),
                        (candidate / "data" / "raw" / "ds_s14_test_data.csv").exists(),
                        (candidate / "models" / "baseline_linear_regression_pipeline.joblib").exists(),
                    ]
                )
                if has_project_files:
                    return candidate
            return start


        PROJECT_ROOT = find_project_root()
        if str(PROJECT_ROOT) not in sys.path:
            sys.path.insert(0, str(PROJECT_ROOT))

        DATA_DIR = PROJECT_ROOT / "data" / "raw"
        MODELS_DIR = PROJECT_ROOT / "models"
        REPORTS_DIR = PROJECT_ROOT / "reports"
        COMPONENT_MODULE_NAME = "bike_demand_pipeline_components"
        COMPONENT_MODULE_PATH = PROJECT_ROOT / f"{COMPONENT_MODULE_NAME}.py"
        MODELS_DIR.mkdir(exist_ok=True)
        REPORTS_DIR.mkdir(exist_ok=True)

        pd.set_option("display.max_columns", 120)
        pd.set_option("display.max_rows", 120)
        pd.set_option("display.max_colwidth", None)
        pd.set_option("display.float_format", lambda value: f"{value:,.4f}")

        sns.set_theme(style="whitegrid", context="notebook")
        plt.rcParams["figure.figsize"] = (11, 5)
        plt.rcParams["axes.titlesize"] = 13
        plt.rcParams["axes.labelsize"] = 11

        versions = pd.DataFrame(
            [
                {"package": "python", "version": platform.python_version()},
                {"package": "pandas", "version": pd.__version__},
                {"package": "numpy", "version": np.__version__},
                {"package": "scikit-learn", "version": sklearn.__version__},
                {"package": "optuna", "version": optuna.__version__},
                {"package": "joblib", "version": joblib.__version__},
            ]
        )
        display(versions)
        '''
    ),
    md(
        """
        **Подвывод по этапу 1:** окружение зафиксировано. Особенно важна версия `scikit-learn`: предоставленный `baseline_linear_regression_pipeline.joblib` корректно загружается на `1.6.1`, поэтому эта версия закреплена в `requirements.txt`.

        """
    ),
    md(
        """
        <a id="stage-2"></a>
        ## Этап 2. Загрузка данных и контракт схемы

        CSV-файлы приходят с человекочитаемыми названиями колонок, а предоставленный baseline pipeline ожидает `snake_case`. Поэтому сначала создаю единую функцию нормализации схемы. Это снижает риск, что baseline и новые модели будут обучаться на разных представлениях одних и тех же данных.
        """
    ),
    code(
        r'''
        COLUMN_RENAME = {
            "Temperature": "temperature",
            "Humidity(%)": "humidity",
            "Wind speed (m/s)": "wind_speed_ms",
            "Visibility (10m)": "visibility_10m",
            "Dew point temperature": "dew_point_temperature",
            "Solar Radiation (MJ/m2)": "solar_radiation_mjm2",
            "Rainfall(mm)": "rainfallmm",
            "Snowfall (cm)": "snowfall_cm",
            "Seasons": "seasons",
            "Holiday": "holiday",
            "Functioning Day": "functioning_day",
            "Time_Period_Evening": "time_period_evening",
            "Time_Period_Late Evening": "time_period_late_evening",
            "Time_Period_Morning": "time_period_morning",
            "Time_Period_Night": "time_period_night",
            TARGET_ORIGINAL: TARGET,
        }

        BASE_NUMERIC_FEATURES = [
            "temperature",
            "humidity",
            "wind_speed_ms",
            "visibility_10m",
            "dew_point_temperature",
            "solar_radiation_mjm2",
            "rainfallmm",
            "snowfall_cm",
        ]
        CATEGORICAL_FEATURES = ["seasons", "holiday", "functioning_day"]
        TIME_FEATURES = [
            "time_period_evening",
            "time_period_late_evening",
            "time_period_morning",
            "time_period_night",
        ]
        BASE_FEATURES = BASE_NUMERIC_FEATURES + CATEGORICAL_FEATURES + TIME_FEATURES


        def read_csv_with_fallback(filename: str) -> pd.DataFrame:
            candidates = [DATA_DIR / filename, Path("/datasets") / filename]
            for candidate in candidates:
                if candidate.exists():
                    return pd.read_csv(candidate)
            raise FileNotFoundError(f"Could not find {filename} in: {candidates}")


        def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
            normalized = df.rename(columns=COLUMN_RENAME).copy()
            expected = set(COLUMN_RENAME.values())
            missing = sorted(expected - set(normalized.columns))
            if missing:
                raise ValueError(f"Missing expected columns after normalization: {missing}")
            return normalized[list(COLUMN_RENAME.values())]


        train_raw = read_csv_with_fallback("ds_s14_train_data.csv")
        test_raw = read_csv_with_fallback("ds_s14_test_data.csv")
        train = normalize_columns(train_raw)
        test = normalize_columns(test_raw)

        X_train = train.drop(columns=TARGET)
        y_train = train[TARGET]
        X_test = test.drop(columns=TARGET)
        y_test = test[TARGET]

        data_overview = pd.DataFrame(
            [
                {"dataset": "train", "rows": train.shape[0], "columns": train.shape[1]},
                {"dataset": "test", "rows": test.shape[0], "columns": test.shape[1]},
            ]
        )
        display(data_overview)
        display(train.head())

        assert train.shape == (7008, 16), "Unexpected train shape"
        assert test.shape == (1752, 16), "Unexpected test shape"
        assert list(train.columns) == list(test.columns), "Train/test schemas differ"
        assert TARGET in train.columns and TARGET in test.columns
        '''
    ),
    md(
        """
        **Подвывод по этапу 2:** данные загружены и приведены к единому контракту схемы. В train `7008` строк, в test `1752` строки; набор колонок совпадает. Дальше все модели получают одинаковые названия признаков, а значит сравнение baseline, KNN и дерева не смешивает разные схемы.

        """
    ),
    md(
        """
        <a id="stage-3"></a>
        ## Этап 3. Baseline компании

        Компания предоставила готовый pipeline линейной регрессии. Его нужно не переобучать, а загрузить и оценить. Это дает точку сравнения: новая модель должна быть не просто обучена, а практически лучше текущего подхода BikeSochi.
        """
    ),
    code(
        r'''
        baseline_path = MODELS_DIR / "baseline_linear_regression_pipeline.joblib"
        baseline_pipeline = joblib.load(baseline_path)
        print(baseline_pipeline)

        baseline_blocks = []
        for block_name, transformer, columns in baseline_pipeline.named_steps["preprocessor"].transformers_:
            baseline_blocks.append(
                {
                    "block": block_name,
                    "transformer": repr(transformer),
                    "columns": ", ".join(columns),
                }
            )
        display(pd.DataFrame(baseline_blocks))
        '''
    ),
    code(
        r'''
        def regression_metrics(y_true: pd.Series, y_pred: np.ndarray) -> Dict[str, float]:
            return {
                "RMSE": root_mean_squared_error(y_true, y_pred),
                "MAE": mean_absolute_error(y_true, y_pred),
                "R2": r2_score(y_true, y_pred),
                "prediction_min": float(np.min(y_pred)),
                "prediction_max": float(np.max(y_pred)),
                "prediction_mean": float(np.mean(y_pred)),
                "negative_predictions": int(np.sum(y_pred < 0)),
            }


        def evaluate_fitted_model(name: str, model: Pipeline, X: pd.DataFrame, y: pd.Series, split: str) -> Dict[str, Any]:
            predictions = model.predict(X)
            result = {"model": name, "split": split}
            result.update(regression_metrics(y, predictions))
            return result


        baseline_results = pd.DataFrame(
            [
                evaluate_fitted_model("company_linear_baseline", baseline_pipeline, X_train, y_train, "train"),
                evaluate_fitted_model("company_linear_baseline", baseline_pipeline, X_test, y_test, "test"),
            ]
        )
        display(baseline_results)
        '''
    ),
    md(
        """
        **Подвывод по baseline:** линейная модель компании дает `RMSE` около `412` на train и `411` на test, а `R2` около `0.59`. Это не случайная модель: она уже объясняет заметную часть спроса. Но у нее есть физически некорректное поведение - отрицательные прогнозы спроса. Значит новая модель должна улучшать не только численные метрики, но и здравый смысл прогноза.

        """
    ),
    md(
        """
        <a id="stage-4"></a>
        ## Этап 4. Первичный аудит и EDA

        EDA здесь нужен не для набора графиков, а для решений: как обрабатывать пропуски, какие признаки масштабировать, какие признаки кодировать, какие нелинейности проверить и какие выводы потом объяснять бизнесу.
        """
    ),
    code(
        r'''
        audit_rows = []
        for dataset_name, df in [("train", train), ("test", test)]:
            audit_rows.append(
                {
                    "dataset": dataset_name,
                    "rows": df.shape[0],
                    "columns": df.shape[1],
                    "duplicates": int(df.duplicated().sum()),
                    "target_min": float(df[TARGET].min()),
                    "target_median": float(df[TARGET].median()),
                    "target_mean": float(df[TARGET].mean()),
                    "target_max": float(df[TARGET].max()),
                    "zero_target_rows": int((df[TARGET] == 0).sum()),
                    "zero_target_share": float((df[TARGET] == 0).mean()),
                }
            )

        audit_overview = pd.DataFrame(audit_rows)
        missing_report = pd.concat(
            [
                train.isna().sum().rename("train_missing"),
                (train.isna().mean() * 100).rename("train_missing_pct"),
                test.isna().sum().rename("test_missing"),
                (test.isna().mean() * 100).rename("test_missing_pct"),
                train.dtypes.astype(str).rename("dtype"),
            ],
            axis=1,
        ).reset_index(names="column")

        category_report = []
        for column in CATEGORICAL_FEATURES + TIME_FEATURES:
            category_report.append(
                {
                    "column": column,
                    "train_unique": train[column].nunique(dropna=False),
                    "test_unique": test[column].nunique(dropna=False),
                    "train_values": sorted(map(str, train[column].dropna().unique().tolist())),
                    "test_values": sorted(map(str, test[column].dropna().unique().tolist())),
                }
            )

        print("Общий аудит")
        display(audit_overview)
        print("Пропуски и типы")
        display(missing_report)
        print("Категориальные и дискретные значения")
        display(pd.DataFrame(category_report))

        assert audit_overview["duplicates"].sum() == 0, "Unexpected duplicate rows"
        assert (train[TARGET] >= 0).all() and (test[TARGET] >= 0).all(), "Target contains negative values"
        '''
    ),
    md(
        """
        **Подвывод по аудиту:** дубликатов нет, целевая переменная неотрицательная, но есть нулевой спрос. Пропуски есть только в погодных числовых признаках и занимают ограниченную долю строк, поэтому удалять строки невыгодно. Корректнее обучать заполнение внутри pipeline только на train-фолдах.
        """
    ),
    code(
        r'''
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        sns.histplot(train[TARGET], bins=40, kde=True, ax=axes[0], color="#2f6f9f")
        axes[0].set_title("Распределение почасового спроса в train")
        axes[0].set_xlabel("Rented Bike Count, велосипедов в час")
        axes[0].set_ylabel("Количество наблюдений")

        sns.boxplot(x=train[TARGET], ax=axes[1], color="#8fbcd4")
        axes[1].set_title("Хвосты и возможные выбросы спроса")
        axes[1].set_xlabel("Rented Bike Count, велосипедов в час")
        plt.tight_layout()
        plt.show()

        target_summary = train[TARGET].describe(percentiles=[0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99]).to_frame("train_target")
        display(target_summary)
        '''
    ),
    md(
        """
        **Подвывод по target:** спрос сильно асимметричен: есть много часов с небольшим спросом и длинный правый хвост с пиковыми значениями. Удалять такие значения только по boxplot нельзя: для проката пики спроса являются бизнес-реальностью, а не автоматической ошибкой. Поэтому модели должны учиться на всем диапазоне спроса.
        """
    ),
    code(
        r'''
        continuous_labels = {
            "temperature": "Temperature, °C",
            "humidity": "Humidity, %",
            "wind_speed_ms": "Wind speed, m/s",
            "visibility_10m": "Visibility, 10 m units",
            "dew_point_temperature": "Dew point temperature, °C",
            "solar_radiation_mjm2": "Solar radiation, MJ/m2",
            "rainfallmm": "Rainfall, mm",
            "snowfall_cm": "Snowfall, cm",
        }

        fig, axes = plt.subplots(4, 2, figsize=(15, 16))
        axes = axes.ravel()
        for ax, column in zip(axes, BASE_NUMERIC_FEATURES):
            sns.histplot(train[column], bins=35, kde=True, ax=ax, color="#477998")
            ax.set_title(f"Распределение: {continuous_labels[column]}")
            ax.set_xlabel(continuous_labels[column])
            ax.set_ylabel("Количество наблюдений")
        plt.tight_layout()
        plt.show()

        fig, axes = plt.subplots(4, 2, figsize=(15, 16))
        axes = axes.ravel()
        sampled_train = train.sample(min(2500, len(train)), random_state=RANDOM_STATE)
        for ax, column in zip(axes, BASE_NUMERIC_FEATURES):
            sns.scatterplot(data=sampled_train, x=column, y=TARGET, alpha=0.35, ax=ax, color="#20639b")
            ax.set_title(f"Спрос и {continuous_labels[column]}")
            ax.set_xlabel(continuous_labels[column])
            ax.set_ylabel("Rented Bike Count, велосипедов в час")
        plt.tight_layout()
        plt.show()

        numeric_corr = (
            train[BASE_NUMERIC_FEATURES + [TARGET]]
            .corr(method="spearman", numeric_only=True)[TARGET]
            .drop(TARGET)
            .sort_values(key=lambda values: values.abs(), ascending=False)
            .reset_index()
            .rename(columns={"index": "feature", TARGET: "spearman_corr_with_target"})
        )
        display(numeric_corr)
        '''
    ),
    md(
        """
        **Подвывод по погодным признакам:** сильнее всего со спросом связан температурный блок: `temperature`, `dew_point_temperature`, `solar_radiation_mjm2`, а влажность имеет отрицательную связь. Scatter-графики показывают нелинейность: рост температуры помогает спросу не одинаково на всем диапазоне, а осадки и снег скорее режут спрос. Это обосновывает KNN/tree и дополнительные weather interaction признаки.
        """
    ),
    code(
        r'''
        cat_target_tables = []
        for column in CATEGORICAL_FEATURES + TIME_FEATURES:
            summary = (
                train.groupby(column, dropna=False)[TARGET]
                .agg(count="count", mean_demand="mean", median_demand="median")
                .reset_index()
                .sort_values("mean_demand", ascending=False)
            )
            summary.insert(0, "feature", column)
            cat_target_tables.append(summary.rename(columns={column: "value"}))

        cat_target_summary = pd.concat(cat_target_tables, ignore_index=True)
        display(cat_target_summary)

        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        plot_columns = ["seasons", "holiday", "functioning_day", "time_period_evening"]
        for ax, column in zip(axes.ravel(), plot_columns):
            order = train.groupby(column)[TARGET].mean().sort_values(ascending=False).index
            sns.barplot(data=train, x=column, y=TARGET, order=order, estimator="mean", errorbar=None, ax=ax, color="#4f8a8b")
            ax.set_title(f"Средний спрос по {column}")
            ax.set_xlabel(column)
            ax.set_ylabel("Средний Rented Bike Count, велосипедов в час")
            ax.tick_params(axis="x", rotation=20)
        plt.tight_layout()
        plt.show()
        '''
    ),
    md(
        """
        **Подвывод по категориальным и временным признакам:** `seasons` и `functioning_day` дают сильный сигнал. Если прокат не функционирует, спрос равен нулю - это не выброс и не ошибка, а режимное состояние. Time-period признаки тоже стоит сохранить: спрос по часам суток отличается, а скрытый `Daytime` можно восстановить как случай, когда все четыре time-period dummy равны `False`.
        """
    ),
    code(
        r'''
        eda_decisions = pd.DataFrame(
            [
                {
                    "EDA observation": "Погодные числовые признаки имеют пропуски, но доля пропусков ограничена.",
                    "Modeling decision": "Не удалять строки; использовать SimpleImputer(strategy='median') внутри pipeline.",
                },
                {
                    "EDA observation": "Категориальные признаки имеют небольшое число уровней.",
                    "Modeling decision": "Использовать SimpleImputer(strategy='most_frequent') и OneHotEncoder(handle_unknown='ignore', drop='first').",
                },
                {
                    "EDA observation": "Time_Period_Daytime скрыт как строка, где все time-period dummy равны False.",
                    "Modeling decision": "Добавить `time_period_daytime` в кастомном transformer.",
                },
                {
                    "EDA observation": "Температура, влажность, солнечная радиация и осадки связаны со спросом нелинейно.",
                    "Modeling decision": "Обучить KNN и Decision Tree; добавить weather interaction признаки.",
                },
                {
                    "EDA observation": "Baseline линейной регрессии дает отрицательные прогнозы.",
                    "Modeling decision": "Проверять диапазон предсказаний финальной модели и считать negative_predictions.",
                },
                {
                    "EDA observation": "Хвосты спроса выглядят как реальные пики, а не технические ошибки.",
                    "Modeling decision": "Не удалять target outliers механически; оценивать RMSE и MAE вместе.",
                },
            ]
        )
        display(eda_decisions)
        '''
    ),
    md(
        """
        **Вывод этапа 4:** данные пригодны для моделирования, но требуют аккуратного pipeline. Главные решения после EDA: не удалять строки с пропусками, не чистить пики спроса механически, восстановить daytime, добавить погодные взаимодействия и обязательно контролировать физический диапазон прогнозов.

        """
    ),
    md(
        """
        <a id="stage-5"></a>
        ## Этап 5. Pipeline и feature engineering

        Теперь превращаю решения EDA в код. Кастомный transformer нужен не для галочки: он гарантирует стабильную схему признаков, восстанавливает скрытый `Daytime` и добавляет погодные взаимодействия без доступа к target. Transformer вынесен в `bike_demand_pipeline_components.py`, чтобы сохраненный `joblib`-pipeline открывался в чистом Python-процессе.
        """
    ),
    code(
        r'''
        from bike_demand_pipeline_components import (
            BikeFeatureEngineer,
            CATEGORICAL_FEATURES,
            ENGINEERED_FEATURES,
            MODEL_FEATURES_AFTER_ENGINEERING,
            NUMERIC_WITH_ENGINEERED,
        )


        def make_preprocessor(scale_numeric: bool) -> ColumnTransformer:
            numeric_steps: List[Tuple[str, Any]] = [("imputer", SimpleImputer(strategy="median"))]
            if scale_numeric:
                numeric_steps.append(("scaler", StandardScaler()))

            numeric_pipeline = Pipeline(numeric_steps)
            categorical_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    (
                        "onehot",
                        OneHotEncoder(
                            handle_unknown="ignore",
                            drop="first",
                            sparse_output=False,
                        ),
                    ),
                ]
            )
            return ColumnTransformer(
                transformers=[
                    ("num", numeric_pipeline, NUMERIC_WITH_ENGINEERED),
                    ("cat", categorical_pipeline, CATEGORICAL_FEATURES),
                ],
                remainder="drop",
                verbose_feature_names_out=True,
            )


        def make_model_pipeline(model: BaseEstimator, scale_numeric: bool) -> Pipeline:
            return Pipeline(
                steps=[
                    ("feature_engineering", BikeFeatureEngineer()),
                    ("preprocessor", make_preprocessor(scale_numeric=scale_numeric)),
                    ("model", model),
                ]
            )


        schema_check = BikeFeatureEngineer().fit_transform(X_train.head(5))
        display(schema_check)
        assert list(schema_check.columns) == MODEL_FEATURES_AFTER_ENGINEERING
        assert schema_check.shape[1] == len(MODEL_FEATURES_AFTER_ENGINEERING)
        '''
    ),
    md(
        """
        **Подвывод по pipeline:** кастомный transformer не обучается на target и не смотрит в test. Он только стабилизирует схему и добавляет признаки, которые прямо следуют из EDA: скрытый daytime, осадки, комфортная температура и взаимодействия погоды. Для KNN числовые признаки масштабируются, для дерева масштабирование не требуется.

        """
    ),
    md(
        """
        <a id="stage-6"></a>
        ## Этап 6. Optuna и 5-fold CV

        Сначала ставлю нижнюю границу через `DummyRegressor`, затем подбираю гиперпараметры для KNN и Decision Tree. Для всех новых моделей используется один и тот же CV-протокол, чтобы сравнение было честным.
        """
    ),
    code(
        r'''
        cv = KFold(n_splits=CV_SPLITS, shuffle=True, random_state=RANDOM_STATE)
        SCORING = {
            "rmse": "neg_root_mean_squared_error",
            "mae": "neg_mean_absolute_error",
            "r2": "r2",
        }


        def summarize_cv_scores(model_name: str, scores: Dict[str, np.ndarray], params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
            return {
                "model": model_name,
                "cv_RMSE_mean": -float(scores["test_rmse"].mean()),
                "cv_RMSE_std": float(scores["test_rmse"].std()),
                "cv_MAE_mean": -float(scores["test_mae"].mean()),
                "cv_R2_mean": float(scores["test_r2"].mean()),
                "fit_time_mean_sec": float(scores["fit_time"].mean()),
                "score_time_mean_sec": float(scores["score_time"].mean()),
                "params": params or {},
            }


        def cross_validate_pipeline(model_name: str, pipeline: Pipeline, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
            scores = cross_validate(
                pipeline,
                X_train,
                y_train,
                scoring=SCORING,
                cv=cv,
                n_jobs=1,
                return_train_score=False,
            )
            return summarize_cv_scores(model_name, scores, params=params)


        dummy_pipeline = Pipeline(steps=[("model", DummyRegressor(strategy="mean"))])
        dummy_cv_result = cross_validate_pipeline("dummy_mean", dummy_pipeline)
        display(pd.DataFrame([dummy_cv_result]).drop(columns="params"))
        '''
    ),
    md(
        """
        **Подвывод по DummyRegressor:** наивная модель нужна как нижняя граница. Она не использует признаки и не требует препроцессора. Если KNN/tree не обгоняют этот уровень, значит вся инженерия признаков бессмысленна.
        """
    ),
    code(
        r'''
        def objective_knn(trial: optuna.Trial) -> float:
            params = {
                "n_neighbors": trial.suggest_int("n_neighbors", 3, 80),
                "weights": trial.suggest_categorical("weights", ["uniform", "distance"]),
                "p": trial.suggest_int("p", 1, 2),
                "leaf_size": trial.suggest_int("leaf_size", 10, 60),
            }
            pipeline = make_model_pipeline(KNeighborsRegressor(**params), scale_numeric=True)
            scores = cross_validate(
                pipeline,
                X_train,
                y_train,
                scoring=SCORING,
                cv=cv,
                n_jobs=1,
                return_train_score=False,
            )
            trial.set_user_attr("mae", -float(scores["test_mae"].mean()))
            trial.set_user_attr("r2", float(scores["test_r2"].mean()))
            trial.set_user_attr("rmse_std", float(scores["test_rmse"].std()))
            return -float(scores["test_rmse"].mean())


        def objective_tree(trial: optuna.Trial) -> float:
            params = {
                "max_depth": trial.suggest_categorical("max_depth", [None, 3, 5, 7, 10, 15, 20, 30]),
                "min_samples_split": trial.suggest_int("min_samples_split", 2, 80),
                "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 50),
                "max_features": trial.suggest_categorical("max_features", [None, "sqrt", "log2", 0.5, 0.8, 1.0]),
                "ccp_alpha": trial.suggest_float("ccp_alpha", 1e-8, 1e-2, log=True),
                "random_state": RANDOM_STATE,
            }
            pipeline = make_model_pipeline(DecisionTreeRegressor(**params), scale_numeric=False)
            scores = cross_validate(
                pipeline,
                X_train,
                y_train,
                scoring=SCORING,
                cv=cv,
                n_jobs=1,
                return_train_score=False,
            )
            trial.set_user_attr("mae", -float(scores["test_mae"].mean()))
            trial.set_user_attr("r2", float(scores["test_r2"].mean()))
            trial.set_user_attr("rmse_std", float(scores["test_rmse"].std()))
            return -float(scores["test_rmse"].mean())


        study_timings = {}

        start = time.perf_counter()
        knn_study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE))
        knn_study.optimize(objective_knn, n_trials=N_TRIALS_KNN, show_progress_bar=False)
        study_timings["KNN"] = time.perf_counter() - start

        start = time.perf_counter()
        tree_study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE))
        tree_study.optimize(objective_tree, n_trials=N_TRIALS_TREE, show_progress_bar=False)
        study_timings["DecisionTree"] = time.perf_counter() - start

        study_summary = pd.DataFrame(
            [
                {
                    "model": "KNN",
                    "trials": len(knn_study.trials),
                    "best_RMSE": knn_study.best_value,
                    "best_MAE": knn_study.best_trial.user_attrs["mae"],
                    "best_R2": knn_study.best_trial.user_attrs["r2"],
                    "best_RMSE_std": knn_study.best_trial.user_attrs["rmse_std"],
                    "time_sec": study_timings["KNN"],
                    "best_params": knn_study.best_params,
                },
                {
                    "model": "DecisionTree",
                    "trials": len(tree_study.trials),
                    "best_RMSE": tree_study.best_value,
                    "best_MAE": tree_study.best_trial.user_attrs["mae"],
                    "best_R2": tree_study.best_trial.user_attrs["r2"],
                    "best_RMSE_std": tree_study.best_trial.user_attrs["rmse_std"],
                    "time_sec": study_timings["DecisionTree"],
                    "best_params": tree_study.best_params,
                },
            ]
        )
        display(study_summary)
        '''
    ),
    code(
        r'''
        knn_best_pipeline = make_model_pipeline(KNeighborsRegressor(**knn_study.best_params), scale_numeric=True)
        tree_best_params = dict(tree_study.best_params)
        tree_best_params["random_state"] = RANDOM_STATE
        tree_best_pipeline = make_model_pipeline(DecisionTreeRegressor(**tree_best_params), scale_numeric=False)

        cv_results = [dummy_cv_result]
        cv_results.append(cross_validate_pipeline("knn_optuna", knn_best_pipeline, params=knn_study.best_params))
        cv_results.append(cross_validate_pipeline("decision_tree_optuna", tree_best_pipeline, params=tree_best_params))

        cv_comparison = pd.DataFrame(cv_results).sort_values("cv_RMSE_mean")
        display(cv_comparison.drop(columns="params"))

        fig, axes = plt.subplots(1, 3, figsize=(17, 5))
        plot_df = cv_comparison.copy()
        sns.barplot(data=plot_df, x="model", y="cv_RMSE_mean", ax=axes[0], color="#49759c")
        axes[0].set_title("CV RMSE: ниже лучше")
        axes[0].set_xlabel("Модель")
        axes[0].set_ylabel("RMSE, велосипедов в час")
        axes[0].tick_params(axis="x", rotation=20)

        sns.barplot(data=plot_df, x="model", y="cv_MAE_mean", ax=axes[1], color="#7aa95c")
        axes[1].set_title("CV MAE: ниже лучше")
        axes[1].set_xlabel("Модель")
        axes[1].set_ylabel("MAE, велосипедов в час")
        axes[1].tick_params(axis="x", rotation=20)

        sns.barplot(data=plot_df, x="model", y="cv_R2_mean", ax=axes[2], color="#c98256")
        axes[2].set_title("CV R2: выше лучше")
        axes[2].set_xlabel("Модель")
        axes[2].set_ylabel("R2")
        axes[2].tick_params(axis="x", rotation=20)
        plt.tight_layout()
        plt.show()
        '''
    ),
    code(
        r'''
        def check_boundary_params(study: optuna.Study, model_name: str) -> pd.DataFrame:
            best = study.best_params
            if model_name == "KNN":
                ranges = {"n_neighbors": (3, 80), "leaf_size": (10, 60), "p": (1, 2)}
            else:
                ranges = {"min_samples_split": (2, 80), "min_samples_leaf": (1, 50)}
            rows = []
            for param, (low, high) in ranges.items():
                value = best.get(param)
                rows.append(
                    {
                        "model": model_name,
                        "param": param,
                        "best_value": value,
                        "search_low": low,
                        "search_high": high,
                        "on_boundary": value in {low, high},
                    }
                )
            return pd.DataFrame(rows)


        boundary_check = pd.concat(
            [
                check_boundary_params(knn_study, "KNN"),
                check_boundary_params(tree_study, "DecisionTree"),
            ],
            ignore_index=True,
        )
        display(boundary_check)
        '''
    ),
    md(
        """
        **Вывод этапа 6:** новые модели сравнивались по одинаковому протоколу 5-fold CV на train. Основной выбор делается по среднему `RMSE`, но рядом проверяются `MAE`, `R2`, разброс по фолдам и границы поиска Optuna. Это защищает от ситуации, когда модель выбрана по случайному удачному фолду или по слишком узкой сетке.

        """
    ),
    md(
        """
        <a id="stage-7"></a>
        ## Этап 7. Финальная проверка на test

        Теперь выбираю модель по train CV, обучаю ее на всем train и один раз применяю к test. Это финальная проверка качества на данных, которые не участвовали в подборе гиперпараметров.
        """
    ),
    code(
        r'''
        best_cv_row = cv_comparison.iloc[0]
        best_model_name = best_cv_row["model"]

        if best_model_name == "knn_optuna":
            final_pipeline = knn_best_pipeline
            final_params = knn_study.best_params
        elif best_model_name == "decision_tree_optuna":
            final_pipeline = tree_best_pipeline
            final_params = tree_best_params
        else:
            raise RuntimeError("Dummy model should not be selected as final model")

        final_pipeline.fit(X_train, y_train)
        final_test_predictions = final_pipeline.predict(X_test)
        final_train_predictions = final_pipeline.predict(X_train)

        final_results = pd.DataFrame(
            [
                evaluate_fitted_model("company_linear_baseline", baseline_pipeline, X_test, y_test, "test"),
                {
                    "model": best_model_name,
                    "split": "test",
                    **regression_metrics(y_test, final_test_predictions),
                },
                {
                    "model": best_model_name,
                    "split": "train_fit_reference",
                    **regression_metrics(y_train, final_train_predictions),
                },
            ]
        )

        baseline_test_rmse = final_results.query("model == 'company_linear_baseline' and split == 'test'")["RMSE"].iloc[0]
        final_test_rmse = final_results.query("model == @best_model_name and split == 'test'")["RMSE"].iloc[0]
        rmse_improvement_pct = (baseline_test_rmse - final_test_rmse) / baseline_test_rmse * 100

        display(final_results)
        print(f"Выбранная модель: {best_model_name}")
        print(f"Улучшение RMSE относительно baseline на test: {rmse_improvement_pct:.2f}%")
        print(f"Параметры финальной модели: {final_params}")
        '''
    ),
    code(
        r'''
        residuals = y_test - final_test_predictions
        baseline_test_predictions = baseline_pipeline.predict(X_test)

        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        sns.scatterplot(x=y_test, y=final_test_predictions, alpha=0.55, ax=axes[0], color="#2f6f9f")
        max_value = max(y_test.max(), final_test_predictions.max())
        axes[0].plot([0, max_value], [0, max_value], color="black", linestyle="--", linewidth=1)
        axes[0].set_title("Финальная модель: факт против прогноза")
        axes[0].set_xlabel("Фактический спрос, велосипедов в час")
        axes[0].set_ylabel("Прогноз, велосипедов в час")

        sns.histplot(residuals, bins=35, kde=True, ax=axes[1], color="#7aa95c")
        axes[1].set_title("Распределение ошибок финальной модели")
        axes[1].set_xlabel("Ошибка y_true - y_pred, велосипедов в час")
        axes[1].set_ylabel("Количество наблюдений")

        comparison_plot = pd.DataFrame(
            {
                "actual": y_test,
                "baseline_prediction": baseline_test_predictions,
                "final_prediction": final_test_predictions,
            }
        ).sample(min(500, len(y_test)), random_state=RANDOM_STATE)
        sns.scatterplot(data=comparison_plot, x="actual", y="baseline_prediction", alpha=0.35, label="baseline", ax=axes[2])
        sns.scatterplot(data=comparison_plot, x="actual", y="final_prediction", alpha=0.35, label=best_model_name, ax=axes[2])
        axes[2].plot([0, max_value], [0, max_value], color="black", linestyle="--", linewidth=1)
        axes[2].set_title("Baseline и финальная модель на test")
        axes[2].set_xlabel("Фактический спрос, велосипедов в час")
        axes[2].set_ylabel("Прогноз, велосипедов в час")
        axes[2].legend()

        plt.tight_layout()
        plt.show()
        '''
    ),
    md(
        """
        **Вывод этапа 7:** финальная проверка выполнена только после выбора модели по CV. В таблице выше baseline и выбранная модель сравниваются на одной test-выборке. Отдельно контролируется физический смысл прогноза: для спроса нежелательны отрицательные значения, даже если средняя ошибка выглядит приемлемо.

        """
    ),
    md(
        """
        <a id="stage-8"></a>
        ## Этап 8. Интерпретация и артефакты

        Интерпретация нужна, чтобы результат был полезен заказчику: модель должна не только предсказывать спрос, но и объяснять, какие факторы стоит мониторить в операционном планировании проката.
        """
    ),
    code(
        r'''
        if best_model_name == "decision_tree_optuna":
            feature_names = final_pipeline.named_steps["preprocessor"].get_feature_names_out()
            importances = final_pipeline.named_steps["model"].feature_importances_
            importance_table = (
                pd.DataFrame({"feature": feature_names, "importance": importances})
                .sort_values("importance", ascending=False)
                .head(20)
            )
            importance_title = "Важность признаков финального дерева"
        else:
            permutation = permutation_importance(
                final_pipeline,
                X_test,
                y_test,
                scoring="neg_root_mean_squared_error",
                n_repeats=10,
                random_state=RANDOM_STATE,
                n_jobs=1,
            )
            importance_table = (
                pd.DataFrame(
                    {
                        "feature": X_test.columns,
                        "importance": permutation.importances_mean,
                        "importance_std": permutation.importances_std,
                    }
                )
                .sort_values("importance", ascending=False)
                .head(20)
            )
            importance_title = "Permutation importance финальной KNN-модели"

        importance_table["feature"] = importance_table["feature"].astype(str)
        display(importance_table)

        fig, ax = plt.subplots(figsize=(11, 7))
        sns.barplot(data=importance_table, y="feature", x="importance", ax=ax, color="#49759c")
        ax.set_title(importance_title)
        ax.set_xlabel("Вклад признака в качество модели")
        ax.set_ylabel("Признак")
        plt.tight_layout()
        plt.show()
        '''
    ),
    code(
        r'''
        model_artifact_path = MODELS_DIR / "bike_demand_model.joblib"
        metadata_path = MODELS_DIR / "bike_demand_model_metadata.json"
        predictions_path = MODELS_DIR / "bike_demand_test_predictions.csv"
        model_card_path = MODELS_DIR / "bike_demand_model_card.json"
        manifest_path = MODELS_DIR / "bike_demand_artifact_manifest.json"
        artifact_inventory_path = MODELS_DIR / "bike_demand_artifact_inventory.csv"

        joblib.dump(final_pipeline, model_artifact_path)

        predictions_frame = X_test.copy()
        predictions_frame[TARGET] = y_test.values
        predictions_frame["prediction"] = final_test_predictions
        predictions_frame["residual"] = y_test.values - final_test_predictions
        predictions_frame.to_csv(predictions_path, index=False)

        def file_sha256(path: Path) -> str:
            digest = hashlib.sha256()
            with path.open("rb") as file:
                for chunk in iter(lambda: file.read(1024 * 1024), b""):
                    digest.update(chunk)
            return digest.hexdigest()

        required_component_names = [
            "BikeFeatureEngineer",
            "BASE_NUMERIC_FEATURES",
            "CATEGORICAL_FEATURES",
            "TIME_FEATURES",
            "BASE_FEATURES",
            "ENGINEERED_FEATURES",
            "NUMERIC_WITH_ENGINEERED",
            "MODEL_FEATURES_AFTER_ENGINEERING",
        ]
        component_module = importlib.import_module(COMPONENT_MODULE_NAME)
        component_symbol_check = pd.DataFrame(
            [
                {"required_name": name, "present": hasattr(component_module, name)}
                for name in required_component_names
            ]
        )
        component_symbols_ok = bool(component_symbol_check["present"].all())
        assert component_symbols_ok, "Feature engineering component module is incomplete"

        component_source_sha256 = file_sha256(COMPONENT_MODULE_PATH)

        baseline_test_metrics = final_results.query(
            "model == 'company_linear_baseline' and split == 'test'"
        ).iloc[0].to_dict()
        final_test_metrics = final_results.query(
            "model == @best_model_name and split == 'test'"
        ).iloc[0].to_dict()

        model_card = {
            "project": "bike_demand_regression",
            "business_goal": "forecast hourly bike rental demand for operational planning in BikeSochi",
            "target": TARGET,
            "primary_metric": "RMSE",
            "secondary_metrics": ["MAE", "R2", "negative_predictions"],
            "selected_model": best_model_name,
            "selected_params": final_params,
            "test_quality": final_test_metrics,
            "baseline_test_quality": baseline_test_metrics,
            "rmse_improvement_pct_vs_baseline": float(rmse_improvement_pct),
            "training_protocol": {
                "model_selection": f"{CV_SPLITS}-fold CV on train only",
                "test_usage": "one final evaluation after model selection",
                "random_state": RANDOM_STATE,
                "optuna_trials": {"knn": N_TRIALS_KNN, "decision_tree": N_TRIALS_TREE},
            },
            "input_contract": {
                "required_columns": X_train.columns.tolist(),
                "target_column": TARGET,
                "row_grain": "one row = one hour of bike rental observations",
                "not_required_at_inference": [TARGET],
            },
            "feature_engineering_contract": {
                "module": COMPONENT_MODULE_NAME,
                "source_path": str(COMPONENT_MODULE_PATH.relative_to(PROJECT_ROOT)),
                "source_sha256": component_source_sha256,
                "required_names": required_component_names,
            },
            "known_limitations": [
                "test sample is from the same source distribution as train, not a future out-of-time period",
                "the model should be revalidated before use in unusual weather, holidays, or new operating regimes",
                "zero-demand functioning-day periods and non-functioning-day periods require separate monitoring",
            ],
            "monitoring_recommendations": [
                "RMSE, MAE, R2 and negative prediction count on fresh labeled batches",
                "share of zero-demand hours and non-functioning-day rows",
                "distribution drift for temperature, humidity, rainfall, snowfall and time-period features",
                "prediction error by season, hour, holiday and functioning_day",
            ],
        }

        component_manifest = {
            "component_module": COMPONENT_MODULE_NAME,
            "component_source_path": str(COMPONENT_MODULE_PATH),
            "component_source_sha256": component_source_sha256,
            "required_component_names": required_component_names,
            "component_symbols_ok": component_symbols_ok,
            "pipeline_steps": [name for name, _ in final_pipeline.steps],
            "preprocessor_transformers": [
                name for name, _, _ in final_pipeline.named_steps["preprocessor"].transformers
            ],
            "model_class": type(final_pipeline.named_steps["model"]).__name__,
            "model_params": final_params,
        }

        metadata = {
            "project": "bike_demand_regression",
            "random_state": RANDOM_STATE,
            "cv_splits": CV_SPLITS,
            "selected_model": best_model_name,
            "selected_params": final_params,
            "baseline_test_metrics": baseline_test_metrics,
            "final_test_metrics": final_test_metrics,
            "rmse_improvement_pct_vs_baseline": rmse_improvement_pct,
            "train_shape": train.shape,
            "test_shape": test.shape,
            "input_columns": X_train.columns.tolist(),
            "engineered_features": ENGINEERED_FEATURES,
            "component_module": COMPONENT_MODULE_NAME,
            "component_source_sha256": component_source_sha256,
            "required_component_names": required_component_names,
            "versions": versions.to_dict(orient="records"),
            "artifact_paths": {
                "model": str(model_artifact_path),
                "metadata": str(metadata_path),
                "test_predictions": str(predictions_path),
                "model_card": str(model_card_path),
                "manifest": str(manifest_path),
                "artifact_inventory": str(artifact_inventory_path),
                "component_source_module": str(COMPONENT_MODULE_PATH),
                "baseline": str(baseline_path),
            },
        }

        with metadata_path.open("w", encoding="utf-8") as file:
            json.dump(metadata, file, ensure_ascii=False, indent=2, default=str)
        with model_card_path.open("w", encoding="utf-8") as file:
            json.dump(model_card, file, ensure_ascii=False, indent=2, default=str)
        with manifest_path.open("w", encoding="utf-8") as file:
            json.dump(component_manifest, file, ensure_ascii=False, indent=2, default=str)

        production_contract = pd.DataFrame(
            [
                {
                    "contract_area": "input_schema",
                    "requirement": "inference data must contain the same raw feature columns as train",
                    "implementation": f"{len(X_train.columns)} input columns are listed in model_card['input_contract']",
                },
                {
                    "contract_area": "feature_engineering_code",
                    "requirement": "saved model must be paired with all custom feature engineering code",
                    "implementation": f"{COMPONENT_MODULE_NAME}.py is tracked; sha256={component_source_sha256[:12]}...",
                },
                {
                    "contract_area": "reproducibility",
                    "requirement": "artifact predictions after reload must match notebook predictions",
                    "implementation": "joblib.load is executed below and compared with np.allclose",
                },
                {
                    "contract_area": "monitoring",
                    "requirement": "production use requires fresh labeled checks and drift monitoring",
                    "implementation": "model card contains metric, segment and feature-drift monitoring recommendations",
                },
            ]
        )

        artifact_inventory = pd.DataFrame(
            [
                {
                    "artifact": "model_pipeline",
                    "path": str(model_artifact_path.relative_to(PROJECT_ROOT)),
                    "exists": model_artifact_path.exists(),
                    "purpose": "full sklearn pipeline with feature engineering, preprocessing and model",
                },
                {
                    "artifact": "metadata",
                    "path": str(metadata_path.relative_to(PROJECT_ROOT)),
                    "exists": metadata_path.exists(),
                    "purpose": "run metrics, selected params, package versions and artifact paths",
                },
                {
                    "artifact": "model_card",
                    "path": str(model_card_path.relative_to(PROJECT_ROOT)),
                    "exists": model_card_path.exists(),
                    "purpose": "business goal, quality, input contract, limits and monitoring",
                },
                {
                    "artifact": "component_manifest",
                    "path": str(manifest_path.relative_to(PROJECT_ROOT)),
                    "exists": manifest_path.exists(),
                    "purpose": "custom component names, module checksum and pipeline structure",
                },
                {
                    "artifact": "test_predictions",
                    "path": str(predictions_path.relative_to(PROJECT_ROOT)),
                    "exists": predictions_path.exists(),
                    "purpose": "row-level final test predictions and residuals",
                },
                {
                    "artifact": "component_source_module",
                    "path": str(COMPONENT_MODULE_PATH.relative_to(PROJECT_ROOT)),
                    "exists": COMPONENT_MODULE_PATH.exists(),
                    "purpose": "Python code required to load and run the saved pipeline",
                },
                {
                    "artifact": "baseline_pipeline",
                    "path": str(baseline_path.relative_to(PROJECT_ROOT)),
                    "exists": baseline_path.exists(),
                    "purpose": "company baseline used for test comparison",
                },
            ]
        )
        artifact_inventory.to_csv(artifact_inventory_path, index=False)

        reloaded_pipeline = joblib.load(model_artifact_path)
        reloaded_predictions = reloaded_pipeline.predict(X_test)
        reload_check = {
            "same_predictions_after_reload": bool(np.allclose(final_test_predictions, reloaded_predictions)),
            "max_abs_prediction_diff": float(np.max(np.abs(final_test_predictions - reloaded_predictions))),
        }
        artifact_check = pd.DataFrame(
            [
                {
                    "check": "component module contains required names",
                    "status": "OK" if component_symbols_ok else "FAIL",
                    "detail": f"{component_symbol_check['present'].sum()} of {len(required_component_names)} names found",
                },
                {
                    "check": "all listed artifacts exist",
                    "status": "OK" if bool(artifact_inventory["exists"].all()) else "FAIL",
                    "detail": f"{artifact_inventory['exists'].sum()} of {len(artifact_inventory)} artifacts found",
                },
                {
                    "check": "predictions match after joblib reload",
                    "status": "OK" if reload_check["same_predictions_after_reload"] else "FAIL",
                    "detail": f"max_abs_prediction_diff={reload_check['max_abs_prediction_diff']:.10f}",
                },
            ]
        )

        display(production_contract)
        display(artifact_inventory)
        display(component_symbol_check)
        display(artifact_check)
        display(pd.DataFrame([reload_check]))
        assert reload_check["same_predictions_after_reload"], "Reloaded model predictions differ"
        assert artifact_inventory["exists"].all(), "Some declared artifacts were not saved"
        '''
    ),
    md(
        """
        **Вывод этапа 8:** сохранен не только алгоритм, а полный рабочий pipeline: feature engineering, preprocessors и модель. К нему привязаны metadata, model card, manifest компонентов, test predictions и inventory артефактов. Кастомный transformer вынесен в импортируемый модуль `bike_demand_pipeline_components.py`, его обязательные имена и checksum зафиксированы. После `joblib.load()` предсказания совпадают, значит артефакт можно передавать дальше без скрытой зависимости от состояния ноутбука.

        Test predictions сохранены только как отчетный артефакт финальной оценки.
        """
    ),
    code(
        r'''
        baseline_test_row = final_results.query("model == 'company_linear_baseline' and split == 'test'").iloc[0]
        final_test_row = final_results.query("model == @best_model_name and split == 'test'").iloc[0]
        final_cv_row = cv_comparison.query("model == @best_model_name").iloc[0]

        def readable_feature_name(feature: str) -> str:
            return (
                feature.replace("num__", "")
                .replace("cat__", "")
                .replace("_", " ")
            )

        top_feature_text = ", ".join(
            readable_feature_name(feature)
            for feature in importance_table["feature"].head(5).astype(str).tolist()
        )
        rmse_abs_improvement = baseline_test_row["RMSE"] - final_test_row["RMSE"]
        mae_abs_improvement = baseline_test_row["MAE"] - final_test_row["MAE"]
        r2_abs_improvement = final_test_row["R2"] - baseline_test_row["R2"]

        final_conclusion = f"""
        <a id="final-conclusions"></a>

        # Финальные выводы

        **Выбранная модель:** `{best_model_name}`.

        **Качество baseline на test:** RMSE = `{baseline_test_row["RMSE"]:.2f}`, MAE = `{baseline_test_row["MAE"]:.2f}`, R2 = `{baseline_test_row["R2"]:.3f}`. Минимальный прогноз baseline = `{baseline_test_row["prediction_min"]:.2f}`, отрицательных прогнозов = `{int(baseline_test_row["negative_predictions"])}`.

        **Качество выбранной модели на CV train:** RMSE = `{final_cv_row["cv_RMSE_mean"]:.2f}` ± `{final_cv_row["cv_RMSE_std"]:.2f}`, MAE = `{final_cv_row["cv_MAE_mean"]:.2f}`, R2 = `{final_cv_row["cv_R2_mean"]:.3f}`.

        **Качество выбранной модели на финальном test:** RMSE = `{final_test_row["RMSE"]:.2f}`, MAE = `{final_test_row["MAE"]:.2f}`, R2 = `{final_test_row["R2"]:.3f}`. По сравнению с baseline RMSE ниже на `{rmse_abs_improvement:.2f}` велосипеда в час (`{rmse_improvement_pct:.2f}%`), MAE ниже на `{mae_abs_improvement:.2f}`, а R2 выше на `{r2_abs_improvement:.3f}`. Минимальный прогноз = `{final_test_row["prediction_min"]:.2f}`, отрицательных прогнозов = `{int(final_test_row["negative_predictions"])}`.

        **Параметры финальной модели:** `{final_params}`.

        **Самые заметные признаки по интерпретации:** {top_feature_text}.

        Практический вывод для BikeSochi: выбранное дерево решений заметно сильнее линейного baseline на финальном test и исправляет его главный физический дефект - отрицательные прогнозы спроса. При планировании нужно учитывать не только температуру, но и ночной/вечерний период, разницу температуры и точки росы, режим работы проката, осадки, влажность и солнечную радиацию. Перед промышленным запуском следует проверить модель на более позднем периоде и отдельно мониторить качество в экстремальную погоду и в сезонные пики.
        """

        display(Markdown(final_conclusion))
        '''
    ),
]

nb.cells = base_cells + cells
nb.metadata["kernelspec"] = {
    "display_name": "Python (ds-sonv-bike-regression)",
    "language": "python",
    "name": "python3",
}
nb.metadata.setdefault("language_info", {})
nb.metadata["language_info"].update({"name": "python", "pygments_lexer": "ipython3"})
nb.nbformat_minor = 5

nbf.write(nb, path)
print(f"Wrote {path} with {len(nb.cells)} cells; preserved {len(base_cells)} original cells.")
