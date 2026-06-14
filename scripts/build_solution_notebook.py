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
component_module_source = Path("bike_demand_pipeline_components.py").read_text(encoding="utf-8")
if "'''" in component_module_source:
    raise ValueError("Component module source cannot contain triple single quotes")

component_bootstrap_code = (
    "COMPONENT_MODULE_SOURCE = '''\\\n"
    f"{component_module_source.rstrip()}\n"
    "'''\n"
    + r'''

previous_component_source = (
    COMPONENT_MODULE_PATH.read_text(encoding="utf-8")
    if COMPONENT_MODULE_PATH.exists()
    else None
)
component_status = "unchanged"
if previous_component_source != COMPONENT_MODULE_SOURCE:
    COMPONENT_MODULE_PATH.write_text(COMPONENT_MODULE_SOURCE, encoding="utf-8")
    component_status = "created" if previous_component_source is None else "updated"

importlib.invalidate_caches()
sys.modules.pop(COMPONENT_MODULE_NAME, None)
component_bootstrap = pd.DataFrame(
    [
        {
            "component_module": COMPONENT_MODULE_NAME,
            "path": str(COMPONENT_MODULE_PATH),
            "status": component_status,
            "source_sha256": hashlib.sha256(
                COMPONENT_MODULE_SOURCE.encode("utf-8")
            ).hexdigest(),
            "source_bytes": len(COMPONENT_MODULE_SOURCE.encode("utf-8")),
        }
    ]
)
display(component_bootstrap)
'''
)

cells = [
    md(
        """
        <a id="author-solution"></a>
        # Авторское решение: прогнозирование спроса на велопрокат BikeSouth

        Смысл задачи простой: BikeSouth нужно заранее понимать, сколько прокатов велосипедов ожидается в конкретный час. Если прогноз завышен, компания зря держит лишний парк и смены. Если занижен, в пиковые часы людям может не хватить доступных велосипедов.

        **Цель:** выбрать модель, которая лучше текущей линейной регрессии прогнозирует `Rented Bike Count` и не выдает невозможный отрицательный спрос.

        **Основная метрика:** `RMSE`, потому что крупные промахи в пиковые часы для проката особенно дороги. Дополнительно считаю `MAE` и `R2`: `MAE` показывает типичную ошибку в прокатах/час, а `R2` помогает понять, насколько модель объясняет изменчивость спроса относительно простого среднего.

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
        9. [Этап 7.1. Устойчивость выигрыша и pilot readiness](#stage-7-1)
        10. [Этап 7.2. Сегментный аудит против baseline](#stage-7-2)
        11. [Этап 8. Интерпретация и артефакты](#stage-8)
        11. [Финальные выводы](#final-conclusions)
        """
    ),
    md(
        """
        <a id="methodology"></a>
        ## Методологическая рамка

        Здесь важно не "выжать красивую метрику", а честно проверить модель. Если test попадет в подбор параметров, результат будет выглядеть лучше, чем он есть на самом деле. Поэтому правила эксперимента фиксируются до моделирования.

        - Все обучаемые преобразования (`imputer`, `scaler`, `encoder`, модель) живут внутри `Pipeline`.
        - Финальная test-выборка не участвует в Optuna, выборе модели или выборе признаков.
        - Для новых моделей используется единая `KFold(n_splits=5, shuffle=True, random_state=42)`.
        - В Optuna минимизируется RMSE. Так как `sklearn` для regression scoring возвращает отрицательные значения, в objective используется `-scores["test_rmse"].mean()`.
        - Baseline компании оценивается отдельно как предоставленная модель. Его test-метрики нужны для итогового сравнения, но не для настройки новых моделей.
        - Графики используются не декоративно: каждый блок EDA должен дать решение для предобработки, признаков или интерпретации.
        - Технические названия признаков остаются видимыми, а русский смысл дается в круглых скобках. Так ревьювер видит исходную схему данных, а бизнес-заказчик не теряет смысл.
        - После общей test-метрики отдельно проверяется устойчивость выигрыша, интервалы прогноза, decile-аудит и сегменты, где новая модель выигрывает у baseline.
        - Финальная рекомендация должна отвечать на операционные вопросы: насколько ошибка меньше baseline, исчезли ли невозможные отрицательные прогнозы, какие часы требуют внимания и почему модель нельзя запускать без проверки на свежем периоде.
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
        import textwrap
        import time
        import warnings
        from pathlib import Path
        from typing import Any

        import joblib
        import matplotlib.pyplot as plt
        import numpy as np
        import optuna
        import pandas as pd
        import seaborn as sns
        import sklearn
        from IPython.display import Markdown, display
        from sklearn.base import BaseEstimator
        from sklearn.compose import ColumnTransformer
        from sklearn.dummy import DummyRegressor
        from sklearn.inspection import permutation_importance
        from sklearn.impute import SimpleImputer
        from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
        from sklearn.model_selection import KFold, cross_val_predict, cross_validate
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

        def find_project_root(start: Path | None = None) -> Path:
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
        Следующая ячейка делает notebook автономным для тренажера: внутри нее встроен исходный код `BikeFeatureEngineer`, который при запуске записывается в импортируемый module-файл. Это длиннее обычной ячейки, зато ревьюверу не нужно вручную загружать отдельный `.py`.
        """
    ),
    code(component_bootstrap_code),
    md(
        """
        **Подвывод по этапу 1:** окружение готово к повторному запуску. Сам ноутбук уже содержит исходный код кастомного transformer и при запуске сам создает нужный module-файл для `joblib`. Поэтому для проверки в тренажере достаточно загрузить notebook и входные данные/baseline, без ручного переноса дополнительных `.py` файлов. Самая важная деталь окружения - `scikit-learn 1.6.1`: на этой версии без ошибок открывается baseline из `joblib`.

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
        RAW_FEATURE_SCHEMA = {
            "Temperature": {
                "name": "temperature",
                "description_ru": "температура воздуха",
                "unit": "°C",
                "role": "numeric",
            },
            "Humidity(%)": {
                "name": "humidity",
                "description_ru": "влажность воздуха",
                "unit": "%",
                "role": "numeric",
            },
            "Wind speed (m/s)": {
                "name": "wind_speed_ms",
                "description_ru": "скорость ветра",
                "unit": "м/с",
                "role": "numeric",
            },
            "Visibility (10m)": {
                "name": "visibility_10m",
                "description_ru": "видимость",
                "unit": "единицы по 10 м",
                "role": "numeric",
            },
            "Dew point temperature": {
                "name": "dew_point_temperature",
                "description_ru": "температура точки росы",
                "unit": "°C",
                "role": "numeric",
            },
            "Solar Radiation (MJ/m2)": {
                "name": "solar_radiation_mjm2",
                "description_ru": "солнечная радиация",
                "unit": "МДж/м²",
                "role": "numeric",
            },
            "Rainfall(mm)": {
                "name": "rainfallmm",
                "description_ru": "количество осадков, дождь",
                "unit": "мм",
                "role": "numeric",
            },
            "Snowfall (cm)": {
                "name": "snowfall_cm",
                "description_ru": "количество снега",
                "unit": "см",
                "role": "numeric",
            },
            "Seasons": {"name": "seasons", "description_ru": "сезон", "role": "categorical"},
            "Holiday": {"name": "holiday", "description_ru": "праздничный день", "role": "categorical"},
            "Functioning Day": {
                "name": "functioning_day",
                "description_ru": "работает ли прокат",
                "role": "categorical",
            },
            "Time_Period_Evening": {
                "name": "time_period_evening",
                "description_ru": "вечерний период",
                "role": "time_indicator",
            },
            "Time_Period_Late Evening": {
                "name": "time_period_late_evening",
                "description_ru": "поздний вечер",
                "role": "time_indicator",
            },
            "Time_Period_Morning": {
                "name": "time_period_morning",
                "description_ru": "утренний период",
                "role": "time_indicator",
            },
            "Time_Period_Night": {
                "name": "time_period_night",
                "description_ru": "ночной период",
                "role": "time_indicator",
            },
            TARGET_ORIGINAL: {
                "name": TARGET,
                "description_ru": "число прокатов велосипедов",
                "unit": "прокатов/час",
                "role": "target",
            },
        }
        '''
    ),
    md(
        """
        К raw-схеме добавляю derived/encoded признаки и из единого metadata-слоя генерирую рабочие структуры: rename, списки признаков, русские описания и единицы измерения.
        """
    ),
    code(
        r'''

        DERIVED_FEATURE_SCHEMA = {
            "seasons_Spring": {"description_ru": "весенний сезон"},
            "seasons_Summer": {"description_ru": "летний сезон"},
            "seasons_Autumn": {"description_ru": "осенний сезон"},
            "seasons_Winter": {"description_ru": "зимний сезон"},
            "holiday_Holiday": {"description_ru": "праздничный день"},
            "holiday_No Holiday": {"description_ru": "не праздничный день"},
            "functioning_day_Yes": {"description_ru": "прокат работает"},
            "functioning_day_No": {"description_ru": "прокат не работает"},
            "time_period_daytime": {"description_ru": "дневной период"},
            "dew_point_gap": {"description_ru": "разница температуры и точки росы"},
            "has_rain": {"description_ru": "наличие дождя"},
            "has_snow": {"description_ru": "наличие снега"},
            "has_precipitation": {"description_ru": "наличие любых осадков"},
            "comfortable_temperature": {"description_ru": "комфортный диапазон температуры"},
            "freezing_weather": {"description_ru": "морозная погода"},
            "hot_weather": {"description_ru": "жаркая погода"},
            "hot_and_humid": {"description_ru": "жара вместе с высокой влажностью"},
            "temp_x_solar": {"description_ru": "взаимодействие температуры и солнечной радиации"},
            "temp_x_humidity": {"description_ru": "взаимодействие температуры и влажности"},
            "rain_x_wind": {"description_ru": "взаимодействие дождя и ветра"},
            "snow_x_freezing": {"description_ru": "снег при морозной погоде"},
        }

        COLUMN_RENAME = {
            raw_name: metadata["name"]
            for raw_name, metadata in RAW_FEATURE_SCHEMA.items()
        }
        BASE_NUMERIC_FEATURES = [
            metadata["name"]
            for metadata in RAW_FEATURE_SCHEMA.values()
            if metadata["role"] == "numeric"
        ]
        CATEGORICAL_FEATURES = [
            metadata["name"]
            for metadata in RAW_FEATURE_SCHEMA.values()
            if metadata["role"] == "categorical"
        ]
        TIME_FEATURES = [
            metadata["name"]
            for metadata in RAW_FEATURE_SCHEMA.values()
            if metadata["role"] == "time_indicator"
        ]
        BASE_FEATURES = BASE_NUMERIC_FEATURES + CATEGORICAL_FEATURES + TIME_FEATURES

        FEATURE_DESCRIPTIONS_RU = {
            metadata["name"]: metadata["description_ru"]
            for metadata in RAW_FEATURE_SCHEMA.values()
        } | {
            name: metadata["description_ru"]
            for name, metadata in DERIVED_FEATURE_SCHEMA.items()
        }

        FEATURE_UNITS = {
            metadata["name"]: metadata["unit"]
            for metadata in RAW_FEATURE_SCHEMA.values()
            if "unit" in metadata
        }

        PARAMETER_DESCRIPTIONS_RU = {
            "n_neighbors": "число соседей",
            "weights": "веса соседей",
            "p": "метрика Минковского",
            "leaf_size": "размер листа поиска",
            "max_depth": "максимальная глубина",
            "min_samples_split": "минимум объектов для split",
            "min_samples_leaf": "минимум объектов в листе",
            "max_features": "число признаков для split",
            "ccp_alpha": "сила pruning",
            "random_state": "seed",
            "strategy": "стратегия dummy",
        }
        '''
    ),
    md(
        """
        В этом шаге собраны небольшие helper-функции для подписей, таблиц и проверки схемы. Они не обучают модель, а только делают следующие ячейки короче и стабильнее.
        """
    ),
    code(
        r'''


        def feature_label_for_reader(feature: str) -> tuple[str, str, str]:
            technical = feature.replace("num__", "").replace("cat__", "")
            description = FEATURE_DESCRIPTIONS_RU.get(technical, technical.replace("_", " "))
            plot_label = f"{technical}\n({description})" if description != technical else technical
            return technical, description, plot_label


        def inline_feature_label(feature: str, *, with_unit: bool = False) -> str:
            technical, description, _ = feature_label_for_reader(feature)
            label = f"{technical} ({description})" if description != technical else technical
            unit = FEATURE_UNITS.get(technical)
            return f"{label}, {unit}" if with_unit and unit else label


        def wrap_plot_label(label: Any, width: int = 34) -> str:
            return "\n".join(textwrap.wrap(str(label), width=width, break_long_words=False))


        def short_params_for_plot(model_name: str, params: dict[str, Any]) -> str:
            if not params:
                return "strategy=mean (среднее)"
            if model_name == "decision_tree_optuna":
                keys = ["max_depth", "min_samples_leaf", "min_samples_split"]
            elif model_name == "knn_optuna":
                keys = ["n_neighbors", "weights", "p"]
            else:
                keys = list(params)[:3]
            parts = []
            for key in keys:
                if key in params:
                    parts.append(f"{key}={params[key]} ({PARAMETER_DESCRIPTIONS_RU.get(key, key)})")
            return "\n".join(parts)


        def add_bar_labels(ax: plt.Axes, fmt: str) -> None:
            for container in ax.containers:
                ax.bar_label(container, fmt=fmt, padding=3, fontsize=9)


        def show_markdown(markdown_text: str) -> None:
            display(Markdown(textwrap.dedent(markdown_text).strip()))


        def show_table(title: str, frame: pd.DataFrame) -> None:
            show_markdown(f"**{title}**")
            display(frame)


        def read_csv_with_fallback(filename: str) -> pd.DataFrame:
            candidates = [DATA_DIR / filename, Path("/datasets") / filename]
            for candidate in candidates:
                if candidate.exists():
                    return pd.read_csv(candidate)
            raise FileNotFoundError(f"Could not find {filename} in: {candidates}")


        def build_raw_schema_report(df: pd.DataFrame, dataset_name: str) -> pd.DataFrame:
            expected_raw_columns = set(RAW_FEATURE_SCHEMA)
            observed_raw_columns = set(df.columns)
            rows = [
                {
                    "dataset": dataset_name,
                    "raw_column": raw_column,
                    "normalized_name": metadata["name"],
                    "role": metadata["role"],
                    "status": "present" if raw_column in observed_raw_columns else "missing_required",
                    "description_ru": metadata["description_ru"],
                }
                for raw_column, metadata in RAW_FEATURE_SCHEMA.items()
            ]
            rows.extend(
                {
                    "dataset": dataset_name,
                    "raw_column": raw_column,
                    "normalized_name": None,
                    "role": "unknown",
                    "status": "ignored_by_current_contract",
                    "description_ru": "колонка есть в данных, но не входит в текущий контракт модели",
                }
                for raw_column in sorted(observed_raw_columns - expected_raw_columns)
            )
            return pd.DataFrame(rows)


        def normalize_columns(df: pd.DataFrame, dataset_name: str) -> tuple[pd.DataFrame, pd.DataFrame]:
            schema_report = build_raw_schema_report(df, dataset_name)
            missing = schema_report.loc[
                schema_report["status"] == "missing_required",
                "raw_column",
            ].tolist()
            if missing:
                raise ValueError(f"{dataset_name}: missing required raw columns: {missing}")

            normalized = df.rename(columns=COLUMN_RENAME).copy()
            required_columns = list(COLUMN_RENAME.values())
            return normalized.loc[:, required_columns], schema_report
        '''
    ),
    md(
        """
        Теперь загружаю train/test и применяю schema contract. Обязательные колонки должны быть на месте. Лишние будущие колонки не ломают notebook: они попадут в audit как неиспользуемые текущей моделью.
        """
    ),
    code(
        r'''


        train_raw = read_csv_with_fallback("ds_s14_train_data.csv")
        test_raw = read_csv_with_fallback("ds_s14_test_data.csv")
        train, train_schema_report = normalize_columns(train_raw, "train")
        test, test_schema_report = normalize_columns(test_raw, "test")
        raw_schema_report = pd.concat([train_schema_report, test_schema_report], ignore_index=True)

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
        show_table("Размеры нормализованных данных", data_overview)
        show_table("Контракт сырых колонок", raw_schema_report)
        display(train.head())

        normalized_role_by_feature = {
            metadata["name"]: metadata["role"]
            for metadata in RAW_FEATURE_SCHEMA.values()
        }
        feature_dictionary = pd.DataFrame(
            [
                {
                    "technical_name": technical_name,
                    "description_ru": description,
                    "role": normalized_role_by_feature.get(technical_name, "derived_or_encoded"),
                    "unit": FEATURE_UNITS.get(technical_name, ""),
                }
                for technical_name, description in FEATURE_DESCRIPTIONS_RU.items()
            ]
        )
        show_table("Словарь признаков", feature_dictionary)

        assert train.shape == (7008, 16), "Unexpected train shape"
        assert test.shape == (1752, 16), "Unexpected test shape"
        assert list(train.columns) == list(test.columns), "Train/test schemas differ"
        assert TARGET in train.columns and TARGET in test.columns
        assert not raw_schema_report["status"].eq("missing_required").any()
        '''
    ),
    md(
        """
        **Подвывод по этапу 2:** train содержит `7008` строк, test - `1752`; набор обязательных колонок совпадает. Это важная точка контроля: дальше сравниваются модели, а не разные версии датасета.

        Схема теперь задается одним `RAW_FEATURE_SCHEMA`: из него генерируются rename-правила, описания и единицы измерения. Для ревью это удобнее, потому что не нужно искать три разных словаря и проверять, не разошлись ли они между собой. Для будущего использования это тоже практично: если появится новая сырая колонка, текущая модель не упадет молча и не начнет использовать признак случайно. Колонка попадет в schema audit как `ignored_by_current_contract`; чтобы взять ее в модель, ее надо явно добавить в schema и pipeline.

        """
    ),
    md(
        """
        <a id="stage-3"></a>
        ## Этап 3. Baseline компании

        Компания предоставила готовый pipeline линейной регрессии. Его нужно не переобучать, а загрузить и оценить. Это дает точку сравнения: новая модель должна быть не просто обучена, а практически лучше текущего подхода BikeSouth.
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
    md(
        """
        Pipeline baseline загружен. Ниже задаю один общий набор метрик, чтобы baseline, CV-модели и финальная модель считались одинаково.
        """
    ),
    code(
        r'''
        def regression_metrics(
            y_true: pd.Series,
            y_pred: np.ndarray,
        ) -> dict[str, Any]:
            return {
                "RMSE": root_mean_squared_error(y_true, y_pred),
                "MAE": mean_absolute_error(y_true, y_pred),
                "R2": r2_score(y_true, y_pred),
                "prediction_min": float(np.min(y_pred)),
                "prediction_max": float(np.max(y_pred)),
                "prediction_mean": float(np.mean(y_pred)),
                "negative_predictions": int(np.sum(y_pred < 0)),
            }


        def evaluate_fitted_model(
            name: str,
            model: Pipeline,
            X: pd.DataFrame,
            y: pd.Series,
            split: str,
        ) -> dict[str, Any]:
            predictions = model.predict(X)
            result = {"model": name, "split": split}
            result.update(regression_metrics(y, predictions))
            return result


        baseline_results = pd.DataFrame(
            [
                evaluate_fitted_model(
                    "company_linear_baseline",
                    baseline_pipeline,
                    X_train,
                    y_train,
                    "train",
                ),
                evaluate_fitted_model(
                    "company_linear_baseline",
                    baseline_pipeline,
                    X_test,
                    y_test,
                    "test",
                ),
            ]
        )
        display(baseline_results)
        '''
    ),
    md(
        """
        Таблица с метриками уже есть. Следом фиксирую расчетный текст с ключевыми числами: он обновится сам при полном перезапуске notebook.
        """
    ),
    code(
        r'''
        baseline_train_row = baseline_results.query("split == 'train'").iloc[0]
        baseline_test_row_for_comment = baseline_results.query("split == 'test'").iloc[0]
        baseline_negative_share = (
            baseline_test_row_for_comment["negative_predictions"] / len(X_test)
        )

        show_markdown(
            f"""
            **Расчетные итоги baseline**

            - Train RMSE: `{baseline_train_row["RMSE"]:.2f}`.
            - Test RMSE: `{baseline_test_row_for_comment["RMSE"]:.2f}`.
            - Test MAE: `{baseline_test_row_for_comment["MAE"]:.2f}`.
            - Test R2: `{baseline_test_row_for_comment["R2"]:.3f}`.
            - Отрицательные test-прогнозы:
              `{int(baseline_test_row_for_comment["negative_predictions"])}` из `{len(X_test)}`
              (`{baseline_negative_share:.1%}`).
            - Минимальный test-прогноз:
              `{baseline_test_row_for_comment["prediction_min"]:.2f}` прокатов/час.
            """
        )
        '''
    ),
    md(
        """
        **Интерпретация baseline:** baseline дает рабочую отправную точку, но у него есть неприятный для бизнеса дефект: линейная модель иногда прогнозирует отрицательный спрос. Для операционного планирования это не просто математическая странность. Такие прогнозы нельзя напрямую отдать менеджеру смены или в систему планирования парка: их пришлось бы вручную обрезать до нуля, а это уже отдельная бизнес-логика поверх модели.

        Поэтому новая модель должна решать две задачи одновременно. Первая - снизить ошибку в понятной единице, то есть в прокатах велосипедов за час. Вторая - вести себя физически разумно: спрос не бывает меньше нуля. Если модель выигрывает по RMSE, но продолжает выдавать невозможные значения, она все еще плохо готова к пилоту.
        """
    ),
    md(
        """
        <a id="stage-4"></a>
        ## Этап 4. Первичный аудит и EDA

        EDA здесь не для украшения. Каждый график должен ответить на практический вопрос и перейти в решение для pipeline:

        - Распределение target показывает, есть ли пики спроса и можно ли считать их ошибками. Здесь это реальные пиковые часы, поэтому target не чистится механически, а основной метрикой остается `RMSE`.
        - Распределения погодных признаков показывают масштабы, хвосты и пропуски. Из этого следует медианное заполнение внутри pipeline и масштабирование для KNN.
        - Scatter-графики `feature -> target` нужны, чтобы увидеть форму связи. Если связь не похожа на прямую, одной линейной модели мало; поэтому дальше проверяются KNN и дерево.
        - Spearman heatmap показывает ранговые связи: отдельно с target и между самими погодными признаками. Это помогает понять, какие признаки дают похожий сигнал и где стоит ожидать погодные взаимодействия.
        - Графики по категориям и time-period признакам нужны для бизнес-смысла: режим работы, сезон и время дня меняют сценарий спроса, а скрытый `Daytime` нужно восстановить как отдельный признак.
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
                    "train_values": sorted(
                        map(str, train[column].dropna().unique().tolist())
                    ),
                    "test_values": sorted(
                        map(str, test[column].dropna().unique().tolist())
                    ),
                }
            )

        show_table("Общий аудит", audit_overview)
        show_table("Пропуски и типы", missing_report)
        show_table("Категориальные и дискретные значения", pd.DataFrame(category_report))

        assert audit_overview["duplicates"].sum() == 0, "Unexpected duplicate rows"
        assert (train[TARGET] >= 0).all()
        assert (test[TARGET] >= 0).all()
        '''
    ),
    md(
        """
        **Подвывод по аудиту:** в данных нет дубликатов, а target неотрицательный. Нулевой спрос не выглядит технической ошибкой: часть таких часов может относиться к закрытому прокату, часть - к реальному отсутствию спроса. Это разные бизнес-ситуации, поэтому нули не удаляю.

        Пропуски есть только в погодных числовых признаках. Удалять строки из-за этого невыгодно: мы потеряем часы наблюдений, а вместе с ними сезонные и погодные сценарии. Заполнение будет обучаться внутри pipeline, чтобы на CV, test и будущем инференсе использовалось одно и то же правило.
        """
    ),
    code(
        r'''
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        sns.histplot(train[TARGET], bins=40, kde=True, ax=axes[0], color="#2f6f9f")
        axes[0].set_title("Распределение почасового спроса в train")
        axes[0].set_xlabel("rented_bike_count (прокаты), прокатов/час")
        axes[0].set_ylabel("Количество наблюдений")

        sns.boxplot(x=train[TARGET], ax=axes[1], color="#8fbcd4")
        axes[1].set_title("Хвосты и возможные выбросы спроса")
        axes[1].set_xlabel("rented_bike_count (прокаты), прокатов/час")
        plt.tight_layout()
        plt.show()

        target_summary = train[TARGET].describe(
            percentiles=[0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99]
        ).to_frame("train_target")
        display(target_summary)
        '''
    ),
    md(
        """
        **Подвывод по target:** распределение спроса перекошено вправо: обычных спокойных часов много, а высокие пики встречаются реже. Для проката эти пики не шум, а самые дорогие часы с точки зрения решения. Ошибка в спокойный час неприятна, но ошибка в час высокого спроса может привести к нехватке велосипедов, очередям или неверному планированию смен.

        Поэтому пики остаются в обучении. Основной метрикой беру `RMSE`, потому что она сильнее наказывает крупные промахи, а рядом обязательно смотрю `MAE`: она показывает более "среднюю" ошибку без такого сильного штрафа за пики.
        """
    ),
    md(
        """
        Дальше смотрю погодные числовые признаки: сначала распределения, затем связь каждого признака с target. Это нужно, чтобы выбрать способ заполнения пропусков и понять, достаточно ли линейной модели.
        """
    ),
    code(
        r'''
        continuous_labels = {
            column: inline_feature_label(column, with_unit=True)
            for column in BASE_NUMERIC_FEATURES
        }

        fig, axes = plt.subplots(4, 2, figsize=(15, 16))
        axes = axes.ravel()
        for index, (ax, column) in enumerate(zip(axes, BASE_NUMERIC_FEATURES)):
            sns.histplot(train[column], bins=35, kde=True, ax=ax, color="#477998")
            title = wrap_plot_label(continuous_labels[column], width=32)
            ax.set_title(f"Распределение: {title}")
            ax.set_xlabel(wrap_plot_label(continuous_labels[column], width=30))
            ax.set_ylabel("Количество наблюдений" if index % 2 == 0 else "")
        plt.tight_layout()
        plt.show()

        fig, axes = plt.subplots(4, 2, figsize=(15, 16))
        axes = axes.ravel()
        sampled_train = train.sample(min(2500, len(train)), random_state=RANDOM_STATE)
        for index, (ax, column) in enumerate(zip(axes, BASE_NUMERIC_FEATURES)):
            sns.scatterplot(
                data=sampled_train,
                x=column,
                y=TARGET,
                alpha=0.35,
                ax=ax,
                color="#20639b",
            )
            ax.set_title(f"Спрос и {wrap_plot_label(continuous_labels[column], width=32)}")
            ax.set_xlabel(wrap_plot_label(continuous_labels[column], width=30))
            ax.set_ylabel("")
        fig.supylabel("rented_bike_count (прокаты), прокатов/час", x=0.01, fontsize=11)
        plt.tight_layout(rect=(0.03, 0, 1, 1))
        plt.show()

        spearman_features = BASE_NUMERIC_FEATURES + [TARGET]
        spearman_matrix = train[spearman_features].corr(
            method="spearman",
            numeric_only=True,
        )
        spearman_plot_labels = {
            feature: wrap_plot_label(inline_feature_label(feature), width=20)
            for feature in spearman_features
        }

        numeric_corr = (
            spearman_matrix[TARGET]
            .drop(TARGET)
            .sort_values(key=lambda values: values.abs(), ascending=False)
            .reset_index()
            .rename(columns={"index": "feature", TARGET: "spearman_corr_with_target"})
        )
        numeric_corr["abs_spearman_corr"] = numeric_corr[
            "spearman_corr_with_target"
        ].abs()
        numeric_corr["feature_label"] = numeric_corr["feature"].map(
            lambda value: inline_feature_label(value, with_unit=True)
        )

        fig, axes = plt.subplots(
            1,
            2,
            figsize=(18, 7.5),
            gridspec_kw={"width_ratios": [1.25, 1]},
        )
        sns.heatmap(
            spearman_matrix.rename(
                index=spearman_plot_labels,
                columns=spearman_plot_labels,
            ),
            cmap="vlag",
            center=0,
            vmin=-1,
            vmax=1,
            annot=True,
            fmt=".2f",
            linewidths=0.5,
            cbar_kws={"label": "Spearman rho"},
            ax=axes[0],
        )
        axes[0].set_title("Spearman heatmap: числовые признаки и target")
        axes[0].tick_params(axis="x", rotation=45, labelsize=8)
        axes[0].tick_params(axis="y", rotation=0, labelsize=8)

        target_corr_plot = numeric_corr.sort_values("spearman_corr_with_target").copy()
        target_corr_plot["feature_plot_label"] = target_corr_plot["feature"].map(
            lambda value: wrap_plot_label(inline_feature_label(value), width=30)
        )
        bar_colors = np.where(
            target_corr_plot["spearman_corr_with_target"] >= 0,
            "#49759c",
            "#c98256",
        )
        axes[1].barh(
            target_corr_plot["feature_plot_label"],
            target_corr_plot["spearman_corr_with_target"],
            color=bar_colors,
        )
        axes[1].axvline(0, color="black", linewidth=1)
        axes[1].set_title("Связь признаков с rented_bike_count (спрос)")
        axes[1].set_xlabel("Spearman rho")
        axes[1].set_ylabel("")
        axes[1].set_xlim(-1, 1)
        axes[1].bar_label(axes[1].containers[0], fmt="%.2f", padding=3, fontsize=9)
        plt.tight_layout()
        plt.show()

        display(
            numeric_corr[
                [
                    "feature",
                    "feature_label",
                    "spearman_corr_with_target",
                    "abs_spearman_corr",
                ]
            ]
        )
        '''
    ),
    md(
        """
        **Как читать Spearman:** коэффициент Спирмена смотрит не на прямую линию, а на порядок значений. Если `rho` ближе к `+1`, то при росте признака спрос чаще тоже становится выше. Если ближе к `-1`, связь обратная. Около `0` означает, что устойчивого монотонного порядка почти нет.

        Heatmap нужна не только ради связи с target. Она показывает, какие погодные признаки несут похожий сигнал между собой: например, температура и точка росы могут двигаться вместе, а влажность часто ведет себя иначе. Это не доказывает причинность и не решает задачу отбора признаков автоматически. Зато помогает понять, где линейная baseline может быть слишком простой и где полезно проверить нелинейные модели и погодные взаимодействия.
        """
    ),
    md(
        """
        **Подвывод по погодным признакам:** Spearman здесь работает как компас, а не как финальное доказательство. Он показывает, что спрос заметнее всего меняется вместе с `temperature` (температура воздуха), `dew_point_temperature` (температура точки росы), `solar_radiation_mjm2` (солнечная радиация) и `humidity` (влажность воздуха).

        Главное видно на scatter-графиках: связь не выглядит одной прямой. Одинаковая температура может означать разный спрос, если час дождливый, ветреный или солнечный. Поэтому KNN и дерево здесь не "усложнение ради усложнения", а нормальная попытка поймать сочетания условий, которые линейная baseline сглаживает.
        """
    ),
    md(
        """
        Теперь проверяю категориальные и time-period признаки. Здесь важен не только средний спрос, но и смысл категорий: закрытый прокат, праздник и время дня описывают разные операционные сценарии.
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
            order = (
                train.groupby(column)[TARGET]
                .mean()
                .sort_values(ascending=False)
                .index
            )
            sns.barplot(
                data=train,
                x=column,
                y=TARGET,
                order=order,
                estimator="mean",
                errorbar=None,
                ax=ax,
                color="#4f8a8b",
            )
            title = wrap_plot_label(inline_feature_label(column), width=32)
            ax.set_title(f"Средний спрос по {title}")
            ax.set_xlabel("")
            ax.set_ylabel("")
            ax.tick_params(axis="x", rotation=20)
        fig.supylabel("Средний спрос, прокатов/час", x=0.01, fontsize=11)
        plt.tight_layout(rect=(0.03, 0, 1, 1))
        plt.show()
        '''
    ),
    md(
        """
        **Подвывод по категориальным и временным признакам:** `functioning_day` нельзя воспринимать как обычную категорию без бизнес-смысла. Если прокат закрыт, нулевой спрос ожидаем и модель не должна "докручивать" его до типичного спроса похожей погоды. Если прокат открыт и спрос нулевой, это уже другой сценарий: возможно, плохая погода, ночь или локальная просадка спроса.

        Time-period признаки описывают разные режимы дня: утро, вечер, ночь и дневной период. `Daytime` в данных спрятан как строка, где все четыре dummy-признака равны `False`, поэтому я восстанавливаю его явно. Так модель получает нормальный признак периода дня, а не отсутствие информации.
        """
    ),
    code(
        r'''
        eda_decisions = pd.DataFrame(
            [
                {
                    "EDA observation": (
                        "Погодные числовые признаки имеют пропуски, "
                        "но доля пропусков ограничена."
                    ),
                    "Modeling decision": (
                        "Не удалять строки; использовать "
                        "SimpleImputer(strategy='median') внутри pipeline."
                    ),
                },
                {
                    "EDA observation": "Категориальные признаки имеют небольшое число уровней.",
                    "Modeling decision": (
                        "Использовать SimpleImputer(strategy='most_frequent') "
                        "и OneHotEncoder(handle_unknown='ignore', drop='first')."
                    ),
                },
                {
                    "EDA observation": (
                        "Time_Period_Daytime скрыт как строка, где все "
                        "time-period dummy равны False."
                    ),
                    "Modeling decision": "Добавить `time_period_daytime` в кастомном transformer.",
                },
                {
                    "EDA observation": (
                        "Температура, влажность, солнечная радиация и осадки "
                        "связаны со спросом нелинейно."
                    ),
                    "Modeling decision": (
                        "Обучить KNN и Decision Tree; добавить weather "
                        "interaction признаки."
                    ),
                },
                {
                    "EDA observation": "Baseline линейной регрессии дает отрицательные прогнозы.",
                    "Modeling decision": (
                        "Проверять диапазон предсказаний финальной модели "
                        "и считать negative_predictions."
                    ),
                },
                {
                    "EDA observation": "Хвосты спроса выглядят как реальные пики, а не технические ошибки.",
                    "Modeling decision": (
                        "Не удалять target outliers механически; "
                        "оценивать RMSE и MAE вместе."
                    ),
                },
            ]
        )
        display(eda_decisions)
        '''
    ),
    md(
        """
        **Вывод этапа 4:** EDA дал не общий "посмотрели данные", а набор решений для pipeline. Пропуски заполняются внутри pipeline, пики спроса остаются в данных, `Daytime` восстанавливается явно, погодные взаимодействия добавляются в custom transformer, отрицательные прогнозы контролируются отдельной метрикой.

        Так модель строится от задачи, а не от списка доступных алгоритмов. Для бизнеса это означает, что мы отдельно учитываем закрытый прокат, пики нагрузки и погодные режимы. Для ревью это означает, что каждый следующий шаг моделирования связан с наблюдением из EDA.

        """
    ),
    md(
        """
        <a id="stage-5"></a>
        ## Этап 5. Pipeline и feature engineering

        Теперь решения из EDA превращаются в один pipeline. Важная идея: снаружи pipeline получает сырые признаки, а внутри сам делает все нужные шаги - добавляет признаки, заполняет пропуски, кодирует категории и обучает модель. Так меньше риска забыть какой-то шаг при повторном запуске или инференсе.

        Дополнительный пункт закрывается отдельным инженерным решением: кастомный `BikeFeatureEngineer` сделан как sklearn-compatible transformer с методами `fit` и `transform`. Он делает только безопасные вещи: восстанавливает `time_period_daytime` (дневной период), добавляет флаги осадков, температурные режимы и погодные взаимодействия. Target он не видит. Исходный код transformer встроен в notebook; при запуске notebook сам записывает `bike_demand_pipeline_components.py`, чтобы сохраненный `joblib` открывался в чистом Python-процессе.
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
            numeric_steps: list[tuple[str, Any]] = [("imputer", SimpleImputer(strategy="median"))]
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
        '''
    ),
    md(
        """
        Проверяю контракт transformer отдельно от обучения моделей. Здесь важно две вещи: стабильный набор выходных признаков и понятное поведение при будущих лишних колонках.
        """
    ),
    code(
        r'''


        schema_check = BikeFeatureEngineer().fit_transform(X_train.head(5))
        display(schema_check)
        assert list(schema_check.columns) == MODEL_FEATURES_AFTER_ENGINEERING
        assert schema_check.shape[1] == len(MODEL_FEATURES_AFTER_ENGINEERING)

        future_feature_probe = X_train.head(3).copy()
        future_feature_probe["future_weather_signal"] = 1.0
        future_transformer = BikeFeatureEngineer().fit(future_feature_probe)
        future_schema_contract = pd.DataFrame(
            [
                {
                    "scenario": "extra future column",
                    "column": "future_weather_signal",
                    "policy": future_transformer.unknown_feature_policy,
                    "status": "ignored_by_current_contract",
                    "how_to_use_later": (
                        "добавить в RAW_FEATURE_SCHEMA/BASE_FEATURES "
                        "и обновить pipeline"
                    ),
                }
            ]
        )
        show_table("Контракт для будущих лишних признаков", future_schema_contract)

        assert "future_weather_signal" in set(future_transformer.ignored_features_in_)
        assert (
            list(future_transformer.transform(future_feature_probe).columns)
            == MODEL_FEATURES_AFTER_ENGINEERING
        )

        additional_task_closure = pd.DataFrame(
            [
                {
                    "reviewer_requirement": "Добавить собственную обработку признаков в стиле sklearn.",
                    "implementation": (
                        "`BikeFeatureEngineer` наследуется от `BaseEstimator` "
                        "и `TransformerMixin`, имеет `fit`/`transform` "
                        "и возвращает стабильную схему колонок."
                    ),
                    "where_checked": (
                        "`schema_check`, `MODEL_FEATURES_AFTER_ENGINEERING`, "
                        "reload сохраненного `joblib`."
                    ),
                },
                {
                    "reviewer_requirement": (
                        "Feature engineering должен быть частью pipeline, "
                        "а не ручной подготовкой перед обучением."
                    ),
                    "implementation": (
                        "Первый шаг каждого нового pipeline - "
                        "`('feature_engineering', BikeFeatureEngineer())`; "
                        "дальше идут imputer/encoder/scaler и модель."
                    ),
                    "where_checked": (
                        "`make_model_pipeline`, `production_contract`, "
                        "`artifact_manifest`."
                    ),
                },
                {
                    "reviewer_requirement": "Новые признаки должны иметь смысл для задачи.",
                    "implementation": (
                        "Добавлены `time_period_daytime`, `has_rain`, "
                        "`has_snow`, `has_precipitation`, `dew_point_gap`, "
                        "`comfortable_temperature`, `freezing_weather`, "
                        "`hot_weather`, `hot_and_humid`, `temp_x_humidity`, "
                        "`temp_x_solar`, `rain_x_wind` и `snow_x_freezing`."
                    ),
                    "where_checked": (
                        "EDA-решения, список `ENGINEERED_FEATURES`, "
                        "importance финальной модели."
                    ),
                },
                {
                    "reviewer_requirement": "Сохраненная модель должна открываться вне ноутбука.",
                    "implementation": (
                        "Класс transformer встроен в notebook как source string, "
                        "при запуске записывается в импортируемый модуль; "
                        "модуль включен в manifest и проверен checksum."
                    ),
                    "where_checked": "`joblib.load`, `component_symbol_check`, `artifact_inventory`.",
                },
                {
                    "reviewer_requirement": (
                        "Будущие лишние колонки не должны случайно ломать "
                        "текущий pipeline."
                    ),
                    "implementation": (
                        "`BikeFeatureEngineer` требует обязательные колонки, "
                        "но по умолчанию игнорирует неизвестные признаки "
                        "и сохраняет их в `ignored_features_in_`."
                    ),
                    "where_checked": (
                        "`future_schema_contract`, `ignored_features_in_`, "
                        "schema audit."
                    ),
                },
            ]
        )
        display(additional_task_closure)
        '''
    ),
    md(
        """
        **Подвывод по pipeline:** дополнительный пункт закрыт не косметически. `BikeFeatureEngineer` работает как обычный sklearn-transformer: его можно положить внутрь `Pipeline`, сохранить через `joblib` и затем загрузить без ручной подготовки признаков рядом с ноутбуком.

        Схема остается понятной: сначала `BikeFeatureEngineer`, потом общий препроцессинг, потом модель. Это снижает риск типичной ошибки, когда в ноутбуке признаки посчитали одним способом, а при инференсе забыли повторить часть логики. KNN получает масштабированные числовые признаки, потому что расстояния чувствительны к масштабу. Дереву масштабирование не нужно: оно режет признаки по порогам и не выигрывает от стандартизации.

        """
    ),
    md(
        """
        <a id="stage-6"></a>
        ## Этап 6. Optuna и 5-fold CV

        Здесь сравниваются не отдельные алгоритмы "на глаз", а полные pipeline. Это важно: качество KNN и дерева считается вместе с теми же правилами обработки пропусков, категорий и инженерных признаков, которые потом попадут в финальную модель.

        `DummyRegressor` нужен как нижняя планка. После него Optuna подбирает параметры KNN и дерева. Для всех моделей используется одна и та же 5-fold CV, поэтому разница в метриках объясняется моделью и параметрами, а не разным протоколом проверки.
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


        def summarize_cv_scores(
            model_name: str,
            scores: dict[str, np.ndarray],
            params: dict[str, Any] | None = None,
        ) -> dict[str, Any]:
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


        def cross_validate_pipeline(
            model_name: str,
            pipeline: Pipeline,
            params: dict[str, Any] | None = None,
        ) -> dict[str, Any]:
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
        **Подвывод по DummyRegressor:** dummy просто предсказывает средний спрос. Это нижняя планка здравого смысла: если KNN или дерево не обгоняют среднее, значит проблема не в тонкой настройке, а в признаках, pipeline или протоколе проверки.

        В бизнес-терминах dummy - это подход "в любой час ожидать среднюю нагрузку". Такая логика заведомо плохо видит утренние/вечерние пики, погоду и закрытый прокат, поэтому сильная модель должна уверенно быть лучше этой планки.
        """
    ),
    md(
        """
        Следующий код задает objective-функции для Optuna. Внутри каждого trial собирается полный pipeline, поэтому CV оценивает не голый алгоритм, а весь путь от сырых признаков до прогноза.
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
            pipeline = make_model_pipeline(
                KNeighborsRegressor(**params),
                scale_numeric=True,
            )
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
                "max_depth": trial.suggest_categorical(
                    "max_depth",
                    [None, 3, 5, 7, 10, 15, 20, 30],
                ),
                "min_samples_split": trial.suggest_int("min_samples_split", 2, 80),
                "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 50),
                "max_features": trial.suggest_categorical(
                    "max_features",
                    [None, "sqrt", "log2", 0.5, 0.8, 1.0],
                ),
                "ccp_alpha": trial.suggest_float("ccp_alpha", 1e-8, 1e-2, log=True),
                "random_state": RANDOM_STATE,
            }
            pipeline = make_model_pipeline(
                DecisionTreeRegressor(**params),
                scale_numeric=False,
            )
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
        knn_study = optuna.create_study(
            direction="minimize",
            sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE),
        )
        knn_study.optimize(objective_knn, n_trials=N_TRIALS_KNN, show_progress_bar=False)
        study_timings["KNN"] = time.perf_counter() - start

        start = time.perf_counter()
        tree_study = optuna.create_study(
            direction="minimize",
            sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE),
        )
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
    md(
        """
        **Как читать подбор параметров:** у KNN главный рычаг - `n_neighbors`: мало соседей дает более резкую модель, много соседей сильнее сглаживает спрос. `weights="distance"` делает ближайшие часы более важными. У дерева главные ограничения - `max_depth`, `min_samples_leaf` и `min_samples_split`: они не дают дереву запомнить отдельные строки train.

        Поэтому дальше сравниваются не все trials Optuna, а лучшие найденные версии KNN и дерева по одному и тому же CV-протоколу.
        """
    ),
    md(
        """
        Теперь переоцениваю лучшие найденные pipeline в одном CV-сравнении рядом с dummy. Это финальный train-only шаг перед единственным обращением к test.
        """
    ),
    code(
        r'''
        knn_best_pipeline = make_model_pipeline(
            KNeighborsRegressor(**knn_study.best_params),
            scale_numeric=True,
        )
        tree_best_params = dict(tree_study.best_params)
        tree_best_params["random_state"] = RANDOM_STATE
        tree_best_pipeline = make_model_pipeline(
            DecisionTreeRegressor(**tree_best_params),
            scale_numeric=False,
        )

        cv_results = [dummy_cv_result]
        cv_results.append(
            cross_validate_pipeline(
                "knn_optuna",
                knn_best_pipeline,
                params=knn_study.best_params,
            )
        )
        cv_results.append(
            cross_validate_pipeline(
                "decision_tree_optuna",
                tree_best_pipeline,
                params=tree_best_params,
            )
        )

        cv_comparison = pd.DataFrame(cv_results).sort_values("cv_RMSE_mean")
        display(cv_comparison.drop(columns="params"))

        plot_df = cv_comparison.copy()
        model_display_names = {
            "dummy_mean": "Dummy\nmean",
            "knn_optuna": "KNN\nOptuna",
            "decision_tree_optuna": "DecisionTree\nOptuna",
        }
        plot_df["model_label"] = (
            plot_df["model"].map(model_display_names).fillna(plot_df["model"])
        )
        plot_df["parameter_summary"] = [
            short_params_for_plot(model, params).replace("\n", "; ")
            for model, params in zip(plot_df["model"], plot_df["params"])
        ]
        display(plot_df[["model", "model_label", "parameter_summary"]])

        fig, axes = plt.subplots(2, 2, figsize=(16, 10))
        axes = axes.ravel()
        sns.barplot(
            data=plot_df,
            x="model_label",
            y="cv_RMSE_mean",
            ax=axes[0],
            color="#49759c",
        )
        axes[0].set_title("CV RMSE: ниже лучше")
        axes[0].set_xlabel("Модель")
        axes[0].set_ylabel("RMSE, прокатов/час")
        axes[0].tick_params(axis="x", rotation=0)
        add_bar_labels(axes[0], "%.1f")

        sns.barplot(
            data=plot_df,
            x="model_label",
            y="cv_MAE_mean",
            ax=axes[1],
            color="#7aa95c",
        )
        axes[1].set_title("CV MAE: ниже лучше")
        axes[1].set_xlabel("Модель")
        axes[1].set_ylabel("MAE, прокатов/час")
        axes[1].tick_params(axis="x", rotation=0)
        add_bar_labels(axes[1], "%.1f")

        sns.barplot(
            data=plot_df,
            x="model_label",
            y="cv_R2_mean",
            ax=axes[2],
            color="#c98256",
        )
        axes[2].set_title("CV R2: выше лучше")
        axes[2].set_xlabel("Модель")
        axes[2].set_ylabel("R2")
        axes[2].tick_params(axis="x", rotation=0)
        add_bar_labels(axes[2], "%.3f")

        axes[3].axis("off")
        axes[3].set_title("Ключевые параметры")
        parameter_blocks = []
        for row in plot_df.itertuples(index=False):
            model_label = row.model_label.replace("\n", " ")
            parameter_summary = row.parameter_summary.replace("; ", "\n")
            parameter_blocks.append(f"{model_label}:\n{parameter_summary}")
        parameter_text = "\n\n".join(parameter_blocks)
        axes[3].text(
            0.0,
            0.98,
            parameter_text,
            va="top",
            ha="left",
            fontsize=10,
            linespacing=1.25,
        )
        plt.tight_layout()
        plt.show()
        '''
    ),
    md(
        """
        Лучшие модели выбраны по CV. Теперь коротко проверяю, не уперлась ли Optuna в край диапазона: если лучший параметр лежит на границе, диапазон мог быть слишком узким.
        """
    ),
    code(
        r'''
        def check_boundary_params(
            study: optuna.Study,
            model_name: str,
        ) -> pd.DataFrame:
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
        **Вывод этапа 6:** модель выбирается по среднему `RMSE` на 5-fold CV, а не по одному удачному разбиению. Рядом оставлены `MAE`, `R2`, разброс по фолдам и проверка границ Optuna. Это важно: модель для пилота должна быть не только лучшей в одной таблице, но и достаточно устойчивой к тому, какие часы попали в конкретный fold.

        Проверка границ нужна как инженерная защита. Если лучший параметр оказался прямо на краю диапазона, это сигнал, что поиск мог быть слишком узким. Здесь такая проверка явно вынесена в таблицу, чтобы выбор модели выглядел не как "Optuna что-то нашла", а как контролируемый эксперимент.

        """
    ),
    md(
        """
        <a id="stage-7"></a>
        ## Этап 7. Финальная проверка на test

        Теперь модель уже выбрана. Я обучаю ее на всем train и один раз применяю к test. До этого test не участвовал ни в Optuna, ни в выборе модели.
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
                evaluate_fitted_model(
                    "company_linear_baseline",
                    baseline_pipeline,
                    X_test,
                    y_test,
                    "test",
                ),
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

        baseline_test_rmse = final_results.query(
            "model == 'company_linear_baseline' and split == 'test'"
        )["RMSE"].iloc[0]
        final_test_rmse = final_results.query(
            "model == @best_model_name and split == 'test'"
        )["RMSE"].iloc[0]
        rmse_improvement_pct = (
            (baseline_test_rmse - final_test_rmse)
            / baseline_test_rmse
            * 100
        )

        display(final_results)
        print(f"Выбранная модель: {best_model_name}")
        print(f"Улучшение RMSE относительно baseline на test: {rmse_improvement_pct:.2f}%")
        print(f"Параметры финальной модели: {final_params}")
        '''
    ),
    md(
        """
        Метрики посчитаны. Теперь смотрю форму ошибок: насколько прогнозы лежат рядом с диагональю, есть ли перекос residuals и как финальная модель выглядит рядом с baseline на одних и тех же test-строках.
        """
    ),
    code(
        r'''
        residuals = y_test - final_test_predictions
        baseline_test_predictions = baseline_pipeline.predict(X_test)

        fig, axes = plt.subplots(2, 2, figsize=(16, 10))
        axes = axes.ravel()
        sns.scatterplot(
            x=y_test,
            y=final_test_predictions,
            alpha=0.55,
            ax=axes[0],
            color="#2f6f9f",
        )
        max_value = max(y_test.max(), final_test_predictions.max())
        axes[0].plot(
            [0, max_value],
            [0, max_value],
            color="black",
            linestyle="--",
            linewidth=1,
        )
        axes[0].set_title("Финальная модель: факт против прогноза")
        axes[0].set_xlabel("Факт, прокатов/час")
        axes[0].set_ylabel("Прогноз, прокатов/час")

        sns.histplot(residuals, bins=35, kde=True, ax=axes[1], color="#7aa95c")
        axes[1].set_title("Распределение ошибок финальной модели")
        axes[1].set_xlabel("Ошибка y_true - y_pred, прокатов/час")
        axes[1].set_ylabel("Количество наблюдений")

        comparison_plot = pd.DataFrame(
            {
                "actual": y_test,
                "baseline_prediction": baseline_test_predictions,
                "final_prediction": final_test_predictions,
            }
        ).sample(min(500, len(y_test)), random_state=RANDOM_STATE)
        sns.scatterplot(
            data=comparison_plot,
            x="actual",
            y="baseline_prediction",
            alpha=0.35,
            label="baseline",
            ax=axes[2],
        )
        sns.scatterplot(
            data=comparison_plot,
            x="actual",
            y="final_prediction",
            alpha=0.35,
            label=best_model_name,
            ax=axes[2],
        )
        axes[2].plot(
            [0, max_value],
            [0, max_value],
            color="black",
            linestyle="--",
            linewidth=1,
        )
        axes[2].set_title("Baseline и финальная модель на test")
        axes[2].set_xlabel("Факт, прокатов/час")
        axes[2].set_ylabel("Прогноз, прокатов/час")
        axes[2].legend()
        axes[3].axis("off")

        plt.tight_layout()
        plt.show()
        '''
    ),
    md(
        """
        После графиков фиксирую численный итог отдельным блоком. Это расчетная часть: значения берутся из таблиц выше, чтобы при полном перезапуске notebook текст и цифры не расходились.
        """
    ),
    code(
        r'''
        baseline_final_row = final_results.query(
            "model == 'company_linear_baseline' and split == 'test'"
        ).iloc[0]
        selected_final_row = final_results.query(
            "model == @best_model_name and split == 'test'"
        ).iloc[0]
        test_rmse_delta = baseline_final_row["RMSE"] - selected_final_row["RMSE"]
        test_mae_delta = baseline_final_row["MAE"] - selected_final_row["MAE"]
        test_r2_delta = selected_final_row["R2"] - baseline_final_row["R2"]

        show_markdown(
            f"""
            **Расчетные итоги финальной test-проверки**

            - Baseline test RMSE: `{baseline_final_row["RMSE"]:.2f}`.
            - Final test RMSE: `{selected_final_row["RMSE"]:.2f}`.
            - Улучшение RMSE: `{test_rmse_delta:.2f}` прокатов/час (`{rmse_improvement_pct:.2f}%`).
            - Улучшение MAE: `{test_mae_delta:.2f}` прокатов/час.
            - Прирост R2: `{test_r2_delta:.3f}`.
            - Отрицательные прогнозы baseline/final:
              `{int(baseline_final_row["negative_predictions"])}` /
              `{int(selected_final_row["negative_predictions"])}`.
            """
        )
        '''
    ),
    md(
        """
        **Интерпретация финальной проверки:** на test финальная модель ошибается заметно меньше baseline и не дает отрицательных прогнозов. Для заказчика это читается просто: прогноз стал ближе к реальному числу прокатов за час, а невозможные значения меньше нуля исчезли.

        Это уже похоже на рабочий кандидат для пилота: результат лучше текущего baseline не только по средней ошибке, но и по здравому диапазону прогнозов. Но это еще не автоматический промышленный запуск. Перед внедрением нужна проверка на более позднем периоде: могла измениться погода, сезонность, режим работы, клиентский поток или доступность велосипедов.
        """
    ),
    md(
        """
        <a id="stage-7-1"></a>
        ## Этап 7.1. Устойчивость выигрыша и pilot readiness

        Здесь я не добавляю новые алгоритмы и не переотбираю модель по test. Модель уже выбрана на train CV. Дальше идут проверки, которые нужны для взрослого пилота: насколько устойчив выигрыш против baseline, какой диапазон неопределенности у прогноза и где модель ошибается сильнее.
        """
    ),
    code(
        r'''
        N_BOOTSTRAP = 3_000
        bootstrap_rng = np.random.default_rng(RANDOM_STATE)
        actual_test = y_test.to_numpy()
        baseline_prediction_test = np.asarray(baseline_test_predictions)
        final_prediction_test = np.asarray(final_test_predictions)

        bootstrap_rows = []
        for _ in range(N_BOOTSTRAP):
            indices = bootstrap_rng.integers(0, len(actual_test), len(actual_test))
            actual_sample = actual_test[indices]
            baseline_sample = baseline_prediction_test[indices]
            final_sample = final_prediction_test[indices]
            bootstrap_rows.append(
                {
                    "RMSE_delta": root_mean_squared_error(
                        actual_sample,
                        baseline_sample,
                    )
                    - root_mean_squared_error(actual_sample, final_sample),
                    "MAE_delta": mean_absolute_error(
                        actual_sample,
                        baseline_sample,
                    )
                    - mean_absolute_error(actual_sample, final_sample),
                }
            )

        bootstrap_improvement = pd.DataFrame(bootstrap_rows)


        def bootstrap_summary_row(column: str, label: str) -> dict[str, Any]:
            values = bootstrap_improvement[column]
            ci_low, ci_high = values.quantile([0.025, 0.975]).tolist()
            return {
                "metric": label,
                "mean_delta": values.mean(),
                "ci_2_5": ci_low,
                "ci_97_5": ci_high,
                "share_positive": (values > 0).mean(),
            }


        bootstrap_summary = pd.DataFrame(
            [
                bootstrap_summary_row(
                    "RMSE_delta",
                    "baseline_RMSE - final_RMSE",
                ),
                bootstrap_summary_row(
                    "MAE_delta",
                    "baseline_MAE - final_MAE",
                ),
            ]
        )
        display(bootstrap_summary)

        fig, axes = plt.subplots(1, 2, figsize=(16, 5))
        bootstrap_plot_specs = [
            ("RMSE_delta", "Bootstrap: выигрыш RMSE", "RMSE delta, прокатов/час"),
            ("MAE_delta", "Bootstrap: выигрыш MAE", "MAE delta, прокатов/час"),
        ]
        for ax, (column, title, xlabel) in zip(axes, bootstrap_plot_specs):
            sns.histplot(
                bootstrap_improvement[column],
                bins=40,
                kde=True,
                color="#49759c",
                ax=ax,
            )
            ci_low, ci_high = bootstrap_improvement[column].quantile(
                [0.025, 0.975]
            )
            ax.axvline(0, color="black", linewidth=1)
            ax.axvline(ci_low, color="#c98256", linestyle="--", linewidth=1.5)
            ax.axvline(ci_high, color="#c98256", linestyle="--", linewidth=1.5)
            ax.set_title(title)
            ax.set_xlabel(xlabel)
            ax.set_ylabel("Количество bootstrap-выборок")
        plt.tight_layout()
        plt.show()

        rmse_bootstrap_row = bootstrap_summary.query(
            "metric == 'baseline_RMSE - final_RMSE'"
        ).iloc[0]
        mae_bootstrap_row = bootstrap_summary.query(
            "metric == 'baseline_MAE - final_MAE'"
        ).iloc[0]

        show_markdown(
            f"""
            **Расчетная проверка устойчивости выигрыша**

            - Bootstrap 95% CI для выигрыша RMSE:
              `{rmse_bootstrap_row["ci_2_5"]:.2f}` -
              `{rmse_bootstrap_row["ci_97_5"]:.2f}` прокатов/час.
            - Доля bootstrap-выборок, где final лучше baseline по RMSE:
              `{rmse_bootstrap_row["share_positive"]:.1%}`.
            - Bootstrap 95% CI для выигрыша MAE:
              `{mae_bootstrap_row["ci_2_5"]:.2f}` -
              `{mae_bootstrap_row["ci_97_5"]:.2f}` прокатов/час.
            """
        )
        '''
    ),
    md(
        """
        Bootstrap отвечает на практичный вопрос: не выглядит ли выигрыш случайным из-за конкретного test-набора. Если почти во всех bootstrap-выборках финальная модель лучше baseline, это сильнее одной строки с RMSE. Для заказчика это означает, что улучшение не держится на нескольких удачных часах.
        """
    ),
    md(
        """
        Следующий слой - интервалы прогноза. Точечный прогноз удобен, но для пилота полезнее знать диапазон риска: обычный прогноз и верхнюю границу, к которой стоит готовить парк и смены. Интервалы ниже калибруются на out-of-fold ошибках train, чтобы не учиться на test.
        """
    ),
    code(
        r'''
        final_oof_predictions = cross_val_predict(
            final_pipeline,
            X_train,
            y_train,
            cv=cv,
            n_jobs=1,
        )
        calibration_frame = pd.DataFrame(
            {
                "oof_prediction": final_oof_predictions,
                "abs_residual": np.abs(y_train.to_numpy() - final_oof_predictions),
            }
        )
        _, demand_band_edges = pd.qcut(
            calibration_frame["oof_prediction"],
            q=[0, 1 / 3, 2 / 3, 1],
            retbins=True,
            duplicates="drop",
        )
        demand_band_edges[0] = -np.inf
        demand_band_edges[-1] = np.inf
        demand_band_labels = [
            "low_predicted_demand (низкий прогноз спроса)",
            "medium_predicted_demand (средний прогноз спроса)",
            "high_predicted_demand (высокий прогноз спроса)",
        ][: len(demand_band_edges) - 1]

        calibration_frame["prediction_band"] = pd.cut(
            calibration_frame["oof_prediction"],
            bins=demand_band_edges,
            labels=demand_band_labels,
            include_lowest=True,
        )
        interval_quantiles = (
            calibration_frame.groupby("prediction_band", observed=True)[
                "abs_residual"
            ]
            .quantile([0.5, 0.8, 0.9])
            .unstack()
            .rename(
                columns={
                    0.5: "q50_abs_residual",
                    0.8: "q80_abs_residual",
                    0.9: "q90_abs_residual",
                }
            )
            .reset_index()
        )
        global_q90_abs_residual = calibration_frame["abs_residual"].quantile(0.9)

        prediction_interval_frame = pd.DataFrame(
            {
                "actual": actual_test,
                "baseline_prediction": baseline_prediction_test,
                "final_prediction": final_prediction_test,
            }
        )
        prediction_interval_frame["prediction_band"] = pd.cut(
            prediction_interval_frame["final_prediction"],
            bins=demand_band_edges,
            labels=demand_band_labels,
            include_lowest=True,
        )
        band_to_q90 = {
            str(key): float(value)
            for key, value in interval_quantiles.set_index("prediction_band")[
                "q90_abs_residual"
            ].to_dict().items()
        }
        interval_radius_90 = (
            prediction_interval_frame["prediction_band"]
            .astype(str)
            .map(band_to_q90)
            .astype(float)
            .fillna(global_q90_abs_residual)
        )
        prediction_interval_frame["prediction_lower_90"] = np.clip(
            prediction_interval_frame["final_prediction"] - interval_radius_90,
            0,
            None,
        )
        prediction_interval_frame["prediction_upper_90"] = (
            prediction_interval_frame["final_prediction"] + interval_radius_90
        )
        prediction_interval_frame["covered_90"] = (
            prediction_interval_frame["actual"].between(
                prediction_interval_frame["prediction_lower_90"],
                prediction_interval_frame["prediction_upper_90"],
            )
        )
        prediction_interval_frame["interval_width_90"] = (
            prediction_interval_frame["prediction_upper_90"]
            - prediction_interval_frame["prediction_lower_90"]
        )

        prediction_interval_summary = pd.DataFrame(
            [
                {
                    "interval": "banded_oof_abs_residual_q90",
                    "test_coverage": prediction_interval_frame["covered_90"].mean(),
                    "mean_width": prediction_interval_frame[
                        "interval_width_90"
                    ].mean(),
                    "median_width": prediction_interval_frame[
                        "interval_width_90"
                    ].median(),
                    "global_q90_abs_residual": global_q90_abs_residual,
                }
            ]
        )
        interval_by_band = (
            prediction_interval_frame.groupby("prediction_band", observed=True)
            .agg(
                n_hours=("actual", "size"),
                actual_mean=("actual", "mean"),
                prediction_mean=("final_prediction", "mean"),
                interval_coverage_90=("covered_90", "mean"),
                mean_interval_width_90=("interval_width_90", "mean"),
            )
            .reset_index()
        )
        band_error_rows = []
        for band, group in prediction_interval_frame.groupby(
            "prediction_band",
            observed=True,
        ):
            band_error_rows.append(
                {
                    "prediction_band": band,
                    "mean_abs_error": mean_absolute_error(
                        group["actual"],
                        group["final_prediction"],
                    ),
                }
            )
        band_error_table = pd.DataFrame(band_error_rows)
        interval_by_band = (
            interval_by_band.merge(band_error_table, on="prediction_band", how="left")
            .merge(interval_quantiles, on="prediction_band", how="left")
        )

        display(prediction_interval_summary)
        display(interval_by_band)

        interval_plot = interval_by_band.copy()
        interval_plot["band_label"] = interval_plot["prediction_band"].astype(str).map(
            lambda label: wrap_plot_label(label, width=34)
        )

        fig, axes = plt.subplots(1, 2, figsize=(17, 6))
        sns.barplot(
            data=interval_plot,
            y="band_label",
            x="q90_abs_residual",
            color="#49759c",
            ax=axes[0],
        )
        axes[0].set_title("OOF q90 ошибки по уровню прогноза")
        axes[0].set_xlabel("q90 абсолютной ошибки, прокатов/час")
        axes[0].set_ylabel("")
        add_bar_labels(axes[0], "%.1f")

        sns.barplot(
            data=interval_plot,
            y="band_label",
            x="interval_coverage_90",
            color="#7aa95c",
            ax=axes[1],
        )
        axes[1].axvline(0.9, color="black", linestyle="--", linewidth=1)
        axes[1].set_title("Покрытие 90% интервала на test")
        axes[1].set_xlabel("Доля фактов внутри интервала")
        axes[1].set_ylabel("")
        axes[1].set_xlim(0, 1)
        add_bar_labels(axes[1], "%.2f")

        plt.tight_layout()
        plt.show()

        interval_summary_row = prediction_interval_summary.iloc[0]
        show_markdown(
            f"""
            **Расчетные итоги по интервалам**

            - Фактическое покрытие 90% interval на test:
              `{interval_summary_row["test_coverage"]:.1%}`.
            - Средняя ширина interval:
              `{interval_summary_row["mean_width"]:.2f}` прокатов/час.
            - Медианная ширина interval:
              `{interval_summary_row["median_width"]:.2f}` прокатов/час.
            """
        )
        '''
    ),
    md(
        """
        Интервал прогноза не делает модель "точнее" сам по себе. Его смысл другой: он показывает диапазон риска. Если верхняя граница высокая, операционная команда может заранее проверить парк и смены, даже если точечный прогноз выглядит умеренным.
        """
    ),
    md(
        """
        Теперь смотрю decile-аудит. Он показывает, как модель ведет себя от самых спокойных часов до верхних 10% фактического спроса. Это важнее обычного scatterplot: бизнесу нужно знать, не начинает ли модель систематически недооценивать самые дорогие часы.
        """
    ),
    code(
        r'''
        decile_frame = prediction_interval_frame.copy()
        decile_frame["actual_demand_decile"] = pd.qcut(
            pd.Series(actual_test).rank(method="first"),
            q=10,
            labels=[f"D{index}" for index in range(1, 11)],
        )
        decile_rows = []
        for decile, group in decile_frame.groupby(
            "actual_demand_decile",
            observed=True,
        ):
            baseline_rmse_decile = root_mean_squared_error(
                group["actual"],
                group["baseline_prediction"],
            )
            final_rmse_decile = root_mean_squared_error(
                group["actual"],
                group["final_prediction"],
            )
            decile_rows.append(
                {
                    "actual_demand_decile": decile,
                    "n_hours": len(group),
                    "actual_mean": group["actual"].mean(),
                    "baseline_RMSE": baseline_rmse_decile,
                    "final_RMSE": final_rmse_decile,
                    "RMSE_delta": baseline_rmse_decile - final_rmse_decile,
                    "baseline_MAE": mean_absolute_error(
                        group["actual"],
                        group["baseline_prediction"],
                    ),
                    "final_MAE": mean_absolute_error(
                        group["actual"],
                        group["final_prediction"],
                    ),
                    "final_bias": (
                        group["final_prediction"] - group["actual"]
                    ).mean(),
                    "final_underprediction_share": (
                        group["final_prediction"] < group["actual"]
                    ).mean(),
                    "interval_coverage_90": group["covered_90"].mean(),
                }
            )

        decile_audit = pd.DataFrame(decile_rows)
        display(decile_audit)

        decile_rmse_plot = decile_audit.melt(
            id_vars=["actual_demand_decile"],
            value_vars=["baseline_RMSE", "final_RMSE"],
            var_name="model",
            value_name="RMSE",
        )
        decile_rmse_plot["model"] = decile_rmse_plot["model"].map(
            {
                "baseline_RMSE": "baseline",
                "final_RMSE": best_model_name,
            }
        )

        fig, axes = plt.subplots(1, 2, figsize=(17, 6))
        sns.lineplot(
            data=decile_rmse_plot,
            x="actual_demand_decile",
            y="RMSE",
            hue="model",
            marker="o",
            ax=axes[0],
        )
        axes[0].set_title("RMSE по decile фактического спроса")
        axes[0].set_xlabel("Decile фактического спроса")
        axes[0].set_ylabel("RMSE, прокатов/час")
        axes[0].legend(title="Модель")

        sns.barplot(
            data=decile_audit,
            x="actual_demand_decile",
            y="final_bias",
            color="#c98256",
            ax=axes[1],
        )
        axes[1].axhline(0, color="black", linewidth=1)
        axes[1].set_title("Bias финальной модели по decile")
        axes[1].set_xlabel("Decile фактического спроса")
        axes[1].set_ylabel("Средний прогноз - факт, прокатов/час")
        add_bar_labels(axes[1], "%.1f")
        plt.tight_layout()
        plt.show()

        top_decile = decile_audit.iloc[-1]
        show_markdown(
            f"""
            **Расчетные итоги decile-аудита**

            - Верхний decile спроса: средний факт
              `{top_decile["actual_mean"]:.2f}` прокатов/час.
            - В верхнем decile выигрыш RMSE против baseline:
              `{top_decile["RMSE_delta"]:.2f}` прокатов/час.
            - Bias финальной модели в верхнем decile:
              `{top_decile["final_bias"]:.2f}` прокатов/час.
            - Доля недооценок в верхнем decile:
              `{top_decile["final_underprediction_share"]:.1%}`.
            """
        )
        '''
    ),
    md(
        """
        Decile-аудит переводит качество модели в риск пилота. Если верхние decile дают сильную недооценку, модель нельзя без контроля использовать для планирования пиков. Если ошибка в верхних decile ниже baseline, это сильный аргумент: модель полезна именно там, где цена промаха выше.
        """
    ),
    md(
        """
        Последняя часть этого слоя - простая таблица действий. Это не автоматическое бизнес-правило для production, а понятный каркас пилота: что делать с низким, обычным и рискованно высоким прогнозом.
        """
    ),
    code(
        r'''
        business_low_threshold, business_high_threshold = y_train.quantile(
            [0.33, 0.66]
        ).tolist()
        prediction_interval_frame["pilot_action_band"] = np.select(
            [
                prediction_interval_frame["prediction_upper_90"]
                >= business_high_threshold,
                prediction_interval_frame["prediction_upper_90"]
                < business_low_threshold,
            ],
            [
                "capacity_watch (риск высокого спроса)",
                "low_load (низкая ожидаемая нагрузка)",
            ],
            default="normal_plan (обычное планирование)",
        )

        pilot_action_table = pd.DataFrame(
            [
                {
                    "pilot_action_band": "capacity_watch (риск высокого спроса)",
                    "rule": (
                        "prediction_upper_90 >= train 66% demand quantile"
                    ),
                    "business_meaning": (
                        "даже с учетом неопределенности есть риск "
                        "высокой нагрузки"
                    ),
                    "recommended_action": (
                        "заранее проверить парк, смену поддержки "
                        "и доступность велосипедов"
                    ),
                    "monitoring_question": (
                        "не недооценивает ли модель пики спроса"
                    ),
                },
                {
                    "pilot_action_band": "normal_plan (обычное планирование)",
                    "rule": "между low и high порогами train",
                    "business_meaning": "типовая нагрузка без явного сигнала риска",
                    "recommended_action": (
                        "использовать прогноз для обычного почасового плана"
                    ),
                    "monitoring_question": (
                        "не растет ли ошибка в отдельных погодных режимах"
                    ),
                },
                {
                    "pilot_action_band": "low_load (низкая ожидаемая нагрузка)",
                    "rule": (
                        "prediction_upper_90 < train 33% demand quantile"
                    ),
                    "business_meaning": (
                        "даже верхняя граница прогноза остается низкой"
                    ),
                    "recommended_action": (
                        "не держать лишний ресурс без отдельной причины"
                    ),
                    "monitoring_question": (
                        "не пропускает ли модель неожиданные всплески спроса"
                    ),
                },
            ]
        )

        pilot_action_summary = []
        for action_band, group in prediction_interval_frame.groupby(
            "pilot_action_band",
            observed=True,
        ):
            pilot_action_summary.append(
                {
                    "pilot_action_band": action_band,
                    "n_hours": len(group),
                    "actual_mean": group["actual"].mean(),
                    "prediction_mean": group["final_prediction"].mean(),
                    "upper_90_mean": group["prediction_upper_90"].mean(),
                    "final_MAE": mean_absolute_error(
                        group["actual"],
                        group["final_prediction"],
                    ),
                }
            )
        pilot_action_summary = pd.DataFrame(pilot_action_summary)

        show_table("Pilot action rules", pilot_action_table)
        show_table("Pilot action summary on test", pilot_action_summary)

        fig, axes = plt.subplots(1, 2, figsize=(17, 6))
        action_plot = pilot_action_summary.copy()
        action_plot["action_label"] = action_plot["pilot_action_band"].map(
            lambda label: wrap_plot_label(label, width=28)
        )
        sns.barplot(
            data=action_plot,
            y="action_label",
            x="n_hours",
            color="#49759c",
            ax=axes[0],
        )
        axes[0].set_title("Сколько test-часов попадает в действие")
        axes[0].set_xlabel("Количество часов")
        axes[0].set_ylabel("")
        add_bar_labels(axes[0], "%.0f")

        sns.barplot(
            data=action_plot,
            y="action_label",
            x="final_MAE",
            color="#7aa95c",
            ax=axes[1],
        )
        axes[1].set_title("MAE финальной модели по действиям")
        axes[1].set_xlabel("MAE, прокатов/час")
        axes[1].set_ylabel("")
        add_bar_labels(axes[1], "%.1f")
        plt.tight_layout()
        plt.show()

        capacity_watch_hours = int(
            (
                prediction_interval_frame["pilot_action_band"]
                == "capacity_watch (риск высокого спроса)"
            ).sum()
        )
        show_markdown(
            f"""
            **Расчетные итоги pilot readiness**

            - Train-порог низкой нагрузки: `{business_low_threshold:.2f}`.
            - Train-порог высокой нагрузки: `{business_high_threshold:.2f}`.
            - Test-часов в зоне `capacity_watch`: `{capacity_watch_hours}`.
            - Эти часы не требуют автоматического решения, но требуют
              приоритетного просмотра при пилоте.
            """
        )
        '''
    ),
    md(
        """
        **Вывод по pilot readiness:** без добавления новых моделей мы получили слой, который обычно отличает исследовательский ноутбук от решения для пилота. Теперь есть не только точечный прогноз, но и проверка устойчивости выигрыша, интервалы неопределенности, аудит верхних decile спроса и понятная таблица действий.

        Отдельно отмечу потенциальный следующий шаг. Когда в программе будут пройдены ансамбли и boosting, их стоит аккуратно проверить как challenger-модели: `RandomForestRegressor`, `ExtraTreesRegressor`, `GradientBoostingRegressor` или `HistGradientBoostingRegressor` часто сильны на табличных нелинейных задачах. В этой работе я их не использую, чтобы не выходить за рамку текущей постановки и изученного материала, но это честное направление для улучшения качества после базового решения.
        """
    ),
    md(
        """
        <a id="stage-7-2"></a>
        ## Этап 7.2. Сегментный аудит против baseline

        Средний `RMSE` отвечает на вопрос "стала ли модель лучше в среднем". Для проката этого мало. Бизнесу важно понять, где именно появляется выигрыш: в пиковом спросе, в дождь, ночью, по сезонам или только на простых часах.

        Ниже baseline и финальная модель сравниваются в одинаковых test-сегментах. Сегменты пересекаются: это не одно разбиение выборки, а набор рабочих срезов для пилота и мониторинга. Такой аудит помогает заранее увидеть, где модель можно использовать увереннее, а где потребуется отдельный контроль.
        """
    ),
    code(
        r'''
        segment_frame = X_test.copy()
        segment_frame[TARGET] = y_test.to_numpy()
        segment_frame["baseline_prediction"] = baseline_test_predictions
        segment_frame["final_prediction"] = final_test_predictions

        low_threshold, high_threshold = y_test.quantile([0.33, 0.66]).tolist()
        segment_frame["demand_level"] = np.select(
            [
                y_test <= low_threshold,
                y_test <= high_threshold,
            ],
            [
                f"low <= {low_threshold:.0f} (низкий спрос)",
                f"medium {low_threshold:.0f}-{high_threshold:.0f} (средний спрос)",
            ],
            default=f"high > {high_threshold:.0f} (высокий спрос)",
        )


        def category_value_label(column: str, value: Any) -> str:
            value_text = str(value)
            description = FEATURE_DESCRIPTIONS_RU.get(f"{column}_{value_text}", value_text)
            return f"{value_text} ({description})" if description != value_text else value_text


        segment_frame["season_segment"] = segment_frame["seasons"].map(
            lambda value: category_value_label("seasons", value)
        )
        segment_frame["holiday_segment"] = segment_frame["holiday"].map(
            lambda value: category_value_label("holiday", value)
        )
        segment_frame["functioning_segment"] = segment_frame["functioning_day"].map(
            lambda value: category_value_label("functioning_day", value)
        )
        segment_frame["rainfall_segment"] = np.where(
            segment_frame["rainfallmm"].fillna(0) > 0,
            "rainfallmm > 0 (есть дождь)",
            "rainfallmm = 0 (без дождя)",
        )
        segment_frame["snowfall_segment"] = np.where(
            segment_frame["snowfall_cm"].fillna(0) > 0,
            "snowfall_cm > 0 (есть снег)",
            "snowfall_cm = 0 (без снега)",
        )
        segment_frame["time_period_segment"] = np.select(
            [
                segment_frame["time_period_morning"].astype(bool),
                segment_frame["time_period_evening"].astype(bool),
                segment_frame["time_period_late_evening"].astype(bool),
                segment_frame["time_period_night"].astype(bool),
            ],
            [
                "time_period_morning (утренний период)",
                "time_period_evening (вечерний период)",
                "time_period_late_evening (поздний вечер)",
                "time_period_night (ночной период)",
            ],
            default="time_period_daytime (дневной период)",
        )

        segment_columns = {
            "demand_level (уровень фактического спроса)": "demand_level",
            "seasons (сезон)": "season_segment",
            "holiday (праздничный день)": "holiday_segment",
            "functioning_day (работает ли прокат)": "functioning_segment",
            "rainfallmm (количество осадков, дождь)": "rainfall_segment",
            "snowfall_cm (количество снега)": "snowfall_segment",
            "time_period (период дня)": "time_period_segment",
        }
        '''
    ),
    md(
        """
        Сегменты готовы. Теперь для каждого среза считаются одинаковые метрики baseline и финальной модели: RMSE, MAE, средний фактический спрос и число отрицательных прогнозов. Это позволяет сравнить модели честно: один и тот же набор часов, одна и та же целевая переменная, разные прогнозы.
        """
    ),
    code(
        r'''


        def segment_metric_table(
            frame: pd.DataFrame,
            segment_group: str,
            segment_column: str,
        ) -> pd.DataFrame:
            rows = []
            for segment_value, group in frame.groupby(segment_column, dropna=False):
                baseline_rmse = root_mean_squared_error(
                    group[TARGET],
                    group["baseline_prediction"],
                )
                final_rmse = root_mean_squared_error(
                    group[TARGET],
                    group["final_prediction"],
                )
                baseline_mae = mean_absolute_error(
                    group[TARGET],
                    group["baseline_prediction"],
                )
                final_mae = mean_absolute_error(
                    group[TARGET],
                    group["final_prediction"],
                )
                rmse_improvement = (
                    (baseline_rmse - final_rmse) / baseline_rmse * 100
                    if baseline_rmse
                    else np.nan
                )
                rows.append(
                    {
                        "segment_group": segment_group,
                        "segment_value": segment_value,
                        "n_hours": len(group),
                        "actual_mean": group[TARGET].mean(),
                        "baseline_RMSE": baseline_rmse,
                        "final_RMSE": final_rmse,
                        "RMSE_delta": baseline_rmse - final_rmse,
                        "RMSE_improvement_pct": rmse_improvement,
                        "baseline_MAE": baseline_mae,
                        "final_MAE": final_mae,
                        "MAE_delta": baseline_mae - final_mae,
                        "baseline_negative_predictions": int(
                            (group["baseline_prediction"] < 0).sum()
                        ),
                        "final_negative_predictions": int(
                            (group["final_prediction"] < 0).sum()
                        ),
                    }
                )
            return pd.DataFrame(rows)


        segment_results = pd.concat(
            [
                segment_metric_table(segment_frame, segment_group, segment_column)
                for segment_group, segment_column in segment_columns.items()
            ],
            ignore_index=True,
        ).sort_values(["RMSE_delta", "n_hours"], ascending=[False, False])

        display(segment_results)

        best_segment = segment_results.iloc[0]
        weakest_segment = segment_results.sort_values("RMSE_delta").iloc[0]
        segment_positive_count = int((segment_results["RMSE_delta"] > 0).sum())
        segment_total_count = len(segment_results)

        show_markdown(
            f"""
            **Расчетные итоги сегментного аудита**

            - Test-срезов, где final лучше baseline по RMSE:
              `{segment_positive_count}` из `{segment_total_count}`.
            - Самый сильный выигрыш: `{best_segment["segment_group"]}` /
              `{best_segment["segment_value"]}`.
              `n = {int(best_segment["n_hours"])}`,
              baseline RMSE `{best_segment["baseline_RMSE"]:.2f}`,
              final RMSE `{best_segment["final_RMSE"]:.2f}`,
              улучшение `{best_segment["RMSE_delta"]:.2f}` прокатов/час
              (`{best_segment["RMSE_improvement_pct"]:.2f}%`).
            - Самый слабый сегмент: `{weakest_segment["segment_group"]}` /
              `{weakest_segment["segment_value"]}`;
              `n = {int(weakest_segment["n_hours"])}`,
              изменение RMSE `{weakest_segment["RMSE_delta"]:.2f}` прокатов/час.
            """
        )
        '''
    ),
    md(
        """
        Таблица выше отсортирована по выигрышу RMSE. Для графика оставляю самые заметные срезы, чтобы не перегружать чтение. Слева видно, на сколько прокатов/час уменьшилась ошибка. Справа - абсолютный RMSE baseline и финальной модели в тех же сегментах; так понятно, речь о большом реальном улучшении или о небольшом выигрыше на легком срезе.
        """
    ),
    code(
        r'''
        plot_segments = segment_results.head(10).copy()
        segment_group_short_labels = {
            "demand_level (уровень фактического спроса)": "Уровень спроса",
            "seasons (сезон)": "Сезон",
            "holiday (праздничный день)": "Праздник",
            "functioning_day (работает ли прокат)": "Режим работы",
            "rainfallmm (количество осадков, дождь)": "Дождь",
            "snowfall_cm (количество снега)": "Снег",
            "time_period (период дня)": "Период дня",
        }
        plot_segments["segment_label"] = [
            wrap_plot_label(
                f"{segment_group_short_labels.get(group, group)}: {value}",
                width=38,
            )
            for group, value in zip(
                plot_segments["segment_group"],
                plot_segments["segment_value"],
            )
        ]

        rmse_pair_plot = plot_segments.melt(
            id_vars=["segment_label"],
            value_vars=["baseline_RMSE", "final_RMSE"],
            var_name="model",
            value_name="RMSE",
        )
        rmse_pair_plot["model"] = rmse_pair_plot["model"].map(
            {
                "baseline_RMSE": "baseline (линейная модель компании)",
                "final_RMSE": f"{best_model_name} (финальная модель)",
            }
        )

        fig, axes = plt.subplots(1, 2, figsize=(18, 8))
        sns.barplot(
            data=plot_segments,
            y="segment_label",
            x="RMSE_delta",
            ax=axes[0],
            color="#49759c",
        )
        axes[0].set_title("Где финальная модель сильнее baseline")
        axes[0].set_xlabel("Снижение RMSE, прокатов/час")
        axes[0].set_ylabel("")
        add_bar_labels(axes[0], "%.1f")

        sns.barplot(
            data=rmse_pair_plot,
            y="segment_label",
            x="RMSE",
            hue="model",
            ax=axes[1],
        )
        axes[1].set_title("Baseline и final RMSE в тех же сегментах")
        axes[1].set_xlabel("RMSE, прокатов/час")
        axes[1].set_ylabel("")
        axes[1].tick_params(axis="y", labelleft=False)
        axes[1].legend(title="Модель")

        plt.tight_layout()
        plt.show()
        '''
    ),
    md(
        """
        **Интерпретация сегментного аудита:** сегментный аудит нужен, чтобы не прятать качество модели за одной средней цифрой. Он показывает, где именно новая модель полезнее baseline. Для BikeSouth это принципиально: ошибка в тихий час и ошибка в час высокого спроса стоят по-разному. В пиковый час промах может означать нехватку велосипедов, перегруженную поддержку, неверное число сотрудников на смене или слишком позднюю реакцию на спрос.

        Поэтому я смотрю не только среднюю test-метрику, а рабочие срезы: уровень спроса, сезон, праздник, режим работы проката, дождь, снег и период дня. Если финальная модель выигрывает в таких срезах, ее польза понятна бизнесу: она помогает лучше готовиться к тем часам, где цена ошибки выше. Это уже аргумент для пилота, а не просто "у модели стало меньше RMSE".

        Слабые или отрицательные срезы тоже важны. Их нельзя прятать за общей средней метрикой: именно они должны попасть в мониторинг пилота. Практический вывод такой: финальную модель можно рассматривать как замену baseline для общего почасового прогноза, но запускать ее стоит вместе с сегментным контролем ошибок по погоде, периоду дня и режиму работы проката.
        """
    ),
    md(
        """
        <a id="stage-8"></a>
        ## Этап 8. Интерпретация и артефакты

        Теперь нужно понять, почему модель принимает такие решения, и сохранить ее так, чтобы результат можно было открыть вне ноутбука. Без этого хорошая метрика остается одноразовым экспериментом.
        """
    ),
    code(
        r'''
        if best_model_name == "decision_tree_optuna":
            feature_names = (
                final_pipeline.named_steps["preprocessor"].get_feature_names_out()
            )
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

        feature_labels = [
            feature_label_for_reader(feature)
            for feature in importance_table["feature"].astype(str).tolist()
        ]
        importance_table[
            [
                "feature_technical",
                "feature_description_ru",
                "feature_plot_label",
            ]
        ] = pd.DataFrame(
            feature_labels,
            index=importance_table.index,
        )
        display(
            importance_table[
                ["feature_technical", "feature_description_ru", "importance"]
                + (
                    ["importance_std"]
                    if "importance_std" in importance_table.columns
                    else []
                )
            ]
        )

        importance_plot = importance_table.copy()
        importance_plot["feature_plot_label_wrapped"] = importance_plot[
            "feature_plot_label"
        ].map(
            lambda label: wrap_plot_label(label, width=34),
        )
        importance_plot["rank"] = np.arange(1, len(importance_plot) + 1)
        importance_plot["positive_importance"] = importance_plot["importance"].clip(
            lower=0,
        )
        total_positive_importance = importance_plot["positive_importance"].sum()
        importance_plot["cumulative_positive_share_pct"] = (
            importance_plot["positive_importance"].cumsum()
            / total_positive_importance
            * 100
            if total_positive_importance > 0
            else np.nan
        )

        fig, axes = plt.subplots(1, 2, figsize=(18, 7))
        sns.barplot(
            data=importance_plot,
            y="feature_plot_label_wrapped",
            x="importance",
            ax=axes[0],
            color="#49759c",
        )
        axes[0].set_title(importance_title)
        axes[0].set_xlabel("Вклад признака в качество модели")
        axes[0].set_ylabel("")
        add_bar_labels(axes[0], "%.3f")

        sns.lineplot(
            data=importance_plot.head(12),
            x="rank",
            y="cumulative_positive_share_pct",
            marker="o",
            ax=axes[1],
            color="#c98256",
        )
        axes[1].set_title("Накопленная доля вклада top-признаков")
        axes[1].set_xlabel("Ранг признака по важности")
        axes[1].set_ylabel("Накопленная доля положительного вклада, %")
        axes[1].set_ylim(0, 105)
        axes[1].xaxis.set_major_locator(plt.MaxNLocator(integer=True))
        plt.tight_layout()
        plt.show()
        '''
    ),
    md(
        """
        Интерпретация готова. Дальше сохраняю не только модель, но и все, что нужно для повторного открытия: predictions, metadata, model card, manifest и исходник custom transformer.
        """
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
        predictions_frame["prediction_lower_90"] = prediction_interval_frame[
            "prediction_lower_90"
        ].to_numpy()
        predictions_frame["prediction_upper_90"] = prediction_interval_frame[
            "prediction_upper_90"
        ].to_numpy()
        predictions_frame["prediction_band"] = prediction_interval_frame[
            "prediction_band"
        ].astype(str).to_numpy()
        predictions_frame["pilot_action_band"] = prediction_interval_frame[
            "pilot_action_band"
        ].to_numpy()
        predictions_frame["interval_covered_90"] = prediction_interval_frame[
            "covered_90"
        ].to_numpy()
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
        '''
    ),
    md(
        """
        Собираю machine-readable metadata, model card и manifest. Это артефакты не для красоты: они фиксируют, что именно обучено, какими данными и с каким входным контрактом.
        """
    ),
    code(
        r'''

        model_card = {
            "project": "bike_demand_regression",
            "business_goal": (
                "прогнозировать число прокатов велосипедов за час "
                "для операционного планирования BikeSouth"
            ),
            "target": TARGET,
            "primary_metric": "RMSE",
            "secondary_metrics": ["MAE", "R2", "negative_predictions"],
            "selected_model": best_model_name,
            "selected_params": final_params,
            "test_quality": final_test_metrics,
            "baseline_test_quality": baseline_test_metrics,
            "rmse_improvement_pct_vs_baseline": float(rmse_improvement_pct),
            "training_protocol": {
                "model_selection": f"{CV_SPLITS}-fold CV только на train",
                "test_usage": "одна финальная оценка после выбора модели",
                "random_state": RANDOM_STATE,
                "optuna_trials": {"knn": N_TRIALS_KNN, "decision_tree": N_TRIALS_TREE},
            },
            "input_contract": {
                "required_columns": X_train.columns.tolist(),
                "target_column": TARGET,
                "row_grain": "одна строка = один час наблюдений велопроката",
                "not_required_at_inference": [TARGET],
                "unknown_columns_policy": (
                    "лишние колонки игнорируются текущим pipeline и должны "
                    "попасть в schema audit перед осознанным добавлением"
                ),
            },
            "feature_engineering_contract": {
                "module": COMPONENT_MODULE_NAME,
                "source_path": str(COMPONENT_MODULE_PATH.relative_to(PROJECT_ROOT)),
                "source_sha256": component_source_sha256,
                "required_names": required_component_names,
            },
            "uncertainty_checks": {
                "paired_bootstrap": bootstrap_summary.to_dict(orient="records"),
                "prediction_interval_summary": prediction_interval_summary.to_dict(
                    orient="records"
                ),
                "prediction_interval_by_band": interval_by_band.to_dict(
                    orient="records"
                ),
                "decile_audit": decile_audit.to_dict(orient="records"),
            },
            "pilot_readiness": {
                "action_rules": pilot_action_table.to_dict(orient="records"),
                "action_summary": pilot_action_summary.to_dict(orient="records"),
            },
            "future_modeling_note": (
                "После изучения ансамблей стоит проверить boosting/forest "
                "challenger-модели как потенциальные улучшатели качества; "
                "в этой работе они не используются, чтобы не выходить "
                "за рамку текущей постановки."
            ),
            "known_limitations": [
                (
                    "test-выборка относится к той же исходной среде, "
                    "что и train; это не будущий out-of-time период"
                ),
                (
                    "перед использованием в необычную погоду, праздники "
                    "или новый режим работы нужна повторная проверка"
                ),
                (
                    "часы с нулевым спросом при работающем прокате "
                    "и часы неработающего проката требуют мониторинга"
                ),
            ],
            "monitoring_recommendations": [
                (
                    "RMSE, MAE, R2 и число отрицательных прогнозов "
                    "на свежих размеченных партиях"
                ),
                "доля часов с нулевым спросом и строк с неработающим прокатом",
                (
                    "drift распределений температуры, влажности, дождя, "
                    "снега и time-period признаков"
                ),
                "ошибка прогноза по сезонам, времени суток, праздникам и functioning_day",
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
                name
                for name, _, _ in final_pipeline.named_steps[
                    "preprocessor"
                ].transformers
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
            "unknown_columns_policy": "ignore_and_report_in_schema_audit",
            "engineered_features": ENGINEERED_FEATURES,
            "component_module": COMPONENT_MODULE_NAME,
            "component_source_sha256": component_source_sha256,
            "required_component_names": required_component_names,
            "bootstrap_improvement": bootstrap_summary.to_dict(orient="records"),
            "prediction_interval_summary": prediction_interval_summary.to_dict(
                orient="records"
            ),
            "pilot_action_summary": pilot_action_summary.to_dict(orient="records"),
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
        '''
    ),
    md(
        """
        Финальный шаг по артефактам: inventory файлов и reload-проверка. Здесь проверяется, что сохраненный `joblib` действительно открывается и дает те же прогнозы.
        """
    ),
    code(
        r'''

        production_contract = pd.DataFrame(
            [
                {
                    "contract_area": "input_schema",
                    "requirement": (
                        "данные для инференса должны содержать те же "
                        "исходные признаки, что и train"
                    ),
                    "implementation": (
                        f"{len(X_train.columns)} входных колонок перечислены "
                        "в model_card['input_contract']"
                    ),
                },
                {
                    "contract_area": "feature_engineering_code",
                    "requirement": (
                        "сохраненная модель должна поставляться вместе "
                        "со всем кастомным feature engineering кодом"
                    ),
                    "implementation": (
                        f"{COMPONENT_MODULE_NAME}.py находится в проекте; "
                        f"sha256={component_source_sha256[:12]}..."
                    ),
                },
                {
                    "contract_area": "reproducibility",
                    "requirement": (
                        "предсказания артефакта после reload должны "
                        "совпадать с предсказаниями ноутбука"
                    ),
                    "implementation": "ниже выполняется joblib.load и сравнение через np.allclose",
                },
                {
                    "contract_area": "monitoring",
                    "requirement": (
                        "production-использование требует свежей разметки "
                        "и мониторинга drift"
                    ),
                    "implementation": (
                        "model card содержит контроль метрик, сегментов "
                        "и drift входных признаков"
                    ),
                },
            ]
        )

        artifact_inventory = pd.DataFrame(
            [
                {
                    "artifact": "model_pipeline",
                    "path": str(model_artifact_path.relative_to(PROJECT_ROOT)),
                    "exists": model_artifact_path.exists(),
                    "purpose": (
                        "полный sklearn pipeline с feature engineering, "
                        "preprocessing и моделью"
                    ),
                },
                {
                    "artifact": "metadata",
                    "path": str(metadata_path.relative_to(PROJECT_ROOT)),
                    "exists": metadata_path.exists(),
                    "purpose": (
                        "метрики запуска, выбранные параметры, версии "
                        "пакетов и пути артефактов"
                    ),
                },
                {
                    "artifact": "model_card",
                    "path": str(model_card_path.relative_to(PROJECT_ROOT)),
                    "exists": model_card_path.exists(),
                    "purpose": "бизнес-цель, качество, входной контракт, ограничения и мониторинг",
                },
                {
                    "artifact": "component_manifest",
                    "path": str(manifest_path.relative_to(PROJECT_ROOT)),
                    "exists": manifest_path.exists(),
                    "purpose": (
                        "имена кастомных компонентов, checksum модуля "
                        "и структура pipeline"
                    ),
                },
                {
                    "artifact": "test_predictions",
                    "path": str(predictions_path.relative_to(PROJECT_ROOT)),
                    "exists": predictions_path.exists(),
                    "purpose": (
                        "построчные test-прогнозы, residuals, "
                        "90% intervals и pilot action band"
                    ),
                },
                {
                    "artifact": "component_source_module",
                    "path": str(COMPONENT_MODULE_PATH.relative_to(PROJECT_ROOT)),
                    "exists": COMPONENT_MODULE_PATH.exists(),
                    "purpose": (
                        "Python-код, необходимый для загрузки "
                        "и работы сохраненного pipeline"
                    ),
                },
                {
                    "artifact": "baseline_pipeline",
                    "path": str(baseline_path.relative_to(PROJECT_ROOT)),
                    "exists": baseline_path.exists(),
                    "purpose": "baseline компании, использованный для сравнения на test",
                },
            ]
        )
        artifact_inventory.to_csv(artifact_inventory_path, index=False)

        reloaded_pipeline = joblib.load(model_artifact_path)
        reloaded_predictions = reloaded_pipeline.predict(X_test)
        reload_check = {
            "same_predictions_after_reload": bool(
                np.allclose(final_test_predictions, reloaded_predictions)
            ),
            "max_abs_prediction_diff": float(
                np.max(np.abs(final_test_predictions - reloaded_predictions))
            ),
        }
        artifact_check = pd.DataFrame(
            [
                {
                    "check": "component module contains required names",
                    "status": "OK" if component_symbols_ok else "FAIL",
                    "detail": (
                        f"найдено {component_symbol_check['present'].sum()} "
                        f"из {len(required_component_names)} обязательных имен"
                    ),
                },
                {
                    "check": "all listed artifacts exist",
                    "status": "OK" if bool(artifact_inventory["exists"].all()) else "FAIL",
                    "detail": (
                        f"найдено {artifact_inventory['exists'].sum()} "
                        f"из {len(artifact_inventory)} артефактов"
                    ),
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
        assert reload_check["same_predictions_after_reload"]
        assert artifact_inventory["exists"].all()
        '''
    ),
    md(
        """
        **Вывод этапа 8:** сохранен не голый алгоритм, а весь рабочий pipeline: feature engineering, обработка пропусков, кодирование категорий и финальная модель. Это важно для ревью и для будущего пилота: модель можно открыть вне ноутбука и получить те же прогнозы, а не пытаться восстановить подготовку данных по памяти.

        Рядом сохранены metadata, model card, manifest, test predictions и inventory. Эти файлы отвечают на разные вопросы: что обучено, на каких входных колонках, с какими метриками, где лежит custom transformer и какие артефакты должны существовать. После `joblib.load()` прогнозы совпали, значит модель не зависит от скрытого состояния ноутбука.

        Test predictions сохранены как отчетный артефакт финальной проверки. Их нельзя использовать для нового обучения: test уже сыграл свою роль как честная финальная оценка выбранной модели.
        """
    ),
    code(
        r'''
        baseline_test_row = final_results.query(
            "model == 'company_linear_baseline' and split == 'test'"
        ).iloc[0]
        final_test_row = final_results.query(
            "model == @best_model_name and split == 'test'"
        ).iloc[0]
        final_cv_row = cv_comparison.query("model == @best_model_name").iloc[0]

        top_feature_rows = [
            feature_label_for_reader(feature)
            for feature in importance_table["feature"].head(5).astype(str).tolist()
        ]
        top_feature_text = "; ".join(
            f"`{technical}` ({description})"
            for technical, description, _ in top_feature_rows
        )
        rmse_abs_improvement = baseline_test_row["RMSE"] - final_test_row["RMSE"]
        mae_abs_improvement = baseline_test_row["MAE"] - final_test_row["MAE"]
        r2_abs_improvement = final_test_row["R2"] - baseline_test_row["R2"]
        baseline_negative_share = (
            baseline_test_row["negative_predictions"] / len(X_test)
        )
        final_negative_share = final_test_row["negative_predictions"] / len(X_test)

        final_calculated_summary = "\n".join(
            [
                '<a id="final-conclusions"></a>',
                "",
                "# Финальные выводы",
                "",
                "## Расчетные итоги",
                "",
                f"- Выбранная модель: `{best_model_name}`.",
                (
                    "- CV train: "
                    f"`RMSE = {final_cv_row['cv_RMSE_mean']:.2f} "
                    f"± {final_cv_row['cv_RMSE_std']:.2f}`, "
                    f"`MAE = {final_cv_row['cv_MAE_mean']:.2f}`, "
                    f"`R2 = {final_cv_row['cv_R2_mean']:.3f}`."
                ),
                (
                    "- Baseline test: "
                    f"`RMSE = {baseline_test_row['RMSE']:.2f}`, "
                    f"`MAE = {baseline_test_row['MAE']:.2f}`, "
                    f"`R2 = {baseline_test_row['R2']:.3f}`."
                ),
                (
                    "- Final test: "
                    f"`RMSE = {final_test_row['RMSE']:.2f}`, "
                    f"`MAE = {final_test_row['MAE']:.2f}`, "
                    f"`R2 = {final_test_row['R2']:.3f}`."
                ),
                (
                    "- Улучшение относительно baseline: "
                    f"`RMSE -{rmse_abs_improvement:.2f}` прокатов/час "
                    f"(`{rmse_improvement_pct:.2f}%`), "
                    f"`MAE -{mae_abs_improvement:.2f}`, "
                    f"`R2 +{r2_abs_improvement:.3f}`."
                ),
                (
                    "- Отрицательные прогнозы baseline: "
                    f"`{int(baseline_test_row['negative_predictions'])}` "
                    f"из `{len(X_test)}` (`{baseline_negative_share:.1%}`), "
                    f"минимальный прогноз "
                    f"`{baseline_test_row['prediction_min']:.2f}`."
                ),
                (
                    "- Отрицательные прогнозы final: "
                    f"`{int(final_test_row['negative_predictions'])}` "
                    f"из `{len(X_test)}` (`{final_negative_share:.1%}`), "
                    f"минимальный прогноз "
                    f"`{final_test_row['prediction_min']:.2f}`, "
                    f"средний прогноз "
                    f"`{final_test_row['prediction_mean']:.2f}`."
                ),
                (
                    "- Сегментный аудит: final лучше baseline в "
                    f"`{segment_positive_count}` из `{segment_total_count}` "
                    "test-срезов; самый сильный выигрыш - "
                    f"`{best_segment['segment_group']}` / "
                    f"`{best_segment['segment_value']}` "
                    f"(`RMSE -{best_segment['RMSE_delta']:.2f}` прокатов/час)."
                ),
                (
                    "- Bootstrap 95% CI выигрыша RMSE: "
                    f"`{rmse_bootstrap_row['ci_2_5']:.2f}` - "
                    f"`{rmse_bootstrap_row['ci_97_5']:.2f}` прокатов/час; "
                    "доля положительного выигрыша "
                    f"`{rmse_bootstrap_row['share_positive']:.1%}`."
                ),
                (
                    "- 90% prediction interval на test покрывает "
                    f"`{interval_summary_row['test_coverage']:.1%}` "
                    "фактических значений; средняя ширина interval "
                    f"`{interval_summary_row['mean_width']:.2f}` прокатов/час."
                ),
                (
                    "- Pilot readiness: test-часов в зоне "
                    f"`capacity_watch` = `{capacity_watch_hours}`."
                ),
                f"- Ключевые признаки: {top_feature_text}.",
                f"- Параметры финальной модели: `{final_params}`.",
                (
                    "- Дополнительный пункт закрыт через "
                    "`BikeFeatureEngineer` внутри `Pipeline`; transformer "
                    "сохранен в импортируемом модуле и проверен после "
                    "`joblib.load()`."
                ),
                (
                    "- Потенциальный следующий шаг после изучения ансамблей: "
                    "проверить boosting/forest challenger-модели как "
                    "улучшатели качества, не смешивая это с текущим "
                    "честным baseline-сравнением."
                ),
            ]
        )

        show_markdown(final_calculated_summary)
        '''
    ),
    md(
        """
        ## Бизнес-интерпретация

        Для BikeSouth результат можно читать так: финальная модель стала лучше текущей линейной baseline-модели в той единице, в которой бизнес принимает решения, - в прокатах велосипедов за час. Ошибка стала ниже, а невозможные отрицательные прогнозы исчезли. Это значит, что прогноз уже не нужно "чинить руками" перед тем, как обсуждать его с операционной командой.

        Практический смысл модели - заранее видеть нагрузку по часам. Если ожидается высокий спрос, команда может раньше подготовить парк, смены и поддержку. Если спрос низкий, не нужно держать лишний запас и людей "на всякий случай". Модель не обещает идеально угадать каждый час, поэтому к точечному прогнозу добавлены интервалы: они показывают не только ожидаемое значение, но и верхнюю границу риска для пилота.

        Отдельно проверено, где именно появляется выигрыш. Это важнее одной средней цифры: прокату нужно понимать, помогает ли модель в пиковом спросе, в дождь/снег, ночью, в праздники и при закрытом прокате. Сегментный аудит, decile-аудит и таблица pilot actions превращают метрику в рабочий список для пилота: сильные сегменты можно использовать увереннее, слабые - заранее поставить на мониторинг.

        Дополнительный пункт закрыт инженерно: новые признаки создаются не ручным кодом перед обучением, а внутри `BikeFeatureEngineer` в составе `Pipeline`. При повторном запуске, сохранении и загрузке используются те же правила подготовки данных. Проверка `joblib.load()` подтверждает, что модель не держится на скрытом состоянии ноутбука.

        Рекомендация: брать модель в пилот для общего почасового прогноза спроса, но не считать ее готовой системой распределения велосипедов по станциям. В данных нет запасов на конкретных станциях, городских событий, цен, ремонтов и отдельной out-of-time проверки на будущем месяце. Перед автоматическим использованием нужно прогнать модель на более свежем периоде и отдельно посмотреть пиковые часы, дождь, снег, нулевой спрос и часы с неработающим прокатом. Все артефакты для такого пилота сохранены: pipeline, metadata, model card, manifest, predictions и inventory.

        Отдельное направление роста - ансамбли и boosting. После того как эти методы будут пройдены в программе, их стоит проверить как challenger-модели: случайный лес, extremely randomized trees и gradient boosting часто хорошо усиливают табличные нелинейные задачи. В текущем решении я их не добавляю намеренно: работа остается в рамках изученных моделей, но честно показывает, куда двигаться дальше для дополнительного прироста качества.
        """
    ),
]

def write_notebook() -> None:
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


if __name__ == "__main__":
    write_notebook()
