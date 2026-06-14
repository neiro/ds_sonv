# DS: bike rental demand regression

Основной ноутбук проекта: `ml-env/work.ipynb`.

Цель проекта - построить и сравнить нелинейные регрессионные модели для
прогнозирования числа прокатов велосипедов за час по погодным и
календарным признакам. Базовая модель компании - готовый линейный pipeline,
с которым нужно сравнить улучшенные модели.

`bike_demand_pipeline_components.py` содержит кастомный transformer, который
используется сохраненным pipeline. Файл нужен для корректной загрузки
`models/bike_demand_model.joblib` в чистом Python-процессе.

## Данные и baseline

Локальные файлы задачи сохраняются в `data/raw/` и `models/`, но не
коммитятся в GitHub:

```text
data/raw/ds_s14_train_data.csv
data/raw/ds_s14_test_data.csv
models/baseline_linear_regression_pipeline.joblib
```

Источники:

```text
https://code.s3.yandex.net/datasets/ds_s14_train_data.csv
https://code.s3.yandex.net/datasets/ds_s14_test_data.csv
https://code.s3.yandex.net/data-scientist/baseline_linear_regression_pipeline.joblib
```

## Как воспроизвести окружение

Используемая версия Python: `3.11.9`.

```powershell
py -3.11 -m venv ml-env
.\ml-env\Scripts\python.exe -m pip install --upgrade pip
.\ml-env\Scripts\python.exe -m pip install -r requirements.txt
.\ml-env\Scripts\python.exe -m ipykernel install --user --name ds-sonv-bike-regression --display-name "Python (ds-sonv-bike-regression)"
```

Для совместимости с предоставленным baseline зафиксирован
`scikit-learn==1.6.1`. На `scikit-learn==1.7.0` файл
`baseline_linear_regression_pipeline.joblib` не загружается.

## План работы

- загрузить train/test выборки и готовый baseline pipeline;
- оценить baseline по `RMSE`, `MAE` и `R2`;
- провести EDA целевой переменной `Rented Bike Count` и признаков погоды;
- подготовить признаки без утечек из тестовой выборки;
- обучить и настроить `KNeighborsRegressor` и `DecisionTreeRegressor`;
- выполнить подбор гиперпараметров через Optuna;
- сравнить модели с baseline на финальном test;
- интерпретировать важность признаков и сформулировать выводы.
