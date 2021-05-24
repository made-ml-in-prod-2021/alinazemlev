ml_project
==============================

ml project

Installation: 
~~~
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
~~~
Usage:
* Train: You can choose configuration for training model. 
  "train_config_sgd" runs SGDCLassifier and "train_config_log" runs LogisticRegression.
  At the training stage, the types of input data are analyzed. 
  Information about this is recorded in the prediction config. Therefore, the fields training config remain empty.
~~~
python ml_project/train/train_model.py configs/train_config_sgd.yaml #SGDClassifier
python ml_project/train/train_model.py configs/train_config_log.yaml #LogisticRegression
~~~
* Predict: Similarly you can choose configuration for prediction. 
  Each predictive configuration corresponds to its own model in training phase. At the prediction stage the information obtained during training about the types of variables is used
~~~
python ml_project/predict/predict_model.py configs/predict_config_sgd.yaml #SGDClassifier
python ml_project/predict/predict_model.py configs/predict_config_log.yaml #LogisticRegression
~~~
Test:
~~~
pytest tests/
~~~

Project Organization
------------

    ├── LICENSE
    ├── Makefile           <- Makefile with commands like `make data` or `make train`
    ├── README.md          <- The top-level README for developers using this project.
    ├── configs
    ├── ml_project            <- Source code and data for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── data           <- Code to download or generate data
    │   │   └── raw        <- The original, immutable data dump.  
    │   │
    │   ├── features       <- Code to turn raw data into features for modeling
    │   │
    │   ├── models         <- Code to build models 
    │   │
    │   ├── enities        <- additional enities which are used to load the configuration
    │   │
    │   ├── train          <- Code for training the models
    │   │
    │   └── predict        <- Code for prediction the models        
    │
    ├── tests              <- Code for testings modules and pipelines
    │   │
    │   ├── data           <- Code for testing data creation
    │   │
    │   ├── features       <- Code for testing features creation
    │   │
    │   ├── models         <- Code for testing models creation 
    │   │
    │   ├── configs        <- temporarily used for saving configs
    │   │
    │   └── predict        <- temporarily used for saving prediction
    │
    ├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                         the creator's initials, and a short `-` delimited description, e.g.
    │                         `1.0-jqp-initial-data-exploration`.
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
    └── tox.ini            <- tox file with settings for running tox; see tox.readthedocs.io


--------
**Самоценка**
1. В описании к пул реквесту описаны основные архитектурные и тактические решения. 2 балла
2. Ноутбук закоммичен в папку с ноутбуками - 2 балла
3. Проект имеет модульную структуру - 2 балла
4. Использованы логгеры - 2 балла
5. Написаны тесты на модули и весь паплайн - 3 балла
6. Для тестов в ряде случаев генерируются синтетические данные (https://faker.readthedocs.io/en/) - 3 балла
7. Обучение конфигурируется с помощью конфигов в yaml. Есть две конфигурации для обучения и две для предсказания - 3 балла
8. Используются датаклассы - 3 балла
9. Используются кастомные трансформеры - 3 балла
10. Описано выше как запускать обучение модели - 3 балла
11. Есть функция predict, которая принимает на вход артефакты от обучения( меня это информация о типах переменных, которая сохранаяется в в конфиге на стадии обучения  и сериализованные объекты)  - 3 балла
12. Самооценка - 1 балл
--------
Итого: 30 баллов.
