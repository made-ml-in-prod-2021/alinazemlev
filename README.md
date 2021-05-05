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
* Train: You can choose configuration for training model. "train_config_sgd" runs SGDCLassifier and "train_config_log" runs LogisticRegression
~~~
python project/train/train_model.py configs/train_config_sgd.yaml #SGDClassifier
python project/train/train_model.py configs/train_config_log.yaml #LogisticRegression
~~~
* Predict: Similarly you can choose configuration for prediction. Each predictive configuration corresponds to its own model in training phase.
~~~
python project/predict/predict_model.py configs/predict_config_sgd.yaml #SGDClassifier
python project/predict/predict_model.py configs/predict_config_log.yaml #LogisticRegression
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
    ├── project            <- Source code and data for use in this project.
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

