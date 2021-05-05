import os
import numpy as np
import pandas as pd
import pytest
from faker import Faker


@pytest.fixture()
def input_data_path():
    curdir = os.path.dirname(__file__)
    LENGTH = 300
    binary_cols = ["fbs", "exang", "sex", "target"]
    num_cols = ["chol", "trestbps", "thalach", "age"]
    discrete_cols = ["ca", "cp", "restecg", "thal", "slope"]
    dict_data = {}
    for col, val in zip(binary_cols, [0.14, 0.32, 0.68, 0.54]):
        dict_data[col] = np.random.binomial(1, val, LENGTH)

    for col, val in zip(num_cols, [[246.6, 51.8],
                                   [131.6, 17.5],
                                   [149.6, 22.9],
                                   [54.4, 9.08]]):
        dict_data[col] = np.random.normal(val[0], val[1], LENGTH)

    for col, val in zip(discrete_cols, [[0.581, 0.2, 0.13, 0.07, 0.019],
                                        [0.47, 0.29, 0.17, 0.07],
                                        [0.5, 0.49, 0.01],
                                        [0.551, 0.38, 0.06, 0.009],
                                        [0.47, 0.46, 0.07]]):
        dict_data[col] = np.random.choice(range(len(val)), size=LENGTH,
                                          p=val)
    dict_data["oldpeak"] = np.random.noncentral_chisquare(1.16, 1.03, LENGTH)
    pd.DataFrame.from_dict(dict_data).to_csv(os.path.join(curdir, "train_data_sample.csv"), index=False)
    return os.path.join(curdir, "train_data_sample.csv")


@pytest.fixture()
def path_to_save():
    currdir = os.path.dirname(__file__)
    return os.path.join(currdir, "configs/predict_conf.yaml")


@pytest.fixture()
def abs_path():
    currdir = os.path.dirname(__file__)
    return currdir


@pytest.fixture()
def target_col():
    return "target"


@pytest.fixture()
def dict_class_params() -> dict:
    return {"type": "LogisticRegression",
            "loss": "log",
            "penalty": "l2",
            "alpha": 0.0001,
            "max_iter": 10000,
            "type_save": ""}


@pytest.fixture()
def dict_features_params() -> dict:
    return {
        "binary_cols": ["fbs", "sex", "exang"],
        "categorical_cols": [],
        "numerical_cols": ["age", "thalach",
                           "oldpeak", "chol",
                           "trestbps", "cp",
                           "thal", "slope",
                           "ca", "restecg"],
        "target_col": ""}


@pytest.fixture()
def fake_data():
    fake = Faker()
    LENGTH = 300
    REPEAT = 5
    COUNT_NUMS = 3
    COUNT_BINARY = 4
    dict_data = {"1": [fake.city() for _ in range(REPEAT)] * (LENGTH // REPEAT),
                 "target": np.random.binomial(1, np.random.rand(), LENGTH)}
    i = 2
    for i in range(2, COUNT_BINARY + 3):
        dict_data[str(i)] = np.random.binomial(1, np.random.rand(), LENGTH)

    for j in range(i, COUNT_NUMS + i):
        dict_data[str(j)] = np.random.normal(np.random.randn(), np.random.rand(), LENGTH)
    return pd.DataFrame.from_dict(dict_data)


