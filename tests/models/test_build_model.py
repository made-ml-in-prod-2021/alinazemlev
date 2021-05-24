import os
import sys

PATH = os.getcwd()
sys.path.insert(0, PATH)
import numpy as np
import pandas as pd
from ml_project.models import init_model, Model, get_clf
from ml_project.enities import ModelsParams, ClassifierParams


def test_init_model(dict_class_params: dict, abs_path: str):
    params = ModelsParams(
        scaling_path=os.path.join(abs_path, "models/scaling.pkl"),
        imputer_path=os.path.join(abs_path, "models/imputer.pkl"),
        categorical_vectorizer_path=os.path.join(abs_path, "models/dv_x.pkl"),
        classifier_path_postfix=os.path.join(abs_path, "models/classifier.pkl"),
        fill_empty=-111,
    )
    class_params = ClassifierParams(**dict_class_params)
    model = init_model(params, class_params)
    assert not model.clf


def test_type_model(dict_class_params: dict, dict_model_params: dict):
    params = ModelsParams(**dict_model_params)
    class_params = ClassifierParams(**dict_class_params)
    model = Model(params, class_params)
    type_model = get_clf(model.class_params)
    assert type_model[-1] == "sgd_" or type_model[-1] == "log_"


def test_fit_model(dict_class_params: dict, dict_model_params: dict, abs_path: str):
    params = ModelsParams(**dict_model_params)
    class_params = ClassifierParams(**dict_class_params)
    model = Model(params, class_params)
    X = np.array([[-1, -1], [-2, -1], [1, 1], [2, 1]])
    Y = np.array([1, 1, 2, 2])
    model.fit(X, Y)
    model.predict([[-0.8, -1]], os.path.join(abs_path, "prediction.csv"))
    assert 1 == pd.read_csv(os.path.join(abs_path, "prediction.csv"), index_col=0).iloc[0, 0]


