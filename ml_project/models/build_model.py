import logging
import json
import pandas as pd
from typing import List, Union
import numpy as np
from sklearn.linear_model import SGDClassifier, LogisticRegression
from sklearn.metrics import f1_score, accuracy_score, roc_auc_score

from ..enities.classifier_params import ClassifierParams
from ..enities.models_params import ModelsParams
import joblib


def init_model(model_params: ModelsParams, class_params: ClassifierParams) -> "Model":
    return Model(model_params, class_params)


def get_clf(class_params) -> List:
    dict_models = {
        "SGDClassifier":
            [SGDClassifier(max_iter=class_params.max_iter,
                           alpha=class_params.alpha,
                           penalty=class_params.penalty,
                           loss=class_params.loss), "sgd_"],

        "LogisticRegression": [LogisticRegression(max_iter=class_params.max_iter,
                                                  penalty=class_params.penalty, ), "log_"]

    }
    if class_params.type not in dict_models.keys():
        raise Exception(f"The model type: {class_params.type} is not supported.")
    return dict_models[class_params.type]


class Model:
    def __init__(self, models_params: ModelsParams,
                 class_params: ClassifierParams):
        self.models_params = models_params
        self.class_params = class_params
        self.clf = None

    @staticmethod
    def get_name(prefix: str, path: str) -> str:
        splits = path.split("/")
        name = prefix + splits[-1]
        return "/".join(splits[:-1]) + "/" + name

    def fit(self, X: np.array, y: np.array) -> "Model":
        clf, prefix = get_clf(self.class_params)
        clf.fit(X, y)
        joblib.dump(clf, Model.get_name(prefix, self.models_params.classifier_path_postfix))
        self.clf = clf
        return self

    def load_from_file(self):
        _, prefix = get_clf(self.class_params)
        self.clf = joblib.load(Model.get_name(prefix, self.models_params.classifier_path_postfix))

    def evaluate(self, X: np.array, y: np.array,
                 metric_path: str, logger: logging.Logger):

        if not self.clf:
            self.load_from_file()

        y_pred = self.clf.predict_proba(X)[:, 1:]
        y_pred_labels = self.clf.predict(X)

        dict_metrics = {
            "f1_score": f1_score(y, y_pred_labels),
            "accuracy": accuracy_score(y, y_pred_labels),
            "roc_auc": roc_auc_score(y, y_pred),
        }
        with open(metric_path, "w") as metric_file:
            json.dump(dict_metrics, metric_file)

        logger.info(f"metrics is {dict_metrics}")

    def predict(self, X: np.array, prediction_path: str):
        if not self.clf:
            self.load_from_file()

        if self.class_params.type_save == "probs":
            y_pred = self.clf.predict_proba(X)[:, 1:]
        else:
            y_pred = self.clf.predict(X)

        pd.DataFrame(y_pred).to_csv(prediction_path)

