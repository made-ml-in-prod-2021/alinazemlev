import pandas as pd
import numpy as np
from typing import List
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction import DictVectorizer
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.linear_model import LogisticRegression
from sklearn.impute import SimpleImputer


class BuilderPipe:
    def __init__(self):
        num_pipeline = Pipeline([("imp", FillMissing()),
                                 ("scale", Scaling())])

        feature = FeatureUnion([("num", num_pipeline),
                                ("bin", BinaryPrepare()),
                                ("cat", CategoricalPrepare())])

        pipe = Pipeline([('create', AnalyzeColumns()),
                         ('feat', feature),
                         ("drop", DropEmpty()),
                         ("clf", LogisticRegression())])
        self.pipe = pipe


class AnalyzeColumns(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.params = {"categorical_cols": [], "binary_cols": [],
                       "numerical_cols": [], "target_col": "target"}

    @staticmethod
    def check_columns(X: pd.DataFrame, columns) -> bool:
        if len(columns) != 0:
            try:
                X[columns]
            except KeyError:
                return False
        return True

    def fit(self, X: pd.DataFrame, y=None) -> "AnalyzeColumns":
        features = X.copy()

        self.params["numerical_cols"] = features.dtypes[features.dtypes == float].index.values.tolist()

        for column in features.dtypes[features.dtypes != float].index.values.tolist():
            if features[column].dtype == str or features[column].dtype == object:
                self.params["categorical_cols"].append(column)
            elif features[column].dtype == int:
                if set(features[column]) == {0, 1}:
                    self.params["binary_cols"].append(column)
                else:
                    self.params["numerical_cols"].append(column)
            else:
                continue
        if len(self.params["numerical_cols"] + self.params["categorical_cols"] + self.params["binary_cols"]) == 0:
            raise Exception("valid columns not found")

        # AnalyzeColumns.save_yaml(self.params, self.params_for_saving.path_to_save)
        return self

    def transform(self, X: pd.DataFrame):
        for cols in [self.params["numerical_cols"],
                     self.params["categorical_cols"],
                     self.params["binary_cols"]]:
            if not AnalyzeColumns.check_columns(X, cols):
                raise Exception(f"{cols} not found in columns of dataset")
        return {"params": self.params,
                "data": X}


class FillMissing(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.imp = None

    def fit(self, X, y=None) -> "FillMissing":
        if X["params"]["numerical_cols"]:
            imp = SimpleImputer(missing_values=np.nan, strategy='mean')
            imp.fit(X["data"][X["params"]["numerical_cols"]])
            self.imp = imp
        return self

    def transform(self, X) -> np.array:
        if X["params"]["numerical_cols"]:
            return self.imp.transform(X["data"][X["params"]["numerical_cols"]])
        return np.array([-111] * len(X["data"])).reshape(-1, 1)


class Scaling(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.scaler = None

    def fit(self, X: np.array, y=None) -> "Scaling":
        if not (X == -111).all():
            scaler = StandardScaler()
            scaler.fit(X)
            self.scaler = scaler
        return self

    def transform(self, X: np.array) -> np.array:
        if not (X == -111).all():
            return self.scaler.transform(X)
        return np.array([-111] * len(X)).reshape(-1, 1)


class BinaryPrepare(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None) -> "BinaryPrepare":
        return self

    def check_values(self, X_subset: pd.DataFrame) -> pd.DataFrame:
        X_subset[(X_subset.values < 0)] = 0
        X_subset[(X_subset.values > 1)] = 1
        return X_subset

    def transform(self, X):
        if X["params"]["binary_cols"]:
            return self.check_values(X["data"][X["params"]["binary_cols"]]).values
        return np.array([-111] * len(X["data"])).reshape(-1, 1)


class CategoricalPrepare(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.dv_X = None

    def fit(self, X, y=None) -> "CategoricalPrepare":
        if X["params"]["categorical_cols"]:
            dv_X = DictVectorizer(sparse=False)
            dv_X.fit(X["data"][X["params"]["categorical_cols"]].to_dict(orient='records'))
            self.dv_X = dv_X
        return self

    def transform(self, X):
        if X["params"]["categorical_cols"]:
            return self.dv_X.transform(X["data"][X["params"]["categorical_cols"]].to_dict(orient='records'))
        return np.array([-111] * len(X["data"])).reshape(-1, 1)


class DropEmpty(BaseEstimator, TransformerMixin):
    def fit(self, X: np.array, y=None) -> "DropEmpty":
        return self

    def get_indexes(self, X: np.array) -> List:
        indexes = []
        for i in range(X.shape[1]):
            if (X[:, i] == -111).all():
                continue
            indexes.append(i)
        if len(indexes) == 0:
            raise Exception("Check columns on prediction")
        return indexes

    def transform(self, X: np.array) -> np.array:
        return X[:, self.get_indexes(X)]
