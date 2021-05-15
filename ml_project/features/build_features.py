import logging
from dataclasses import asdict
import yaml
import pandas as pd
import numpy as np
from typing import List, Union, Dict
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction import DictVectorizer
from sklearn.pipeline import Pipeline, FeatureUnion
from ml_project.enities.train_pipeline_params import TrainPipelineParams
from ml_project.enities.predict_pipeline_params import PredictPipelineParams

from ..enities.features_params import FeaturesParams
from ..enities.params_for_save import SaveParams
from ..enities.models_params import ModelsParams
import joblib
from sklearn.impute import SimpleImputer

PARAMS = Union[TrainPipelineParams, PredictPipelineParams]


def build_pipeline(pipeline_params: PARAMS, logger: logging.Logger) -> Pipeline:
    num_pipeline = Pipeline([("imp", FillMissing(pipeline_params.models_params)),
                             ("scale", Scaling(pipeline_params.models_params))])

    feature = FeatureUnion([("num", num_pipeline),
                            ("bin", BinaryPrepare(pipeline_params.models_params, logger)),
                            ("cat", CategoricalPrepare(pipeline_params.models_params))])

    pipe = Pipeline([('create', AnalyzeColumns(pipeline_params.features_params,
                                               pipeline_params.params_for_save)),
                     ('feat', feature),
                     ("drop", DropEmpty(pipeline_params.models_params))])

    return pipe


def extract_target(df: pd.DataFrame, params: FeaturesParams) -> pd.Series:
    target = df[params.target_col]
    return target


class AnalyzeColumns(BaseEstimator, TransformerMixin):
    def __init__(self, params: FeaturesParams,
                 params_for_saving: SaveParams):
        self.params = params
        self.params_for_saving = params_for_saving

    @staticmethod
    def save_yaml(params: FeaturesParams, path: str):
        config = yaml.load(open(path, "r"), Loader=yaml.FullLoader)
        config['features_params'] = asdict(params)
        yaml.dump(config, open(path, "w"))

    @staticmethod
    def check_columns(X: pd.DataFrame, columns) -> bool:
        if len(columns) != 0:
            try:
                X[columns]
            except KeyError:
                return False
        return True

    def fit(self, X: pd.DataFrame) -> "AnalyzeColumns":
        try:
            X[self.params.target_col]
        except KeyError:
            raise KeyError("Column with target not found")

        columns_without_target = set(X.columns) - {self.params.target_col}
        features = X[columns_without_target]

        self.params.numerical_cols = features.dtypes[features.dtypes == float].index.values.tolist()

        for column in features.dtypes[features.dtypes != float].index.values.tolist():
            if features[column].dtype == str or features[column].dtype == object:
                self.params.categorical_cols.append(column)
            elif features[column].dtype == int:
                if set(features[column]) == {0, 1}:
                    self.params.binary_cols.append(column)
                else:
                    self.params.numerical_cols.append(column)
            else:
                continue
        if len(self.params.numerical_cols + self.params.categorical_cols + self.params.binary_cols) == 0:
            raise Exception("valid columns not found")

        AnalyzeColumns.save_yaml(self.params, self.params_for_saving.path_to_save)
        return self

    def transform(self, X: pd.DataFrame) -> Dict[str, Union[FeaturesParams, pd.DataFrame]]:
        for cols in [self.params.numerical_cols,
                     self.params.categorical_cols,
                     self.params.binary_cols]:
            if not AnalyzeColumns.check_columns(X, cols):
                raise Exception(f"{cols} not found in columns of dataset")
        return {"params": self.params,
                "data": X}


class FillMissing(BaseEstimator, TransformerMixin):
    def __init__(self, models_params: ModelsParams):
        self.models_params = models_params

    def fit(self, X: Dict[str, Union[FeaturesParams, pd.DataFrame]]) -> "FillMissing":
        if X["params"].numerical_cols:
            imp = SimpleImputer(missing_values=np.nan, strategy='mean')
            imp.fit(X["data"][X["params"].numerical_cols])
            joblib.dump(imp, self.models_params.imputer_path)
        return self

    def transform(self, X: Dict[str, Union[FeaturesParams, pd.DataFrame]]) -> np.array:
        if X["params"].numerical_cols:
            try:
                imp = joblib.load(self.models_params.imputer_path)
            except FileNotFoundError:
                raise FileNotFoundError(f"Not found {self.models_params.imputer_path}. Check numerical cols on train")

            return imp.transform(X["data"][X["params"].numerical_cols])
        return np.array([self.models_params.fill_empty] * len(X["data"])).reshape(-1, 1)


class Scaling(BaseEstimator, TransformerMixin):
    def __init__(self, models_params: ModelsParams):
        self.models_params = models_params

    def fit(self, X: np.array) -> "Scaling":
        if not (X == self.models_params.fill_empty).all():
            scaler = StandardScaler()
            scaler.fit(X)
            joblib.dump(scaler, self.models_params.scaling_path)
        return self

    def transform(self, X: np.array) -> np.array:
        if not (X == self.models_params.fill_empty).all():
            try:
                scaler = joblib.load(self.models_params.scaling_path)
            except FileNotFoundError:
                raise FileNotFoundError(f"Not found {self.models_params.scaling_path}. Check numerical cols on train")

            return scaler.transform(X)
        return np.array([self.models_params.fill_empty] * len(X)).reshape(-1, 1)


class BinaryPrepare(BaseEstimator, TransformerMixin):
    def __init__(self, models_params: ModelsParams, logger: logging.Logger):
        self.models_params = models_params
        self.logger = logger

    def fit(self, X: Dict[str, Union[FeaturesParams, pd.DataFrame]]) -> "BinaryPrepare":
        return self

    def check_values(self, X_subset: pd.DataFrame) -> pd.DataFrame:
        if ((X_subset.values < 0) | (X_subset.values > 1)).any():
            self.logger.warning('Binary cols have unknown values. '
                                'Check it out on the prediction')

            X_subset[(X_subset.values < 0)] = 0
            X_subset[(X_subset.values > 1)] = 1
        return X_subset

    def transform(self, X: Dict[str, Union[FeaturesParams, pd.DataFrame]]):
        if X["params"].binary_cols:
            return self.check_values(X["data"][X["params"].binary_cols]).values
        return np.array([self.models_params.fill_empty] * len(X["data"])).reshape(-1, 1)


class CategoricalPrepare(BaseEstimator, TransformerMixin):
    def __init__(self, models_params: ModelsParams):
        self.models_params = models_params

    def fit(self, X: Dict[str, Union[FeaturesParams, pd.DataFrame]]) -> "CategoricalPrepare":
        if X["params"].categorical_cols:
            dv_X = DictVectorizer(sparse=False)
            dv_X.fit(X["data"][X["params"].categorical_cols].to_dict(orient='records'))
            joblib.dump(dv_X, self.models_params.categorical_vectorizer_path)
        return self

    def transform(self, X: Dict[str, Union[FeaturesParams, pd.DataFrame]]):
        if X["params"].categorical_cols:
            try:
                dv_X = joblib.load(self.models_params.categorical_vectorizer_path)
            except FileNotFoundError:
                raise FileNotFoundError(
                    f"Not found {self.models_params.categorical_vectorizer_path}. Check categorical cols on train")

            return dv_X.transform(X["data"][X["params"].categorical_cols].to_dict(orient='records'))
        return np.array([self.models_params.fill_empty] * len(X["data"])).reshape(-1, 1)


class DropEmpty(BaseEstimator, TransformerMixin):
    def __init__(self, models_params: ModelsParams):
        self.models_params = models_params

    def fit(self, X: np.array) -> "DropEmpty":
        return self

    def get_indexes(self, X: np.array) -> List:
        indexes = []
        for i in range(X.shape[1]):
            if (X[:, i] == self.models_params.fill_empty).all():
                continue
            indexes.append(i)
        if len(indexes) == 0:
            raise Exception("Check columns on prediction")
        return indexes

    def transform(self, X: np.array) -> np.array:
        return X[:, self.get_indexes(X)]
