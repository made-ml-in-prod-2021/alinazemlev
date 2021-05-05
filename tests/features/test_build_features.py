import os
import sys
PATH = os.getcwd()
sys.path.insert(0, PATH)

import pandas as pd
from py._path.local import LocalPath
from project.features.build_features import build_pipeline, extract_target
import logging
from project.enities import (
    SplitParams,
    FeaturesParams,
    ModelsParams,
    ClassifierParams,
    SaveParams,
    TrainPipelineParams

)


def test_build_pipeline(tmpdir: LocalPath, input_data_path: str, target_col: str,
                        abs_path: str, dict_class_params: dict, path_to_save: str,
                        fake_data: pd.DataFrame):
    logger = logging.Logger("Test")
    expected_metric_path = tmpdir.join("metrics.json")
    params = TrainPipelineParams(
        input_data_path=input_data_path,
        metric_path=expected_metric_path,
        split_params=SplitParams(val_size=0.1, random_state=1),
        features_params=FeaturesParams(
            numerical_cols=[],
            categorical_cols=[],
            binary_cols=[],
            target_col=target_col,
        ),
        models_params=ModelsParams(
            scaling_path=os.path.join(abs_path, "models/scaling.pkl"),
            imputer_path=os.path.join(abs_path, "models/imputer.pkl"),
            categorical_vectorizer_path=os.path.join(abs_path, "models/dv_x.pkl"),
            classifier_path_postfix=os.path.join(abs_path, "models/classifier.pkl"),
            fill_empty=-111,
        ),
        class_params=ClassifierParams(**dict_class_params),
        params_for_save=SaveParams(
            path_to_save=path_to_save
        )
    )
    pipe = build_pipeline(params, logger)
    X = pipe.fit_transform(fake_data)
    assert X.shape[1] == 12
    assert (X.mean() <= 1).all()
    assert (X.mean() >= 0).all()


def test_extract_target(target_col: str, fake_data: pd.DataFrame):
    features_params = FeaturesParams(
        numerical_cols=[],
        categorical_cols=[],
        binary_cols=[],
        target_col=target_col,
    )
    y = extract_target(fake_data, features_params)
    assert sorted(y.unique().tolist()) == [0, 1]
