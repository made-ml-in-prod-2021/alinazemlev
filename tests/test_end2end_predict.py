import os
import pandas as pd
from py._path.local import LocalPath
import sys
PATH = os.getcwd()
sys.path.insert(0, PATH)

from ml_project.predict import (
    predict_pipeline,

)

from ml_project.enities import (
    SplitParams,
    FeaturesParams,
    ModelsParams,
    ClassifierParams,
    SaveParams,
    TrainPipelineParams,
    PredictPipelineParams

)

from ml_project.train import (
    train_pipeline,

)


def test_train_e2e(
        tmpdir: LocalPath,
        input_data_path: str,
        target_col: str,
        path_to_save: str,
        dict_class_params: dict,
        abs_path: str,
        dict_features_params: dict,
):
    expected_model_path = os.path.join(abs_path, "models/log_classifier.pkl")
    expected_prediction_path = os.path.join(abs_path, "predict/prediction.csv")

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
    train_pipeline(params)
    pred_params = PredictPipelineParams(
        input_data_path=input_data_path,
        metric_path=expected_metric_path,
        prediction_path=expected_prediction_path,
        features_params=FeaturesParams(**dict_features_params),
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
    predict_pipeline(pred_params)
    assert os.path.exists(expected_prediction_path)
    assert os.path.exists(expected_model_path)
    assert pd.read_csv(expected_prediction_path).shape[0] > 1
    assert os.path.exists(params.metric_path)
