from dataclasses import dataclass
from ml_project.enities.params_for_save import SaveParams
from ml_project.enities.features_params import FeaturesParams
from ml_project.enities.models_params import ModelsParams
from ml_project.enities.classifier_params import ClassifierParams
from marshmallow_dataclass import class_schema
import yaml


@dataclass()
class PredictPipelineParams:
    input_data_path: str
    metric_path: str
    prediction_path: str
    params_for_save: SaveParams
    features_params: FeaturesParams
    models_params: ModelsParams
    class_params: ClassifierParams



PredictPipelineParamsSchema = class_schema(PredictPipelineParams)


def read_predict_pipeline_params(path: str) -> PredictPipelineParamsSchema:
    with open(path, "r") as input_stream:
        schema = PredictPipelineParamsSchema()
        return schema.load(yaml.safe_load(input_stream))