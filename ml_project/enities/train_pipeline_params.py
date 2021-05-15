from dataclasses import dataclass
from ml_project.enities.params_for_save import SaveParams
from ml_project.enities.features_params import FeaturesParams
from ml_project.enities.models_params import ModelsParams
from ml_project.enities.split_params import SplitParams
from ml_project.enities.classifier_params import ClassifierParams
from marshmallow_dataclass import class_schema
import yaml


@dataclass()
class TrainPipelineParams:
    input_data_path: str
    metric_path: str
    params_for_save: SaveParams
    features_params: FeaturesParams
    models_params: ModelsParams
    class_params: ClassifierParams
    split_params: SplitParams


TrainingPipelineParamsSchema = class_schema(TrainPipelineParams)


def read_training_pipeline_params(path: str) -> TrainPipelineParams:
    with open(path, "r") as input_stream:
        schema = TrainingPipelineParamsSchema()
        return schema.load(yaml.safe_load(input_stream))