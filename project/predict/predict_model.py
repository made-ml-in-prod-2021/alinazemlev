import logging
import click
import os
import sys
PATH = os.getcwd()
sys.path.insert(0, PATH)
from project.enities.predict_pipeline_params import PredictPipelineParams, read_predict_pipeline_params
from project.features.build_features import build_pipeline
from project.enities.logger_params import setup_logging
from project.data.make_dataset import read_data
from project.models.build_model import init_model

APPLICATION_NAME = "project"
WARN_APPLICATION_NAME = "project_warn"
LOGGING_YAML = PATH+ "/configs/logger_conf.yaml"

logger = logging.getLogger(APPLICATION_NAME)
logger_2 = logging.getLogger(WARN_APPLICATION_NAME)
setup_logging(LOGGING_YAML)


def predict_pipeline(predict_pipeline_params: PredictPipelineParams):
    logger.info(f"start predict pipeline with params {predict_pipeline_params}")

    data = read_data(predict_pipeline_params.input_data_path)
    logger.info(f"data.shape is {data.shape}")

    pipe = build_pipeline(predict_pipeline_params, logger_2)

    X = pipe.transform(data)
    logger.info(f"predict_features.shape is {X.shape}")

    model = init_model(predict_pipeline_params.models_params,
                       predict_pipeline_params.class_params)
    model.predict(X, predict_pipeline_params.prediction_path)
    logger.info(f"prediction {predict_pipeline_params.class_params.type_save} "
                f"was saved in {predict_pipeline_params.prediction_path}")


@click.command(name="predict_pipeline")
@click.argument("config_path")
def train_pipeline_command(config_path: str):
    params = read_predict_pipeline_params(config_path)
    predict_pipeline(params)


if __name__ == "__main__":
    train_pipeline_command()
