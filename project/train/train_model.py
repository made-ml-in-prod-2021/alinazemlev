import os
import logging
import click
import sys
PATH = os.getcwd()
sys.path.insert(0, PATH)
from project.enities.train_pipeline_params import TrainPipelineParams, read_training_pipeline_params
from project.features.build_features import extract_target, build_pipeline
from project.enities.logger_params import setup_logging
from project.data.make_dataset import read_data, split_train_val_data
from project.models.build_model import init_model

APPLICATION_NAME = "project"
WARN_APPLICATION_NAME = "project_warn"
LOGGING_YAML = PATH + "/configs/logger_conf.yaml"


logger = logging.getLogger(APPLICATION_NAME)
logger_2 = logging.getLogger(WARN_APPLICATION_NAME)
setup_logging(LOGGING_YAML)


def train_pipeline(training_pipeline_params: TrainPipelineParams):
    logger.info(f"start train pipeline with params {training_pipeline_params}")

    data = read_data(training_pipeline_params.input_data_path)
    logger.info(f"data.shape is {data.shape}")

    train_df, val_df = split_train_val_data(
        data, training_pipeline_params.split_params
    )
    logger.info(f"train_df.shape is {train_df.shape}")
    logger.info(f"val_df.shape is {val_df.shape}")

    pipe = build_pipeline(training_pipeline_params, logger_2)
    X_train = pipe.fit_transform(train_df)
    y_train = extract_target(train_df, training_pipeline_params.features_params)

    logger.info(f"train_features.shape is {X_train.shape}")

    model = init_model(training_pipeline_params.models_params,
                       training_pipeline_params.class_params)
    model.fit(X_train, y_train)

    X_val = pipe.transform(val_df)
    y_val = extract_target(val_df, training_pipeline_params.features_params)

    logger.info(f"val_features.shape is {X_val.shape}")

    model.evaluate(X_val, y_val, training_pipeline_params.metric_path, logger)


@click.command(name="train_pipeline")
@click.argument("config_path")
def train_pipeline_command(config_path: str):
    params = read_training_pipeline_params(config_path)
    train_pipeline(params)


if __name__ == "__main__":
    train_pipeline_command()
