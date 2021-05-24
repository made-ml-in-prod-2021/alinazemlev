import logging
import os
import yaml
from logging import config

APPLICATION_NAME = "ml_project"
WARN_APPLICATION_NAME = "ml_project_warn"
LOGGING_YAML = "/configs/logger_conf.yaml"


def build_logger(PATH: str) -> list:
    logger = logging.getLogger(APPLICATION_NAME)
    logger_2 = logging.getLogger(WARN_APPLICATION_NAME)
    setup_logging(PATH+LOGGING_YAML)
    return [logger, logger_2]


def setup_logging(file_yaml):
    """
    :param file_yaml:
    :return:
    """
    assert os.path.exists(file_yaml)
    with open(file_yaml) as con_fin:
        config.dictConfig(yaml.safe_load(con_fin))
