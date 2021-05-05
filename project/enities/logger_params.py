import os
import yaml
from logging import config


def setup_logging(file_yaml):
    """
    :param file_yaml:
    :return:
    """
    assert os.path.exists(file_yaml)
    with open(file_yaml) as con_fin:
        config.dictConfig(yaml.safe_load(con_fin))
