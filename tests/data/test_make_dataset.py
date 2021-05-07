import os
import sys
PATH = os.getcwd()
sys.path.insert(0, PATH)
from project.data import read_data, split_train_val_data
from project.enities import SplitParams


def test_load_dataset(input_data_path: str, target_col: str):
    data = read_data(input_data_path)
    assert len(data) > 10
    assert target_col in data.keys()


def test_split_dataset(tmpdir, input_data_path: str):
    val_size = 0.2
    splitting_params = SplitParams(random_state=239, val_size=val_size,)
    data = read_data(input_data_path)
    train, val = split_train_val_data(data, splitting_params)
    assert train.shape[0] > 10
    assert val.shape[0] > 10