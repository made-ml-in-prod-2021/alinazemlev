import os
import sys
PATH = os.getcwd()
sys.path.insert(0, PATH)
from ml_project.data import read_data, split_train_val_data
from ml_project.enities import SplitParams


def test_load_dataset(input_data_path: str, target_col: str):
    data = read_data(input_data_path)
    assert len(data) > 10 # check that loaded data is correct
    assert target_col in data.keys() # check that target column in data


def test_split_dataset(tmpdir, input_data_path: str):
    val_size = 0.2
    splitting_params = SplitParams(random_state=239, val_size=val_size,)
    data = read_data(input_data_path)
    train, val = split_train_val_data(data, splitting_params)
    assert train.shape[0] > 10  # check that loaded data is correct
    assert val.shape[0] > 10  # check that loaded data is correct
