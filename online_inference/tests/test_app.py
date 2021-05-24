import os
import sys
PATH = os.getcwd()
sys.path.insert(0, PATH)
from fastapi.testclient import TestClient
from app import app, load_glob_model, ITEMS
from make_request import create_data

client = TestClient(app)


def test_predict():
    load_glob_model()
    data = create_data()
    response = client.get("/predict/", json={"data": [data.iloc[0].tolist()],
                                             "features": list(data.columns)})

    assert response.status_code == 200
    assert list(response.json()[0].keys()) == ['id', 'label']


def test_predict_bad_features():
    load_glob_model()
    response = client.get("/predict/", json={"data": [list(range(ITEMS))],
                                             "features": [str(x) for x in range(ITEMS)]})

    assert response.status_code == 400
