import logging
import os
import sys
import joblib
import uvicorn
from typing import List, Union, Optional
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, conlist
PATH = os.getcwd()
sys.path.insert(0, PATH)
import pipeline
from pipeline import BuilderPipe

logger = logging.getLogger(__name__)
NAME_MODEL = "model.pkl"
ITEMS = 13


def load_object(path: str) -> BuilderPipe:
    with open(path, "rb") as f:
        return joblib.load(f)


class DataModel(BaseModel):
    data: List[conlist(Union[int, float, str, None],
                       min_items=ITEMS,
                       max_items=ITEMS)]
    features: List[str]


class DiseasePrediction(BaseModel):
    id: str
    label: int


model: Optional[BuilderPipe] = None


def make_predict(
        data: List, features: List[str], model: BuilderPipe,
) -> List[DiseasePrediction]:
    data = pd.DataFrame(data, columns=features)
    ids = data.index
    predicts = model.pipe.predict(data)

    return [
        DiseasePrediction(id=id_, label=float(label)) for id_, label in zip(ids, predicts)
    ]


app = FastAPI()


@app.get("/")
def main():
    return "it is entry point of our predictor"


@app.on_event("startup")
def load_model():
    global model
    model_path = os.getenv("PATH_TO_MODEL")
    if model_path is None:
        err = f"PATH_TO_MODEL {model_path} is None"
        logger.error(err)
        raise RuntimeError(err)

    model = load_object(model_path)


def load_glob_model():
    global model
    model_path = PATH + "/" + NAME_MODEL
    model = load_object(model_path)


@app.get("/predict/", response_model=List[DiseasePrediction])
def predict(request: DataModel):
    try:
        return make_predict(request.data, request.features, model)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=os.getenv("PORT", 8000))
