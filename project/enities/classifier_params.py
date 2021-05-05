from dataclasses import dataclass


@dataclass()
class ClassifierParams:
    type: str
    loss: str
    penalty: str
    alpha: float
    max_iter: int
    type_save: str
