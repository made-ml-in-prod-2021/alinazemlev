from dataclasses import dataclass

@dataclass()
class ModelsParams:
    scaling_path: str
    imputer_path: str
    categorical_vectorizer_path: str
    classifier_path_postfix: str
    fill_empty: int