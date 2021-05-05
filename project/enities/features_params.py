from dataclasses import dataclass
from typing import List, Optional


@dataclass()
class FeaturesParams:
    categorical_cols: List[str]
    numerical_cols: List[str]
    binary_cols: List[str]
    target_col: str