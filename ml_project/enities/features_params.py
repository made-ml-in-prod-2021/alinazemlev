from dataclasses import dataclass, field


@dataclass()
class FeaturesParams:
    categorical_cols: field(default_factory=list)
    numerical_cols: field(default_factory=list)
    binary_cols: field(default_factory=list)
    target_col: str