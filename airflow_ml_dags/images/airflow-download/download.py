import os
import pandas as pd

import click
from sklearn.datasets import make_regression


@click.command("download")
@click.argument("output_dir")
def download(output_dir: str):
    SAMPLES = 300
    FEATURES = 50
    INFORMATIVE = 25
    cols = [f"column_{str(x)}" for x in range(FEATURES)]
    PATH_DATA = "data.csv"
    PATH_TARGET ="target.csv"
    X, y, _ = make_regression(n_samples=SAMPLES, n_features=FEATURES,
                              n_informative=INFORMATIVE,
                              effective_rank=5, coef=True, bias=0.0,
                              noise=0.1)

    os.makedirs(output_dir, exist_ok=True)
    pd.DataFrame(X, columns=cols).to_csv(os.path.join(output_dir, PATH_DATA))
    pd.DataFrame(y, columns=["target"]).to_csv(os.path.join(output_dir, PATH_TARGET))


if __name__ == '__main__':
    download()