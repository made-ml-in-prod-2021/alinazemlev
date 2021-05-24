import numpy as np
import pandas as pd
import requests


def create_data():
    LENGTH = 300
    COUNT = 3
    binary_cols = ["fbs", "exang", "sex"]
    num_cols = ["chol", "trestbps", "thalach", "age"]
    discrete_cols = ["ca", "cp", "restecg", "thal", "slope"]
    dict_data = {}

    for val in binary_cols:
        dict_data[val] = np.random.binomial(1, np.random.rand(), LENGTH)

    for val in num_cols:
        dict_data[val] = np.random.normal(np.random.randn(), np.random.rand(), LENGTH)

    for val in discrete_cols:
        dict_data[val] = np.random.choice(range(COUNT), size=LENGTH, p=[0.2, 0.7, 0.1])

    dict_data["oldpeak"] = np.random.noncentral_chisquare(1.16, 1.03, LENGTH)
    return pd.DataFrame.from_dict(dict_data)


if __name__ == "__main__":
    data = create_data()
    request_features = list(data.columns)
    for i in range(100):
        request_data = [
            x.item() if isinstance(x, np.generic) else x for x in data.iloc[i].tolist()
        ]
        print(request_data)
        response = requests.get(
            "http://0.0.0.0:80/predict/",
            json={"data": [request_data], "features": request_features},
        )
        print(response.status_code)
        print(response.json()[0].keys())
