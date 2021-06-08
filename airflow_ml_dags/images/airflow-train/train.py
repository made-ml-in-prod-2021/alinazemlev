import os
import pandas as pd
import pickle
import click
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

@click.command("train")
@click.option("--input_dir")
@click.option("--output_dir")
def train(input_dir: str, output_dir):

    PATH_DATA = "data.csv"
    PATH_TARGET = "target.csv"
    PATH_MODEL = "model.pkl"
    PATH_METRIC = "metric.txt"
    TEST_SIZE = 0.3

    data = pd.read_csv(os.path.join(input_dir, PATH_DATA))
    target = pd.read_csv(os.path.join(input_dir, PATH_TARGET))

    #split train-validation
    X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=TEST_SIZE)

    #preprocess data
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.fit_transform(X_test)

    #fit model
    model = LinearRegression()
    model.fit(X_train, y_train)

    #validate
    mse = mean_squared_error(y_test, model.predict(X_test))
    #save
    os.makedirs(output_dir, exist_ok=True)
    pickle.dump(model, open(os.path.join(output_dir, PATH_MODEL), "wb"))
    with open(os.path.join(output_dir, PATH_METRIC), "w") as fout:
        fout.writelines(str(mse))

if __name__ == "__main__":
    train()
