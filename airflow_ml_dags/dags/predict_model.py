from airflow import DAG
from airflow.operators.docker_operator import DockerOperator
from airflow.operators.dummy_operator import DummyOperator
from airflow.contrib.sensors.file_sensor import FileSensor
from variables import *

with DAG(
        "predict_model",
        default_args=default_args,
        schedule_interval="@daily",
        start_date=DATE_START,
) as dag:
    start = DummyOperator(task_id="start_predict")
    wait_sensor_data = FileSensor(task_id="wait_dataset_predict",
                             filepath="."+DATA_DIR+"/data.csv",
                             poke_interval=10,
                             retries=100)
    wait_sensor_model = FileSensor(task_id="wait_model_predict",
                                  filepath=f"."+FULL_PROD_MODEL_PATH+"/model.pkl",
                                  poke_interval=10,
                                  retries=100)

    cmd_input = (
        f" --input_dir {DATA_DIR}"
        f" --model_dir {FULL_PROD_MODEL_PATH}"
        f" --output_dir {PREDICTION_DIR}"
    )

    predict = DockerOperator(
        image="airflow-predict",
        command=cmd_input,
        network_mode="bridge",
        task_id="docker-airflow-predict",
        do_xcom_push=False,
        volumes=[f"{VOLUME}:/data"]
    )
    end = DummyOperator(task_id="end_predict")

    start >> [wait_sensor_data, wait_sensor_model] >> predict >> end