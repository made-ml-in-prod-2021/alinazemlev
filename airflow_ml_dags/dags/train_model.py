from airflow import DAG
from airflow.operators.docker_operator import DockerOperator
from airflow.operators.dummy_operator import DummyOperator
from airflow.contrib.sensors.file_sensor import FileSensor
from variables import *


with DAG(
        "train_model",
        default_args=default_args,
        schedule_interval="@weekly",
        start_date=DATE_START,
) as dag:
    start = DummyOperator(task_id="start_task")
    wait_sensor = FileSensor(task_id="wait_dataset",
                             filepath="."+DATA_DIR+"/data.csv",
                             poke_interval=10,
                             retries=100)
    cmd_input = (
        f" --input_dir {DATA_DIR}"
        f" --output_dir {MODEL_DIR}"
    )

    train = DockerOperator(
        image="airflow-train",
        command=cmd_input,
        network_mode="bridge",
        task_id="docker-airflow-train",
        do_xcom_push=False,
        volumes=[f"{VOLUME}:/data"]
    )
    end = DummyOperator(task_id="run_this_last")

    start >> wait_sensor >> train >> end