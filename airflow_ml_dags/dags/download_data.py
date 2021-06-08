from airflow import DAG
from airflow.operators.docker_operator import DockerOperator
from airflow.operators.dummy_operator import DummyOperator
from variables import *

with DAG(
        "download_data",
        default_args=default_args,
        schedule_interval="@daily",
        start_date=DATE_START,
) as dag:
    start = DummyOperator(task_id="run_this_first")
    download = DockerOperator(
        image="airflow-download",
        command=DATA_DIR,
        network_mode="bridge",
        task_id="docker-airflow-download",
        do_xcom_push=False,
        volumes=[f"{VOLUME}:/data"]
    )
    end = DummyOperator(task_id="run_this_last")

    start >> download >> end