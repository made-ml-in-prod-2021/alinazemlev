import pytest
from airflow.models import DagBag


@pytest.fixture
def dag_bag():
    return DagBag(dag_folder="dags/",
                  include_examples=False)


def test_import_dags(dag_bag: DagBag):
    assert dag_bag is not None
    assert dag_bag.import_errors == {}


def test_download_data_loaded(dag_bag: DagBag):
    dag = dag_bag.get_dag(dag_id='download_data')
    assert dag_bag.import_errors == {}
    assert dag is not None
    assert len(dag.tasks) == 3


def test_download_data_structure(dag_bag: DagBag):
    dag = dag_bag.get_dag(dag_id='download_data')
    dag_dict = {
        "run_this_first": ["docker-airflow-download"],
        "docker-airflow-download": ["run_this_last"],
        "run_this_last": []

    }
    assert dag.task_dict.keys() == dag_dict.keys()
    for task_id, downstream_list in dag_dict.items():
        assert dag.has_task(task_id)
        task = dag.get_task(task_id)
        assert task.downstream_task_ids == set(downstream_list)


def test_train_model_loaded(dag_bag: DagBag):
    dag = dag_bag.get_dag(dag_id='train_model')
    assert dag_bag.import_errors == {}
    assert dag is not None
    assert len(dag.tasks) == 4


def test_train_model_structure(dag_bag: DagBag):
    dag = dag_bag.get_dag(dag_id='train_model')
    dag_dict = {
        "start_task": ["wait_dataset"],
        "wait_dataset": ["docker-airflow-train"],
        "docker-airflow-train": ["run_this_last"],
        "run_this_last": [],

    }
    assert dag.task_dict.keys() == dag_dict.keys()
    for task_id, downstream_list in dag_dict.items():
        assert dag.has_task(task_id)
        task = dag.get_task(task_id)
        assert task.downstream_task_ids == set(downstream_list)


def test_predict_model_loaded(dag_bag: DagBag):
    dag = dag_bag.get_dag(dag_id='predict_model')
    assert dag_bag.import_errors == {}
    assert dag is not None
    assert len(dag.tasks) == 5


def test_predict_model_structure(dag_bag: DagBag):
    dag = dag_bag.get_dag(dag_id='predict_model')
    dag_dict = {
        "start_predict": ["wait_dataset_predict", "wait_model_predict"],
        "wait_dataset_predict": ["docker-airflow-predict"],
        "wait_model_predict": ["docker-airflow-predict"],
        "docker-airflow-predict": ["end_predict"],
        "end_predict": [],

    }
    assert dag.task_dict.keys() == dag_dict.keys()
    for task_id, downstream_list in dag_dict.items():
        assert dag.has_task(task_id)
        task = dag.get_task(task_id)
        assert task.downstream_task_ids == set(downstream_list)
