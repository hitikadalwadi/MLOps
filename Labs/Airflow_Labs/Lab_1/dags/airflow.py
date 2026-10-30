# Import necessary libraries and modules
from airflow import DAG
from airflow.providers.standard.operators.python import PythonOperator
from datetime import datetime, timedelta
from src.lab import (
    load_data,
    data_preprocessing,
    build_save_model,
    evaluate_model,
    compare_with_previous_run,
)

# All task outputs are base64 strings or plain dicts, so the default
# JSON XCom backend works and XCom pickling does not need to be enabled.

# Define default arguments for your DAG
default_args = {
    'owner': 'hd',
    'start_date': datetime(2026, 10, 29),
    'retries': 0,  # Number of retries in case of task failure
    'retry_delay': timedelta(minutes=5),  # Delay before retries
}

# Create a DAG instance named 'Airflow_Lab1' with the defined default arguments
with DAG(
    'Airflow_Lab1',
    default_args=default_args,
    description='Wine classifier: train, evaluate, and compare against the previous run',
    schedule=None,  # Trigger manually
    catchup=False,
) as dag:

    # Task to load data, calls the 'load_data' Python function
    load_data_task = PythonOperator(
        task_id='load_data_task',
        python_callable=load_data,
    )

    # Task to perform data preprocessing, depends on 'load_data_task'
    data_preprocessing_task = PythonOperator(
        task_id='data_preprocessing_task',
        python_callable=data_preprocessing,
        op_args=[load_data_task.output],
    )

    # Task to build and save a model, depends on 'data_preprocessing_task'
    build_save_model_task = PythonOperator(
        task_id='build_save_model_task',
        python_callable=build_save_model,
        op_args=[data_preprocessing_task.output, "model.sav"],
    )

    # Task to score the saved model on the held-out test set and record the metrics
    evaluate_model_task = PythonOperator(
        task_id='evaluate_model_task',
        python_callable=evaluate_model,
        op_args=["model.sav", build_save_model_task.output],
    )

    # Task to compare this run's metrics with the previous run's
    compare_runs_task = PythonOperator(
        task_id='compare_runs_task',
        python_callable=compare_with_previous_run,
        op_args=[evaluate_model_task.output],
    )

    # Set task dependencies
    (
        load_data_task
        >> data_preprocessing_task
        >> build_save_model_task
        >> evaluate_model_task
        >> compare_runs_task
    )

# If this script is run directly, allow command-line interaction with the DAG
if __name__ == "__main__":
    dag.test()
