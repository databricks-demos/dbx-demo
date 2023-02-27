import json
import os
import pathlib
import time
from typing import Dict, Tuple, Any

import numpy as np
import pandas as pd
import requests
from databricks_cli.configure.config import _get_api_client
from databricks_cli.configure.provider import EnvironmentVariableConfigProvider
from databricks_cli.sdk import JobsService, ApiClient
from mlflow.tracking import MlflowClient
from pyspark.sql import SparkSession

from lendingclub_scoring.data.DataProvider import LendingClubDataProvider


def create_spark_session():
    return SparkSession.builder.master("local[1]").getOrCreate()


def prepare_scoring_data(spark) -> pd.DataFrame:
    inpt_path = pathlib.Path(".") / "tests" / "unit" / "data" / "lendingclub.parquet"
    inpt_df = spark.read.format("parquet").load(str(inpt_path.absolute()))
    inpt_df.createOrReplaceTempView("lendingclub")
    data_provider = LendingClubDataProvider(spark, "select * from lendingclub", 100)
    return data_provider.prepare_data().drop(columns=["bad_loan"])


def get_model_version_for_stage(model_name: str, stage: str) -> str:
    mlflow_client = MlflowClient()
    versions = mlflow_client.get_latest_versions(model_name, stages=[stage])
    return str(max([int(v.version) for v in versions]))


def get_api_clent() -> ApiClient:
    config = EnvironmentVariableConfigProvider().get_config()
    api_client = _get_api_client(config, command_name="")
    return api_client


def read_config(file_name: str) -> Dict[str, str]:
    file_content = (pathlib.Path("conf") / file_name).read_text()
    return json.loads(file_content)


def create_serving_endpoint(
    api_client: ApiClient, endpoint_name: str, model_name: str, model_version: str
):
    req = {
        "name": endpoint_name,
        "config": {
            "served_models": [
                {
                    "name": model_name,
                    "model_name": model_name,
                    "model_version": model_version,
                    "workload_size": "Small",
                    "scale_to_zero_enabled": False,
                }
            ],
            "traffic_config": {
                "routes": [{"served_model_name": model_name, "traffic_percentage": 100}]
            },
        },
    }
    return api_client.perform_query("POST", "/serving-endpoints", data=req)


def delete_endpoint(api_client: ApiClient, endpoint_name: str):
    return api_client.perform_query("DELETE", f"/serving-endpoints/{endpoint_name}")


def check_if_endpoint_is_ready(api_client: ApiClient, endpoint_name: str):
    res = api_client.perform_query("GET", f"/serving-endpoints/{endpoint_name}")
    print(res)
    state = res.get("state")
    if state:
        return state.get("ready") == "READY"
    else:
        return False


def wait_for_endpoint_to_become_ready(
    api_client: ApiClient, endpoint_name: str, timeout: int = 360, step: int = 10
):
    waited = 0
    while not check_if_endpoint_is_ready(api_client, endpoint_name):
        waited += step
        if waited >= timeout:
            return False
        time.sleep(step)
    return True


def query_endpoint(endpoint_name: str, df: pd.DataFrame) -> Tuple[Any, int]:
    url = f"{os.environ.get('DATABRICKS_HOST')}serving-endpoints/{endpoint_name}/invocations"
    headers = {
        "Authorization": f'Bearer {os.environ.get("DATABRICKS_TOKEN")}',
        "Content-Type": "application/json",
    }
    ds_dict = {"dataframe_split": df.to_dict(orient="split")}
    data_json = json.dumps(ds_dict, allow_nan=True)
    start_time = time.time_ns()
    response = requests.request(method="POST", headers=headers, url=url, data=data_json)
    end_time = time.time_ns()
    if response.status_code != 200:
        raise Exception(
            f"Request failed with status {response.status_code}, {response.text}"
        )
    return response.json(), (end_time - start_time) / 1_000_000


def test_endpoint(
    endpoint_name: str,
    latency_threshold: int,
    qps_threshold: int,
    test_data_df: pd.DataFrame,
):
    durations = []
    for i in range(300):
        res_json, duration = query_endpoint(endpoint_name, test_data_df)
        durations.append(duration)
        preds = res_json.get("predictions")
        if preds:
            if len(preds) != 10:
                raise Exception("Wrong number of predictions!")
    p95 = np.percentile(durations, 95)
    qps = len(durations) / (sum(durations) / 1000)
    print(f"Observed latency (P90): {p95}. Observed QPS: {qps}.")
    if p95 > latency_threshold:
        raise Exception(
            f"Latency requirements are not met. Observed P99 for the requests is {p95}. At least {latency_threshold} was expected."
        )
    if qps < qps_threshold:
        raise Exception(
            f"QPS requirements are not met. Observed QPS for the requests is {qps}. At least {qps_threshold} was expected."
        )


def get_max_version_for_model(
    api_client: ApiClient, endpoint_name: str, model_name: str
):
    res = api_client.perform_query("GET", f"/serving-endpoints/{endpoint_name}")
    if res.status_code == 404:
        return None
    if res.status_code != 200:
        raise Exception(f"Error requesting endpoint! Response code: {res.status_code}")
    served_models = res.json()["config"]["served_models"]
    versions = []
    for model in served_models:
        if model["model_name"] == model_name:
            versions.append(model["model_version"])
    max_version = str(max([int(x) for x in versions]))
    return max_version


def deploy_new_version_to_existing_endpoint(
    api_client: ApiClient, endpoint_name: str, model_name: str, version_to_deploy: str
):
    update_endpoint_req = {
        "served_models": [
            {
                "name": model_name,
                "model_name": model_name,
                "model_version": version_to_deploy,
                "workload_size": "Small",
                "scale_to_zero_enabled": False,
            }
        ],
        "traffic_config": {
            "routes": [{"served_model_name": model_name, "traffic_percentage": 100}]
        },
    }
    res = api_client.perform_query(
        "PUT", f"/serving-endpoints/{endpoint_name}/config", data=update_endpoint_req
    )
    print(res)


def get_model_endpoint_config(
    api_client: ApiClient, endpoint_name: str
) -> Dict[str, Any]:
    try:
        res = api_client.perform_query("GET", f"/serving-endpoints/{endpoint_name}")
        return res
    except:
        return None


def perform_prod_deployment():
    conf = read_config("lendingclub_config.json")
    model_name = conf["model-name"]
    endpoint_name = conf["endpoint-name"]
    api_client = get_api_clent()
    spark = create_spark_session()
    df = prepare_scoring_data(spark)[:10]
    existing_endpt_conf = get_model_endpoint_config(api_client, endpoint_name)
    model_version = get_model_version_for_stage(model_name, "Production")
    if existing_endpt_conf:
        deploy_new_version_to_existing_endpoint(
            api_client, endpoint_name, model_name, model_version
        )
    else:
        create_serving_endpoint(api_client, endpoint_name, model_name, model_version)
    time.sleep(100)
    test_endpoint(endpoint_name, 1000, 1, df)


def perform_integration_test():
    conf = read_config("lendingclub_config.json")
    model_name = conf["model-name"]
    p95_threshold = conf.get("p95-threshold", 1500)
    qps_threshold = conf.get("qps-threshold", 1)
    endpoint_name = f"{model_name}-integration-test-endpoint"
    model_version = get_model_version_for_stage(model_name, "Staging")
    api_client = get_api_clent()
    spark = create_spark_session()
    test_data_df = prepare_scoring_data(spark)[:10]
    creation_res = create_serving_endpoint(
        api_client, endpoint_name, model_name, model_version
    )
    time.sleep(100)
    if wait_for_endpoint_to_become_ready(api_client, endpoint_name):
        test_endpoint(endpoint_name, p95_threshold, qps_threshold, test_data_df)
        delete_endpoint(api_client, endpoint_name)
    else:
        print("Endpoint failed to become ready within timeout. ")
        raise Exception("Endpoint failed to become ready within timeout. ")


import os

os.environ["MLFLOW_TRACKING_URI"] = "databricks"
perform_integration_test()
perform_prod_deployment()
