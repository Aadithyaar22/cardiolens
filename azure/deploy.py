"""
deploy.py
---------
Deploys the trained CardioLens model to an Azure ML Managed Online Endpoint
using the Azure ML SDK v2.

Prerequisites (one-time):
    pip install azure-ai-ml azure-identity
    az login
    Set in .env:
        AZURE_SUBSCRIPTION_ID, AZURE_RESOURCE_GROUP, AZURE_WORKSPACE_NAME
        AZURE_BLOB_CONNECTION_STRING (for model artifact storage)

Run:
    python azure/deploy.py
"""

from __future__ import annotations

import os
from pathlib import Path

from azure.ai.ml import MLClient
from azure.ai.ml.entities import (
    CodeConfiguration,
    Environment,
    ManagedOnlineDeployment,
    ManagedOnlineEndpoint,
    Model,
)
from azure.identity import DefaultAzureCredential


REPO_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = REPO_ROOT / "reports" / "champion_model.joblib"

ENDPOINT_NAME = os.getenv("AZURE_ENDPOINT_NAME", "cardiolens-endpoint")
DEPLOYMENT_NAME = "blue"


def get_ml_client() -> MLClient:
    return MLClient(
        credential=DefaultAzureCredential(),
        subscription_id=os.environ["AZURE_SUBSCRIPTION_ID"],
        resource_group_name=os.environ["AZURE_RESOURCE_GROUP"],
        workspace_name=os.environ["AZURE_WORKSPACE_NAME"],
    )


def main() -> None:
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model not found at {MODEL_PATH}. Run `python -m src.train` first."
        )

    client = get_ml_client()

    print(f">> Registering model from {MODEL_PATH}")
    registered_model = client.models.create_or_update(
        Model(
            path=str(MODEL_PATH),
            name="cardiolens",
            description="Heart disease risk model — calibrated champion",
            type="custom_model",
        )
    )

    print(">> Building inference environment")
    env = Environment(
        name="cardiolens-env",
        conda_file=str(Path(__file__).parent / "conda.yaml"),
        image="mcr.microsoft.com/azureml/openmpi4.1.0-ubuntu22.04:latest",
    )

    print(f">> Creating endpoint `{ENDPOINT_NAME}`")
    endpoint = ManagedOnlineEndpoint(
        name=ENDPOINT_NAME,
        description="CardioLens heart disease risk prediction endpoint",
        auth_mode="key",
    )
    client.online_endpoints.begin_create_or_update(endpoint).result()

    print(f">> Deploying `{DEPLOYMENT_NAME}` to endpoint")
    deployment = ManagedOnlineDeployment(
        name=DEPLOYMENT_NAME,
        endpoint_name=ENDPOINT_NAME,
        model=registered_model,
        environment=env,
        code_configuration=CodeConfiguration(
            code=str(Path(__file__).parent), scoring_script="score.py"
        ),
        instance_type="Standard_DS2_v2",
        instance_count=1,
    )
    client.online_deployments.begin_create_or_update(deployment).result()

    # Route 100% of traffic to the new deployment
    endpoint.traffic = {DEPLOYMENT_NAME: 100}
    client.online_endpoints.begin_create_or_update(endpoint).result()

    keys = client.online_endpoints.get_keys(ENDPOINT_NAME)
    print(f">> Endpoint live: {client.online_endpoints.get(ENDPOINT_NAME).scoring_uri}")
    print(f">> Primary key (set as AZURE_ENDPOINT_KEY in .env): {keys.primary_key}")


if __name__ == "__main__":
    main()
