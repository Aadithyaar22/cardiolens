"""
blob_upload.py
--------------
One-shot helper that pushes the trained champion model to Azure Blob Storage
so the inference endpoint can pull it dynamically at startup.

Run:
    python azure/blob_upload.py
"""

from __future__ import annotations

import os
from pathlib import Path

from azure.storage.blob import BlobServiceClient


REPO_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = REPO_ROOT / "reports" / "champion_model.joblib"


def main() -> None:
    conn = os.environ["AZURE_BLOB_CONNECTION_STRING"]
    container = os.getenv("MODEL_BLOB_CONTAINER", "cardiolens-models")
    blob_name = os.getenv("MODEL_BLOB_NAME", "champion_model.joblib")

    service = BlobServiceClient.from_connection_string(conn)
    if container not in [c.name for c in service.list_containers()]:
        service.create_container(container)

    blob = service.get_blob_client(container=container, blob=blob_name)
    with open(MODEL_PATH, "rb") as f:
        blob.upload_blob(f, overwrite=True)

    print(f">> Uploaded {MODEL_PATH.name} → {container}/{blob_name}")


if __name__ == "__main__":
    main()
