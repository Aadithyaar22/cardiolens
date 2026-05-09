# `azure/` — MLOps Backbone

This subdirectory deploys CardioLens to Azure and is **presentable as its own mini-project**.

## What this layer demonstrates independently

> **"Azure as the MLOps backbone — model registry, dynamic inference serving, and production monitoring."**

Three Azure services, each doing one job:

| Service | Role in CardioLens |
|---|---|
| **Azure ML Managed Online Endpoint** | Hosts the calibrated champion as a REST API behind a key-protected `https://` URL with autoscaling. |
| **Azure Blob Storage** | Stores the `joblib` model artifact. The endpoint pulls it dynamically at container startup, so retraining is a blob upload — no redeploy. |
| **Azure Application Insights** | Ingests structured logs from `score.py` (latency, prediction volume, mean probability per request). Powers the production dashboards. |

## Files

| File | Purpose |
|---|---|
| `deploy.py` | Azure ML SDK v2 deployment — registers model, creates endpoint, deploys, routes traffic |
| `score.py` | Endpoint scoring script. Runs `init()` once to pull from Blob, runs `run()` per request and emits structured logs |
| `blob_upload.py` | One-shot helper to push a freshly trained model to Blob |
| `conda.yaml` | Inference container conda environment |

## Step-by-step deploy

```bash
# 1. Authenticate
az login

# 2. Train locally so reports/champion_model.joblib exists
python -m src.train

# 3. Push the artifact to Blob
python azure/blob_upload.py

# 4. Create the endpoint and deploy
python azure/deploy.py
```

Hit the endpoint:

```bash
curl -X POST $SCORING_URI \
  -H "Authorization: Bearer $AZURE_ENDPOINT_KEY" \
  -H "Content-Type: application/json" \
  -d '{"data":[{"age":54,"sex":1,"cp":3,"trestbps":130,"chol":240,"fbs":0,"restecg":0,"thalach":150,"exang":0,"oldpeak":1.0,"slope":2,"ca":0,"thal":3}]}'
```

## Talking points for viva

- "I split the model artifact from the deployment package so retraining is a blob upload, not a redeploy. The endpoint pulls from Blob at `init()`."
- "Every `run()` call emits structured logs — Application Insights ingests them automatically. I can query `customDimensions.latency_ms` in KQL to see latency percentiles."
- "Managed Online Endpoints give me autoscaling, blue/green deployments, and key-based auth out of the box — I'm focused on the ML, not the infra."
- "The student subscription gives free tier on most of this — Blob storage + a `Standard_DS2_v2` instance for the endpoint stays within free credits."
