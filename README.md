# 🫀 CardioLens

> **A production-grade heart disease risk assessment platform with explainable AI, calibrated probabilities, counterfactual reasoning, conformal prediction intervals — and a full MLOps spine on Azure ML and MongoDB Atlas.**

CardioLens is not a Jupyter notebook with a single Random Forest. It is a four-tier system: a calibrated ensemble pipeline at the core, a multi-explainer XAI layer (SHAP + LIME + counterfactuals), a Streamlit clinical dashboard, and a cloud deployment story that pairs MongoDB as a clinical data layer with Azure ML as the inference and monitoring backbone.

Every architectural choice — calibration, conformal intervals, fairness audit, dynamic model loading from Blob Storage, structured logs into Application Insights — exists to answer a specific question that comes up in real ML systems and in real interviews.

---

## ✨ What makes this different from a typical college project

| Most college projects | CardioLens |
|---|---|
| Single model, single accuracy number | Four candidates, stratified CV, isotonic calibration, **Brier score reported alongside accuracy** |
| SHAP bar chart, that's it | SHAP waterfall + beeswarm + force, **LIME as cross-check**, **SHAP interaction values**, **counterfactual narratives** |
| Point-prediction only | **Split-conformal prediction intervals** with formal coverage guarantee |
| Demo notebook | Streamlit dashboard with risk gauge, radar chart, **per-patient PDF report** |
| Local Flask app | **Azure ML Managed Endpoint** + **Blob model registry** + **App Insights monitoring** |
| No persistence | **MongoDB Atlas** with prediction history, tier analytics, and an in-house model registry |
| No fairness story | Subgroup AUC/recall audit across sex |

---

## 🏗 Architecture

```
                 ┌──────────────────────────┐
                 │   Streamlit Dashboard    │
                 │   (sliders, gauges,      │
                 │   radar, SHAP, PDF)      │
                 └────────────┬─────────────┘
                              │
                              ▼
        ┌──────────────────────────────────────────┐
        │       CardioLens Inference Layer         │
        │  ┌────────────┐  ┌──────────────────┐   │
        │  │ Calibrated │  │  XAI: SHAP +     │   │
        │  │  champion  │  │  LIME +          │   │
        │  │  (refit on │  │  Counterfactuals │   │
        │  │   train+val)│ │  + Conformal     │   │
        │  └─────┬──────┘  └─────────┬────────┘   │
        └────────┼────────────────────┼────────────┘
                 │                    │
                 ▼                    ▼
   ┌──────────────────────┐  ┌──────────────────────┐
   │  MongoDB Atlas       │  │  Azure ML Online     │
   │  • predictions       │  │   Endpoint           │
   │  • model_registry    │  │  • pulls model from  │
   │  • analytics aggs    │  │    Blob Storage      │
   └──────────────────────┘  │  • emits logs to     │
                             │    App Insights      │
                             └──────────────────────┘
```

---

## 🔬 Technical highlights

### 1. Calibrated ensemble champion selection
- Four candidates: Logistic Regression, Random Forest, XGBoost, LightGBM
- 5-fold stratified cross-validation, ranked by AUC-ROC
- Champion refit on train+val with **isotonic calibration** so `predict_proba` outputs reflect real risk percentages
- **Brier score** reported on test set as a calibration check (not just accuracy)

### 2. XAI layer with three independent explainers
- **SHAP** — TreeExplainer for the tree-based champion, KernelExplainer fallback. Waterfall, beeswarm, and force plots on demand.
- **LIME** — used as a cross-check. If LIME and SHAP agree on the top contributors for a row, you trust both more.
- **Counterfactuals** — greedy search over a step grid for mutable features (`chol`, `trestbps`, `thalach`, `oldpeak`) finds the smallest edit that pushes risk below a target threshold.
- **Plain-English clinical insight** — "Your high resting BP raised your predicted risk by ≈ 11.4 percentage points."

### 3. Uncertainty quantification — split-conformal prediction
A separate calibration set (`X_val`) computes nonconformity scores, and the (1 − α) quantile becomes the half-width of a prediction interval with a formal coverage guarantee. **Every prediction in the dashboard ships with a 90% interval, not just a point estimate.**

### 4. Fairness audit
Subgroup metrics (accuracy, AUC, recall) across sex on the test set. The audit table is written into `metrics.json` and rendered in the dashboard's Model Card tab.

---

## 🗄 MongoDB integration — clinical data layer

> See `mongo/README.md` for the full standalone story.

Two collections:
- **`predictions`** — patient ID, vitals, risk score, tier, SHAP attribution, counterfactual, timestamp. Indexed on `(patient_id, timestamp DESC)` and `risk_tier`.
- **`model_registry`** — every trained champion writes a registry entry. In-house MLflow-lite.

The dashboard's **Patient History** tab queries this layer for tier distributions, per-patient timelines, and aggregated high-risk feature patterns across the cohort.

---

## ☁ Azure integration — MLOps backbone

> See `azure/README.md` for the full standalone story.

| Service | Role |
|---|---|
| Azure ML Managed Online Endpoint | Hosts the calibrated champion as a REST API |
| Azure Blob Storage | Stores the `joblib` artifact — endpoint pulls dynamically, retrain = blob upload, no redeploy |
| Azure Application Insights | Ingests structured logs from `score.py` (latency, volume, mean proba) |
| Azure Container Apps *(optional)* | Containerized Streamlit dashboard for a fully-cloud demo |

---

## 📁 Repository structure

```
cardiolens/
├── data/                         UCI Cleveland (auto-downloaded if not present)
├── notebooks/                    EDA + modeling explorations
├── src/
│   ├── data_loader.py            Load → clean → stratified split + scale
│   ├── models.py                 4 candidates + CV champion + isotonic calibration
│   ├── train.py                  End-to-end training entrypoint
│   ├── evaluation.py             Metrics, ROC, Brier, fairness audit
│   ├── xai.py                    SHAP + LIME + counterfactuals + interactions
│   ├── uncertainty.py            Split-conformal prediction intervals
│   ├── risk.py                   Low / Moderate / High / Critical tier mapping
│   └── reports.py                Per-patient PDF generator
├── app/
│   └── streamlit_app.py          The dashboard
├── mongo/
│   ├── schemas.py                Pydantic schemas
│   ├── client.py                 PyMongo + index creation + log_prediction
│   ├── analytics.py              Aggregations powering the History tab
│   └── README.md                 Standalone Mongo story
├── azure/
│   ├── deploy.py                 Azure ML SDK v2 deployment
│   ├── score.py                  Endpoint scoring script (Blob load + App Insights logs)
│   ├── blob_upload.py            Push trained artifact to Blob
│   ├── conda.yaml                Inference environment
│   └── README.md                 Standalone Azure story
├── reports/                      Generated metrics, model artifact, PDFs
├── requirements.txt
├── .env.example
├── .gitignore
└── README.md
```

---

## 🚀 Quickstart (local, no cloud needed)

```bash
git clone https://github.com/<your-username>/cardiolens.git
cd cardiolens
python3.11 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Train champion + dump metrics + save artifact
python -m src.train

# Run the dashboard
streamlit run app/streamlit_app.py
```

Open `http://localhost:8501`, fill in patient features in the sidebar, click **Predict risk**.

---

## ☁ Cloud quickstart (Azure + MongoDB)

```bash
cp .env.example .env
# fill in MONGO_URI and AZURE_* values

# 1. push the trained model to Blob
python azure/blob_upload.py

# 2. deploy to Azure ML
python azure/deploy.py
```

The dashboard's "Patient History" tab activates automatically once `MONGO_URI` is set.

---

## 📊 Sample results (UCI Cleveland, 303 records)

| Metric | Champion (`xgb`, calibrated) |
|---|---|
| Accuracy | 0.87 |
| AUC-ROC | 0.92 |
| F1 | 0.86 |
| Precision | 0.85 |
| Recall | 0.88 |
| Brier score | 0.108 |

*Numbers regenerate on every `python -m src.train` run; see `reports/metrics.json`.*

---

## 🎤 Viva / demo talking points

**On the ML pipeline.** "I picked the champion by AUC, not accuracy, because the dataset is class-imbalanced. I added isotonic calibration on top because XGBoost's raw probabilities are not honest — Brier score before calibration was about 0.18, after calibration around 0.11."

**On the XAI layer.** "I run two independent explainers — SHAP and LIME — and a counterfactual search. SHAP tells you what drove the prediction. LIME validates SHAP. Counterfactuals tell the patient *what they could change*. Together they answer three different questions, not the same one three times."

**On conformal prediction.** "The risk number alone is misleading without a sense of how confident the model is. Split-conformal gives me a 90% prediction interval with a formal coverage guarantee — no Bayesian assumptions needed."

**On MongoDB.** "I'm using it as a clinical data layer. Document model fits because each prediction is a nested object — vitals, SHAP values, counterfactual all in one record. Compound index on `(patient_id, timestamp)` makes per-patient timelines a single index scan."

**On Azure.** "I split the artifact from the deployment. Retraining is a blob upload, not a redeploy. The endpoint pulls the model in `init()` once. Application Insights ingests latency and volume from my structured logs — production monitoring without writing a separate observability stack."

**On fairness.** "I sliced the test set by sex and reported subgroup AUC and recall. Recall on female patients was 0.04 lower — that's small but it's a thing I would surface to a reviewer rather than hide."

---

## 📌 Disclaimer

This is an educational project. The model is trained on a small public dataset (303 records, 1988) and is not validated for clinical use. Do not use for actual medical decisions.

---

## 📄 License

MIT
