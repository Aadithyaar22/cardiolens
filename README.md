# 🫀 CardioLens

[![Live Demo](https://img.shields.io/badge/Live%20Demo-cardiolens--heart.streamlit.app-red?logo=streamlit&logoColor=white)](https://cardiolens-heart.streamlit.app)
[![GitHub](https://img.shields.io/badge/GitHub-Aadithyaar22%2Fcardiolens-black?logo=github)](https://github.com/Aadithyaar22/cardiolens)
[![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python&logoColor=white)](https://python.org)
[![MongoDB](https://img.shields.io/badge/MongoDB-Atlas-green?logo=mongodb&logoColor=white)](https://mongodb.com/atlas)
[![Streamlit](https://img.shields.io/badge/Streamlit-Deployed-FF4B4B?logo=streamlit&logoColor=white)](https://cardiolens-heart.streamlit.app)
[![License](https://img.shields.io/badge/License-MIT-yellow)](LICENSE)

> **A production-grade heart disease risk assessment platform — not a notebook, a system.**
> Calibrated ML · Deep Clinical XAI · Conformal Prediction · MongoDB Atlas · Live on Streamlit Cloud.

---

## 🌐 Live Demo

**[https://cardiolens-heart.streamlit.app](https://cardiolens-heart.streamlit.app)**

Enter any patient profile → get an instantaneous risk score, full clinical explanation of every contributing factor, a "what-if" counterfactual showing what to change, a 90% prediction interval with mathematical guarantee, and a downloadable PDF clinical report.

---

## 🎯 What CardioLens Does

| Output | Description |
|---|---|
| **Risk Score** | Calibrated probability of heart disease (0–100%) |
| **Risk Tier** | Low / Moderate / High / Critical with clinical guidance |
| **SHAP Explanation** | Per-feature attribution — which factors drove this specific prediction |
| **LIME Cross-check** | Independent second explainer validating SHAP |
| **Deep Clinical Reasoning** | What the value means → how it affects the heart → what it leads to |
| **Combined Verdict** | Synthesised medical conclusion across all top factors |
| **Counterfactual** | "If cholesterol drops from X to Y, risk falls from 74% to 28%" |
| **Prediction Interval** | 90% conformal interval — a mathematical guarantee |
| **PDF Report** | Downloadable clinical-grade patient report |
| **MongoDB Persistence** | Every prediction saved with full metadata for cohort analysis |

---

## ✨ What Makes This Different

| Typical college project | CardioLens |
|---|---|
| Single model, one accuracy number | 4 models, 5-fold CV, isotonic calibration, Brier score |
| SHAP bar chart | SHAP + LIME + interaction values + counterfactuals + conformal intervals |
| "The model predicted X" | Full clinical reasoning: mechanism → consequence → action |
| Point estimate only | 90% prediction interval with formal coverage guarantee |
| Local notebook | Live deployed at cardiolens-heart.streamlit.app |
| No persistence | MongoDB Atlas: patient history, tier analytics, cohort insights |
| No fairness analysis | Subgroup AUC and recall audit across sex |

---

## 🏗 Architecture

```
┌─────────────────────────────────────────────────────────────┐
│              Streamlit Dashboard (Live)                      │
│  Blood-flow canvas · Gauge · Radar · SHAP · LIME · PDF      │
└──────────────┬──────────────┬──────────────┬────────────────┘
               │              │              │
       ┌───────▼──────┐ ┌─────▼──────┐ ┌────▼────────────┐
       │   ML Core    │ │ XAI Layer  │ │  Uncertainty    │
       │ RF champion  │ │ SHAP+LIME  │ │ Split-conformal │
       │ AUC 0.969    │ │ Clinical   │ │ 90% guarantee   │
       │ Brier 0.077  │ │ reasoning  │ │                 │
       └──────────────┘ └────────────┘ └─────────────────┘
               │
       ┌───────▼──────────────┐     ┌────────────────────────┐
       │   MongoDB Atlas      │     │   Azure ML (code-ready) │
       │ predictions coll.    │     │ Managed Endpoint        │
       │ model_registry coll. │     │ Blob Storage            │
       │ Cohort analytics     │     │ Application Insights    │
       └──────────────────────┘     └────────────────────────┘
```

---

## 📊 Model Performance

**Champion: Random Forest** — selected by 5-fold stratified cross-validation

| Metric | Score | What it means |
|---|---|---|
| **AUC-ROC** | **0.9692** | Ranks sick above healthy 97% of the time |
| **Accuracy** | 0.9016 | 90% of patients correctly classified |
| **Recall** | **0.9286** | Catches 93 out of 100 real cardiac cases |
| **Precision** | 0.8667 | 87% of flagged patients truly have disease |
| **F1 Score** | 0.8966 | Strong balance of precision and recall |
| **Brier Score** | **0.0766** | Excellent calibration — probabilities are honest |

**All 4 candidates (CV AUC):**

| Model | CV AUC | Description |
|---|---|---|
| 🏆 **Random Forest** | **0.8792 ± 0.0376** | Champion — 400 trees, majority vote |
| Logistic Regression | 0.8682 ± 0.0532 | Linear baseline, highly competitive |
| XGBoost | 0.8508 ± 0.0425 | Sequential error-correcting trees |
| LightGBM | 0.8481 ± 0.0418 | Leaf-wise boosting |

**Fairness audit:**

| Subgroup | n | Accuracy | AUC | Recall |
|---|---|---|---|---|
| Female | 20 | 0.950 | 1.000 | 0.857 |
| Male | 41 | 0.878 | 0.950 | 0.952 |

---

## 🔬 Explainability — The Core Innovation

### SHAP (Primary Explainer)
TreeExplainer computes **exact Shapley values** — mathematically guaranteed feature attribution. Not an approximation. Every prediction decomposes as:

```
final_risk = baseline + SHAP(age) + SHAP(chol) + SHAP(oldpeak) + ... + SHAP(thal)
```

### LIME (Independent Cross-check)
Generates 5,000 perturbed patient variants, scores all of them, fits a local linear model — completely independent from SHAP. **Agreement between SHAP and LIME = high-confidence explanation.**

### Deep Clinical Reasoning Engine
The key differentiator. For each top feature, three layers of medical explanation are generated based on the actual raw value:

**Example — ST Depression = 2.3mm:**
- **What it means:** *"Marked ST segment depression — strong objective evidence of myocardial ischaemia."*
- **How it affects the heart:** *"During ischaemia, the subendocardial myocardium depolarises abnormally, shifting the ECG's ST segment downward. Depression ≥2mm is a Class I indication for cardiac investigation."*
- **What it leads to:** *"Predicts multi-vessel coronary disease. Associated with 5-10x increased cardiac event risk vs a negative stress test."*

**Example — Reversible Thalassemia Defect:**
- **What it means:** *"Thallium scan shows reduced blood flow under stress that recovers at rest — the direct imaging definition of myocardial ischaemia."*
- **How it affects the heart:** *"A critically narrowed coronary artery cannot increase flow during stress. Thallium creates a cold spot in underperfused zones. At rest, flow recovers."*
- **What it leads to:** *"Class I indication for coronary angiography. Tissue at immediate risk of infarction if the causative stenosis is untreated."*

### Combined Clinical Conclusion
All top factors synthesised into one conclusive medical verdict:
> *"The dominant contributors are ST depression and thalassemia status — each independently associated with significant coronary artery disease. Together, this warrants urgent cardiovascular evaluation. Coronary angiography should be strongly considered."*

### Counterfactual Explanations
Greedy search over modifiable features (cholesterol, BP, heart rate, ST depression) finds the minimum real-world intervention:
> *"If your cholesterol drops from 2.03 to 0.53 (scaled units), predicted risk falls from 30.7% to 14.4%."*

### Conformal Prediction Intervals
Split-conformal prediction provides a **90% coverage guarantee** — a mathematical theorem, not an estimate. Uses 46 validation patients as calibration. No Bayesian assumptions required.

---

## 🗄 MongoDB — Clinical Data Layer

**`predictions` collection** — one document per inference:
```json
{
  "patient_id": "P-0001",
  "timestamp": "2025-05-08T14:32:11Z",
  "input_vitals": { "age": 54, "chol": 240, "oldpeak": 2.3 },
  "risk_score": 0.73,
  "risk_tier": "High",
  "shap_values": { "ca": 0.18, "thal": 0.15, "oldpeak": 0.14 },
  "counterfactual": "If cholesterol drops from 2.03 to 0.53...",
  "interval": { "lower": 0.55, "upper": 0.91 }
}
```

**`model_registry` collection** — in-house MLflow-lite tracking every trained model.

**5 aggregation pipelines:**
- Recent predictions timeline
- Risk tier distribution chart
- Top risk drivers across High/Critical cohort (`$objectToArray` on SHAP sub-document)
- Weekly volume and mean risk trend
- Per-patient history search

---

## 📁 Project Structure

```
cardiolens/
├── src/
│   ├── data_loader.py    # Load → clean → stratified split → StandardScaler
│   ├── models.py         # 4 candidates + CV champion + isotonic calibration
│   ├── train.py          # End-to-end training entrypoint
│   ├── evaluation.py     # Metrics + ROC + Brier + fairness audit
│   ├── xai.py            # SHAP + LIME + deep clinical reasoning engine
│   │                     # + counterfactuals + combined conclusion
│   ├── uncertainty.py    # Split-conformal prediction
│   ├── risk.py           # 4-tier risk stratification
│   └── reports.py        # ReportLab PDF generator
├── app/
│   └── streamlit_app.py  # 5-tab dashboard with live MongoDB + animations
├── mongo/
│   ├── schemas.py        # Pydantic v2 document schemas
│   ├── client.py         # PyMongo client + index creation
│   └── analytics.py      # 5 aggregation pipelines
├── azure/
│   ├── deploy.py         # Azure ML SDK v2 deployment
│   ├── score.py          # Scoring script with App Insights logging
│   └── blob_upload.py    # Push artifact to Blob Storage
├── data/cleveland.csv    # UCI Cleveland Heart Disease (303 patients)
├── reports/              # Model artifact + metrics + PDFs
├── requirements.txt
└── .env.example
```

---

## 🚀 Run Locally

```bash
git clone https://github.com/Aadithyaar22/cardiolens.git
cd cardiolens

# Create environment
conda create -n cardiolens python=3.11 -y
conda activate cardiolens
pip install -r requirements.txt

# Mac M-series only
brew install libomp

# Train the model
python -m src.train

# Launch
streamlit run app/streamlit_app.py
```

---

## 🔧 Environment Variables

Create `.env` in the project root:

```env
# MongoDB Atlas (free M0 tier)
MONGO_URI=mongodb+srv://<user>:<password>@cluster0.xxxxx.mongodb.net/
CARDIOLENS_DB=cardiolens

# Azure ML (optional)
AZURE_SUBSCRIPTION_ID=
AZURE_RESOURCE_GROUP=cardiolens-rg
AZURE_WORKSPACE_NAME=cardiolens-workspace
AZURE_ENDPOINT_URI=
AZURE_ENDPOINT_KEY=
AZURE_BLOB_CONNECTION_STRING=
```

---

## 📦 Tech Stack

| Category | Technologies |
|---|---|
| **ML** | scikit-learn, XGBoost, LightGBM, Random Forest |
| **Explainability** | SHAP (TreeExplainer), LIME, custom clinical reasoning engine |
| **Uncertainty** | Split-conformal prediction (custom) |
| **Dashboard** | Streamlit, Plotly, HTML5 Canvas |
| **Database** | MongoDB Atlas, PyMongo, Pydantic v2 |
| **Cloud** | Azure ML, Azure Blob Storage, Azure Application Insights |
| **Reports** | ReportLab |
| **Deployment** | Streamlit Community Cloud |
| **Language** | Python 3.11 |

---

## 📋 Dataset

**UCI Cleveland Heart Disease Dataset**
- 303 patients · 13 features · Collected 1988, Cleveland Clinic Foundation
- Binary target: 0 = no disease, 1 = disease (any severity)
- [UCI ML Repository](https://archive.ics.uci.edu/ml/datasets/Heart+Disease)

---

## ⚠️ Disclaimer

CardioLens is an educational and research project. It is not validated for clinical use and does not constitute medical advice. All predictions and explanations are for demonstration purposes only. Consult a qualified clinician for any medical decisions.

---

## 📄 License

MIT License — free to use, modify, and distribute with attribution.

---

## 👥 Built By

<table>
  <tr>
    <td align="center" width="50%">
      <br>
      <strong>Aadithya A R</strong><br>
      <sub>B.Tech CSE (AI & ML)</sub><br>
      <sub>Global Academy of Technology, Bengaluru</sub><br>
      <sub>ML Pipeline · XAI Engine · Clinical Reasoning · Dashboard · MongoDB · Deployment</sub><br><br>
      <a href="https://github.com/Aadithyaar22">
        <img src="https://img.shields.io/badge/GitHub-Aadithyaar22-black?logo=github&style=flat-square" />
      </a>
    </td>
    <td align="center" width="50%">
      <br>
      <strong>Yadunandan M Nimbalkar</strong><br>
      <sub>B.Tech CSE (AI & ML)</sub><br>
      <sub>Global Academy of Technology, Bengaluru</sub><br>
      <sub>Co-Builder · Project Architecture · Research · Testing · Validation</sub><br><br>
      <a href="https://github.com/yadu080">
        <img src="https://img.shields.io/badge/GitHub-Yadunandan-black?logo=github&style=flat-square" />
      </a>
    </td>
  </tr>
</table>

<br>

<div align="center">
  <sub>⭐ Star this repo if you found it useful</sub>
</div>
