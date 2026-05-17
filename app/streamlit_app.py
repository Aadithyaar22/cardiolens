"""
CardioLens  ·  app/streamlit_app.py  ·  UI v3 — Final
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Run from repo root:
    streamlit run app/streamlit_app.py
"""

from __future__ import annotations
import sys, json, os
from pathlib import Path
from datetime import datetime, timezone

import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.data_loader import get_data, NUMERIC_FEATURES
from src.risk import stratify
from src.uncertainty import ConformalBinary
from src.xai import (
    deep_clinical_report, combined_conclusion,
    build_lime_explainer, build_shap_explainer,
    clinical_insight, explain_instance,
    generate_counterfactual, lime_explain,
)
from src.reports import build_report

# ── Optional integrations (graceful fallback) ────────────────────────────────
try:
    from mongo.client import log_prediction, get_client
    _mongo_ok = get_client()  # test connection
    MONGO_AVAILABLE = True
except Exception:
    MONGO_AVAILABLE = False

AZURE_URI = os.getenv("AZURE_ENDPOINT_URI", "")
AZURE_KEY = os.getenv("AZURE_ENDPOINT_KEY", "")
AZURE_AVAILABLE = bool(AZURE_URI and AZURE_KEY)

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="CardioLens · Heart Risk AI",
    page_icon="🫀",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── FONTS + GLOBAL CSS ───────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Cormorant+Garamond:ital,wght@0,400;0,600;0,700;1,400&family=DM+Sans:opsz,wght@9..40,300;9..40,400;9..40,500;9..40,600&family=JetBrains+Mono:wght@400;500;600&display=swap');

:root{
  --c:#b01020;  --c2:#7a000f;  --cg:rgba(176,16,32,0.18);
  --bg:#070407; --s1:#0f080a;  --s2:#180c0f;  --s3:#220e12;
  --bd:rgba(176,16,32,0.20);   --bd2:rgba(176,16,32,0.40);
  --t:#f2eaea;  --tm:#9a8080;  --ts:#5c4040;
  --gr:#15803d; --am:#b45309;  --or:#c2410c;
  --fd:'Cormorant Garamond',serif;
  --fb:'DM Sans',sans-serif;
  --fm:'JetBrains Mono',monospace;
  --r:14px; --r2:20px; --r3:28px;
}

/* ── reset ── */
html,body,[data-testid="stApp"],[data-testid="stAppViewContainer"]{
  background:var(--bg)!important; color:var(--t)!important;
  font-family:var(--fb)!important;
}
*{ box-sizing:border-box; }

/* ── sidebar ── */
[data-testid="stSidebar"]{
  background:linear-gradient(170deg,#0f0609 0%,#0a0407 100%)!important;
  border-right:1px solid var(--bd)!important;
}
[data-testid="stSidebar"] *{ color:var(--t)!important; }
[data-testid="stSidebar"] [data-testid="stSlider"] [role="slider"]{
  background:var(--c)!important;
}
[data-testid="stSidebar"] [data-baseweb="select"] *{
  background:#1a0d10!important;
}

/* ── tabs ── */
[data-baseweb="tab-list"]{
  background:transparent!important;
  border-bottom:1px solid var(--bd)!important;
  gap:2px;
}
[data-baseweb="tab"]{
  background:transparent!important; color:var(--ts)!important;
  border:none!important; font-family:var(--fb)!important;
  font-weight:500!important; font-size:13px!important;
  padding:10px 20px!important; border-radius:8px 8px 0 0!important;
  transition:all .2s!important; letter-spacing:.02em!important;
}
[aria-selected="true"]{
  color:var(--c)!important;
  border-bottom:2px solid var(--c)!important;
  background:rgba(176,16,32,.06)!important;
}

/* ── buttons ── */
.stButton>button{
  background:linear-gradient(135deg,var(--c),var(--c2))!important;
  color:#fff!important; border:none!important;
  border-radius:10px!important; font-family:var(--fb)!important;
  font-weight:600!important; font-size:14px!important;
  padding:.65rem 1.5rem!important; letter-spacing:.03em;
  box-shadow:0 4px 24px rgba(176,16,32,.35);
  transition:all .22s!important;
}
.stButton>button:hover{
  transform:translateY(-2px)!important;
  box-shadow:0 8px 36px rgba(176,16,32,.6)!important;
}
[data-testid="stDownloadButton"]>button{
  background:transparent!important;
  border:1px solid var(--c)!important;
  color:var(--c)!important; border-radius:10px!important;
}

/* ── dataframes ── */
[data-testid="stDataFrame"]{ border-radius:12px!important; overflow:hidden; }
[data-testid="stDataFrame"] th{
  background:#1a0d10!important; color:var(--t)!important;
  font-family:var(--fm)!important; font-size:11px!important;
}
[data-testid="stDataFrame"] td{ color:var(--t)!important; font-size:13px!important; }

/* ── card ── */
.card{
  background:var(--s2); border:1px solid var(--bd);
  border-radius:var(--r2); padding:24px; margin-bottom:16px;
  transition:border-color .3s,box-shadow .3s;
  animation:slideUp .45s cubic-bezier(.22,1,.36,1) both;
}
.card:hover{ border-color:var(--bd2); box-shadow:0 12px 48px rgba(176,16,32,.12); }

/* ── title ── */
.hero-title{
  font-family:var(--fd)!important; font-size:3.2rem!important;
  font-weight:700!important; line-height:1.0!important;
  background:linear-gradient(135deg,#f2e0e0 0%,var(--c) 45%,#ff8080 100%);
  -webkit-background-clip:text; -webkit-text-fill-color:transparent;
  letter-spacing:-.01em;
}
.hero-sub{
  font-family:var(--fm)!important; font-size:10px!important;
  color:var(--ts)!important; letter-spacing:.18em; text-transform:uppercase;
}

/* ── risk badge ── */
.badge{
  display:inline-block; padding:5px 16px; border-radius:999px;
  font-family:var(--fm); font-size:12px; font-weight:600;
  letter-spacing:.08em; text-transform:uppercase;
  animation:pulse 2.4s ease-in-out infinite;
}
/* ── metric pills ── */
.mrow{ display:flex; gap:10px; flex-wrap:wrap; margin:10px 0; }
.mpill{
  flex:1; min-width:80px;
  background:rgba(176,16,32,.07);
  border:1px solid var(--bd); border-radius:12px;
  padding:10px 12px; text-align:center;
}
.mpill .v{
  font-family:var(--fm); font-size:19px; font-weight:600; color:var(--c);
}
.mpill .l{
  font-size:9px; text-transform:uppercase; letter-spacing:.1em;
  color:var(--ts); margin-top:3px;
}

/* ── insight rows ── */
.irow{
  display:flex; align-items:flex-start; gap:12px;
  padding:11px 16px; margin:6px 0;
  background:rgba(255,255,255,.025);
  border-left:3px solid var(--c);
  border-radius:0 10px 10px 0;
  animation:slideUp .4s ease both;
}
.itext{ font-size:14px; line-height:1.55; }

/* ── cf box ── */
.cfbox{
  background:linear-gradient(135deg,rgba(21,128,61,.08),rgba(21,128,61,.03));
  border:1px solid rgba(21,128,61,.22); border-radius:14px;
  padding:20px; margin-top:10px;
}
.cfbox h5{
  color:#4ade80; font-family:var(--fm); font-size:10px;
  text-transform:uppercase; letter-spacing:.12em; margin-bottom:8px;
}

/* ── status dot ── */
.sdot{
  display:inline-block; width:8px; height:8px;
  border-radius:50%; margin-right:6px;
  animation:pulse 2s ease-in-out infinite;
}

/* ── scrollbar ── */
::-webkit-scrollbar{ width:5px; height:5px; }
::-webkit-scrollbar-track{ background:var(--bg); }
::-webkit-scrollbar-thumb{ background:var(--bd2); border-radius:99px; }

/* ── keyframes ── */
@keyframes slideUp{
  from{ opacity:0; transform:translateY(16px); }
  to{   opacity:1; transform:translateY(0); }
}
@keyframes pulse{
  0%,100%{ box-shadow:0 0 0 0 rgba(176,16,32,.5); }
  50%{     box-shadow:0 0 0 8px rgba(176,16,32,0); }
}
@keyframes hb{
  0%,100%{ transform:scale(1); }
  14%{ transform:scale(1.18); }
  28%{ transform:scale(1); }
  42%{ transform:scale(1.09); }
}
.hb{ animation:hb 1.4s ease-in-out infinite; display:inline-block; }
@keyframes fadeIn{
  from{ opacity:0; } to{ opacity:1; }
}

/* ── iframe canvas ── */
iframe[title="streamlit_components_v1_html.html"]{ display:none!important; }

/* ── misc ── */
.stPlotlyChart{ border-radius:var(--r)!important; overflow:hidden; }
[data-testid="stExpander"]{
  background:var(--s2)!important; border:1px solid var(--bd)!important;
  border-radius:var(--r)!important;
}
[data-testid="stExpander"] summary{ color:var(--t)!important; }
</style>
""", unsafe_allow_html=True)

# ── Blood-flow canvas ────────────────────────────────────────────────────────
components.html("""
<canvas id="bc" style="position:fixed;top:0;left:0;width:100vw;height:100vh;
  pointer-events:none;z-index:0;opacity:0.14;"></canvas>
<script>
(function(){
  const cv=document.getElementById('bc'),ctx=cv.getContext('2d');
  function resize(){cv.width=innerWidth;cv.height=innerHeight;}
  resize(); window.addEventListener('resize',resize);
  const S=[0.12,0.48,0.84], cells=[];
  for(let i=0;i<65;i++){
    const s=S[i%3];
    cells.push({
      x:s*innerWidth+(Math.random()-.5)*130,
      y:Math.random()*innerHeight,
      r:3.5+Math.random()*8,
      vx:(Math.random()-.5)*.45,
      vy:.25+Math.random()*.65,
      w:Math.random()*Math.PI*2,
      ws:.008+Math.random()*.018,
      wa:.5+Math.random()*1.1,
      op:.3+Math.random()*.55,
      t:Math.random()<.68?0:1,
      s:s
    });
  }
  function rbc(x,y,r,op){
    ctx.save(); ctx.globalAlpha=op;
    const g=ctx.createRadialGradient(x-r*.25,y-r*.25,r*.06,x,y,r);
    g.addColorStop(0,'#dd2035'); g.addColorStop(.55,'#b01020'); g.addColorStop(1,'#6a000c');
    ctx.fillStyle=g;
    ctx.beginPath(); ctx.ellipse(x,y,r,r*.58,0,0,Math.PI*2); ctx.fill();
    ctx.globalAlpha=op*.28; ctx.fillStyle='#5a000a';
    ctx.beginPath(); ctx.ellipse(x,y,r*.34,r*.18,0,0,Math.PI*2); ctx.fill();
    ctx.restore();
  }
  function plasma(x,y,r,op){
    ctx.save(); ctx.globalAlpha=op*.45;
    const g=ctx.createRadialGradient(x-r*.3,y-r*.3,0,x,y,r);
    g.addColorStop(0,'#ffe0c8'); g.addColorStop(.5,'#cc7055'); g.addColorStop(1,'transparent');
    ctx.fillStyle=g;
    ctx.beginPath(); ctx.arc(x,y,r,0,Math.PI*2); ctx.fill();
    ctx.restore();
  }
  function vessels(t){
    ctx.save(); ctx.globalAlpha=.04; ctx.strokeStyle='#b01020';
    ctx.lineWidth=32; ctx.lineCap='round';
    S.forEach((s,i)=>{
      ctx.beginPath();
      const x=s*innerWidth;
      ctx.moveTo(x,0);
      ctx.bezierCurveTo(
        x+Math.sin(t*.00025+i)*22, innerHeight*.33,
        x-Math.sin(t*.0003+i+1)*18, innerHeight*.66,
        x+Math.sin(t*.00018+i+2)*15, innerHeight
      );
      ctx.stroke();
    });
    ctx.restore();
  }
  let t=0;
  (function draw(){
    requestAnimationFrame(draw); t++;
    ctx.clearRect(0,0,cv.width,cv.height);
    vessels(t);
    cells.forEach(c=>{
      c.w+=c.ws;
      c.x+=c.vx+Math.sin(c.w)*c.wa*.09;
      c.y+=c.vy;
      const tx=c.s*innerWidth;
      c.vx+=(tx-c.x)*.00025; c.vx*=.975;
      if(c.y-c.r>cv.height){ c.y=-c.r; c.x=tx+(Math.random()-.5)*110; }
      c.t===0?rbc(c.x,c.y,c.r,c.op):plasma(c.x,c.y,c.r*1.5,c.op);
    });
  })();
})();
</script>
""", height=0)

# ── Plotly base layout ───────────────────────────────────────────────────────
PL = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(15,8,10,0.5)",
    font=dict(family="DM Sans,sans-serif", color="#f2eaea"),
    margin=dict(t=44,b=24,l=20,r=20),
)

# ── Input help text ──────────────────────────────────────────────────────────
HELP = {
    "age":      "Age in years. Risk rises sharply after 45 in men, 55 in women.",
    "sex":      "Biological sex. Men statistically have higher early-onset cardiac risk.",
    "cp":       "Chest pain type:\n• Typical angina — classic squeezing pressure on exertion\n• Atypical angina — non-classic chest discomfort\n• Non-anginal — chest pain unrelated to heart\n• Asymptomatic — NO pain (paradoxically highest risk here)",
    "trestbps": "Resting blood pressure (mm Hg). Normal <120. Stage 2 hypertension ≥140.",
    "chol":     "Total cholesterol (mg/dL). Desirable <200. High ≥240.",
    "fbs":      "Fasting blood sugar >120 mg/dL = possible diabetes = major cardiac risk factor.",
    "restecg":  "Resting ECG:\n• Normal\n• ST-T abnormality — possible ischemia\n• LV hypertrophy — thickened heart muscle from chronic high BP",
    "thalach":  "Max heart rate during stress test. Low for your age = poor cardiovascular fitness.",
    "exang":    "Exercise triggered chest pain? Strong predictor of coronary artery disease.",
    "oldpeak":  "ST depression during exercise vs rest. ≥2.0 = clinically significant ischemia.",
    "slope":    "Shape of ECG at peak exercise:\n• Upsloping — typically benign\n• Flat — borderline\n• Downsloping — most abnormal, strongest predictor",
    "ca":       "Major coronary arteries with >50% blockage (fluoroscopy). More = worse.",
    "thal":     "Thallium stress test:\n• Normal — healthy blood flow\n• Fixed defect — permanent scar (old heart attack)\n• Reversible defect — ischemia under stress, recovers at rest",
}

FNAMES = {
    "age":"age","sex":"sex","cp":"chest pain type","trestbps":"resting BP",
    "chol":"cholesterol","fbs":"fasting blood sugar","restecg":"resting ECG",
    "thalach":"max heart rate","exang":"exercise angina","oldpeak":"ST depression",
    "slope":"ST slope","ca":"vessels blocked","thal":"thalassemia",
}

# ── Cached resources ─────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    # Try Azure Blob Storage first, fall back to local
    conn = os.getenv("AZURE_BLOB_CONNECTION_STRING")
    if conn:
        try:
            from azure.storage.blob import BlobServiceClient
            from io import BytesIO
            service = BlobServiceClient.from_connection_string(conn)
            container = os.getenv("MODEL_BLOB_CONTAINER", "cardiolens-models")
            blob_name = os.getenv("MODEL_BLOB_NAME", "champion_model.joblib")
            blob = service.get_blob_client(container=container, blob=blob_name)
            buf = BytesIO()
            blob.download_blob().readinto(buf)
            buf.seek(0)
            return joblib.load(buf)
        except Exception as e:
            st.toast(f"Azure Blob failed, using local model: {e}", icon="⚠️")
    path = REPO_ROOT/"reports"/"champion_model.joblib"
    if not path.exists():
        st.error("Run `python -m src.train` first.")
        st.stop()
    return joblib.load(path)

@st.cache_resource
def load_explainers():
    data  = get_data()
    b     = load_model()
    base  = b["model"].calibrated_classifiers_[0].estimator
    shap_ = build_shap_explainer(base, data.X_train)
    lime_ = build_lime_explainer(data.X_train)
    conf_ = ConformalBinary(alpha=0.1).calibrate(b["model"], data.X_val, data.y_val)
    return data, b["model"], shap_, lime_, conf_

# ── Sidebar ──────────────────────────────────────────────────────────────────
def sidebar_inputs(scaler):
    st.sidebar.markdown("""
    <div style='font-family:"Cormorant Garamond",serif;font-size:1.4rem;
    color:#f2e0e0;margin-bottom:2px;font-weight:600;'>Patient Profile</div>
    <div style='font-size:11px;color:#5c4040;margin-bottom:18px;line-height:1.6;'>
    Hover <b style="color:#9a8080">ℹ</b> on each field for clinical context
    </div>""", unsafe_allow_html=True)

    age   = st.sidebar.slider("Age", 25, 80, 54, help=HELP["age"])
    sex   = st.sidebar.selectbox("Sex", ["Female","Male"], 1, help=HELP["sex"])
    cp    = st.sidebar.selectbox("Chest pain type",
              ["Typical angina","Atypical angina","Non-anginal","Asymptomatic"],
              3, help=HELP["cp"])
    trbps = st.sidebar.slider("Resting BP (mm Hg)", 90, 200, 130, help=HELP["trestbps"])
    chol  = st.sidebar.slider("Cholesterol (mg/dL)", 120, 560, 240, help=HELP["chol"])
    fbs   = st.sidebar.checkbox("Fasting blood sugar >120 mg/dL", help=HELP["fbs"])
    recg  = st.sidebar.selectbox("Resting ECG",
              ["Normal","ST-T abnormality","LV hypertrophy"], help=HELP["restecg"])
    thal_ = st.sidebar.slider("Max heart rate", 70, 210, 150, help=HELP["thalach"])
    exang = st.sidebar.checkbox("Exercise-induced angina", help=HELP["exang"])
    op    = st.sidebar.slider("ST depression (oldpeak)", 0.0, 6.5, 1.0, .1, help=HELP["oldpeak"])
    slp   = st.sidebar.selectbox("ST slope",
              ["Upsloping","Flat","Downsloping"], help=HELP["slope"])
    ca    = st.sidebar.selectbox("Major vessels blocked (0–3)", [0,1,2,3], help=HELP["ca"])
    thal  = st.sidebar.selectbox("Thalassemia",
              ["Normal","Fixed defect","Reversible defect"], help=HELP["thal"])

    raw = {
        "age":age, "sex":1 if sex=="Male" else 0,
        "cp":["Typical angina","Atypical angina","Non-anginal","Asymptomatic"].index(cp)+1,
        "trestbps":trbps,"chol":chol,"fbs":int(fbs),
        "restecg":["Normal","ST-T abnormality","LV hypertrophy"].index(recg),
        "thalach":thal_,"exang":int(exang),"oldpeak":op,
        "slope":["Upsloping","Flat","Downsloping"].index(slp)+1,
        "ca":ca,
        "thal":[3,6,7][["Normal","Fixed defect","Reversible defect"].index(thal)],
    }
    df = pd.DataFrame([raw])
    df[NUMERIC_FEATURES] = scaler.transform(df[NUMERIC_FEATURES])
    return df, raw

# ── Charts ───────────────────────────────────────────────────────────────────
def make_gauge(prob, color):
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=prob*100,
        delta={"reference":50,"increasing":{"color":"#dc2626"},"decreasing":{"color":"#15803d"}},
        number={"suffix":"%","font":{"size":58,"family":"Cormorant Garamond"}},
        gauge={
            "axis":{"range":[0,100],"tickfont":{"size":10},"tickcolor":"#5c4040"},
            "bar":{"color":color,"thickness":.24},
            "bgcolor":"rgba(0,0,0,0)","borderwidth":0,
            "steps":[
                {"range":[0,20],"color":"rgba(21,128,61,.1)"},
                {"range":[20,50],"color":"rgba(180,83,9,.1)"},
                {"range":[50,75],"color":"rgba(194,65,12,.1)"},
                {"range":[75,100],"color":"rgba(176,16,32,.14)"},
            ],
            "threshold":{"line":{"color":color,"width":3},"thickness":.78,"value":prob*100},
        },
    ))
    fig.update_layout(**PL, height=260)
    return fig

def make_radar(raw):
    B={"trestbps":120,"chol":200,"thalach":165,"oldpeak":.1,"age":45}
    labs=["Resting BP","Cholesterol","Max HR","ST Depression","Age"]
    fs=list(B.keys())
    np_=[max(raw[f]/max(B[f],.01),.05) for f in fs]
    fig=go.Figure()
    fig.add_trace(go.Scatterpolar(r=np_,theta=labs,fill="toself",name="Patient",
        line=dict(color="#b01020",width=2),fillcolor="rgba(176,16,32,.15)"))
    fig.add_trace(go.Scatterpolar(r=[1]*5,theta=labs,fill="toself",name="Healthy",
        line=dict(color="#15803d",width=1.5,dash="dot"),fillcolor="rgba(21,128,61,.06)"))
    fig.update_layout(**PL,height=280,
        polar=dict(bgcolor="rgba(0,0,0,0)",
            radialaxis=dict(visible=True,range=[0,2.3],tickfont=dict(size=8),color="#5c4040"),
            angularaxis=dict(tickfont=dict(size=10))),
        legend=dict(bgcolor="rgba(0,0,0,0)",font=dict(size=10),x=0.8,y=1.15))
    return fig

def make_shap_bar(names, vals):
    order=np.argsort(np.abs(vals))[::-1][:9]
    f=[names[i] for i in order][::-1]
    v=[vals[i] for i in order][::-1]
    c=["rgba(176,16,32,.88)" if x>0 else "rgba(21,128,61,.88)" for x in v]
    fig=go.Figure(go.Bar(x=v,y=f,orientation="h",
        marker=dict(color=c,line=dict(width=0)),
        hovertemplate="<b>%{y}</b><br>SHAP: %{x:.4f}<extra></extra>"))
    fig.add_vline(x=0,line_color="rgba(255,255,255,.15)",line_width=1)
    fig.update_layout(**PL,height=340,
        title=dict(text="Feature contributions — SHAP",
                   font=dict(size=14,family="Cormorant Garamond")),
        xaxis_title="← Lowers risk  |  Raises risk →",
        xaxis=dict(zeroline=False,gridcolor="rgba(255,255,255,.04)"),
        yaxis=dict(gridcolor="rgba(255,255,255,.03)"))
    return fig

def make_lime_bar(rows):
    ru=[r[0][:40] for r in rows]; wt=[r[1] for r in rows]
    c=["rgba(176,16,32,.82)" if w>0 else "rgba(21,128,61,.82)" for w in wt]
    fig=go.Figure(go.Bar(x=wt,y=ru,orientation="h",
        marker=dict(color=c,line=dict(width=0)),
        hovertemplate="<b>%{y}</b><br>Weight: %{x:.4f}<extra></extra>"))
    fig.add_vline(x=0,line_color="rgba(255,255,255,.15)",line_width=1)
    fig.update_layout(**PL,height=280,
        title=dict(text="LIME cross-check",
                   font=dict(size=14,family="Cormorant Garamond")),
        xaxis_title="← Lowers risk  |  Raises risk →",
        xaxis=dict(zeroline=False,gridcolor="rgba(255,255,255,.04)"))
    return fig

def make_roc(fpr,tpr):
    fig=go.Figure()
    fig.add_trace(go.Scatter(x=fpr,y=tpr,mode="lines",name="ROC",
        line=dict(color="#b01020",width=2.5),
        fill="tozeroy",fillcolor="rgba(176,16,32,.1)"))
    fig.add_trace(go.Scatter(x=[0,1],y=[0,1],mode="lines",name="Random",
        line=dict(color="#5c4040",width=1,dash="dot")))
    fig.update_layout(**PL,height=260,
        title=dict(text="ROC Curve",font=dict(size=14,family="Cormorant Garamond")),
        xaxis_title="False Positive Rate",yaxis_title="True Positive Rate",
        xaxis=dict(gridcolor="rgba(255,255,255,.04)"),
        yaxis=dict(gridcolor="rgba(255,255,255,.04)"))
    return fig

# ── Header ───────────────────────────────────────────────────────────────────
st.markdown("""
<div style='padding:22px 0 10px;'>
  <div class='hero-title'><span class='hb'>🫀</span> CardioLens</div>
  <div class='hero-sub' style='margin-top:6px;'>
    Heart disease risk · Explainable AI · UCI Cleveland · v0.1
  </div>
</div>""", unsafe_allow_html=True)

# ── Status bar ───────────────────────────────────────────────────────────────
mongo_color = "#15803d" if MONGO_AVAILABLE else "#5c4040"
mongo_label = "MongoDB connected" if MONGO_AVAILABLE else "MongoDB offline"
blob_conn = os.getenv("AZURE_BLOB_CONNECTION_STRING", "")
azure_color = "#15803d" if blob_conn else "#5c4040"
azure_label = "Azure Blob Storage connected" if blob_conn else "Azure offline"
AZURE_AVAILABLE = bool(blob_conn)
st.markdown(f"""
<div style='display:flex;gap:20px;margin-bottom:6px;align-items:center;'>
  <span style='font-family:"JetBrains Mono",monospace;font-size:11px;color:#5c4040;'>SYSTEM STATUS</span>
  <span style='font-size:12px;color:{mongo_color};'>
    <span class='sdot' style='background:{mongo_color};'></span>{mongo_label}
  </span>
  <span style='font-size:12px;color:{azure_color};'>
    <span class='sdot' style='background:{azure_color};'></span>{azure_label}
  </span>
</div>""", unsafe_allow_html=True)

# ── Load ─────────────────────────────────────────────────────────────────────
bundle = load_model()
data, model, shap_exp, lime_exp, conformal = load_explainers()
x_scaled, raw = sidebar_inputs(bundle["scaler"])
patient_id = st.sidebar.text_input("Patient ID", "P-0001")
st.sidebar.markdown("<hr style='border-color:rgba(176,16,32,.18);margin:12px 0;'>",
                    unsafe_allow_html=True)
predict_btn = st.sidebar.button("🫀  Analyse Risk", type="primary", use_container_width=True)
st.sidebar.markdown("""<div style='font-size:10px;color:#3a2020;margin-top:8px;
text-align:center;line-height:1.7;'>Educational use only.<br>
Not a substitute for clinical diagnosis.</div>""", unsafe_allow_html=True)

tab1,tab2,tab3,tab4,tab5 = st.tabs([
    "📊 Risk Overview","🔬 Why this result?","💡 What can change?",
    "🗄 Patient History","📋 Model Card"
])

# ── Compute ───────────────────────────────────────────────────────────────────
if predict_btn:
    with st.spinner("Analysing cardiovascular markers…"):
        proba  = float(model.predict_proba(x_scaled)[0,1])
        tier   = stratify(proba)
        ivl    = conformal.interval(model, x_scaled)
        exp    = explain_instance(shap_exp, x_scaled)
        insg   = clinical_insight(exp)
        cf     = generate_counterfactual(model, x_scaled)
        lime_r = lime_explain(lime_exp, model, x_scaled)
        deep   = deep_clinical_report(exp, raw, top_k=4)
        concl  = combined_conclusion(deep, proba)

    st.session_state.R = dict(proba=proba,tier=tier,ivl=ivl,exp=exp,
                               insg=insg,cf=cf,lime_r=lime_r,raw=raw,
                               pid=patient_id,ts=datetime.now(timezone.utc),
                               deep=deep,concl=concl)

    # ── Log to MongoDB ───────────────────────────────────────────────────────
    if MONGO_AVAILABLE:
        try:
            log_prediction(
                patient_id=patient_id, raw_features=raw,
                risk_proba=proba, risk_tier=tier.label,
                shap_values=dict(zip(exp.feature_names, exp.values.tolist())),
                counterfactual=cf.narrative,
                interval={"lower": ivl.lower, "upper": ivl.upper},
            )
            st.toast("✅ Prediction saved to MongoDB", icon="🗄")
        except Exception as e:
            st.toast(f"MongoDB write failed: {e}", icon="⚠️")

    # ── Score via Azure endpoint (if available) ──────────────────────────────
    if AZURE_AVAILABLE:
        try:
            import requests
            resp = requests.post(
                AZURE_URI,
                headers={"Authorization":f"Bearer {AZURE_KEY}",
                         "Content-Type":"application/json"},
                json={"data":[raw]},
                timeout=8,
            )
            if resp.ok:
                az_result = resp.json()
                st.session_state.azure_proba = az_result.get("probabilities",[proba])[0]
                st.toast("✅ Azure endpoint responded", icon="☁️")
        except Exception as e:
            st.toast(f"Azure call failed: {e}", icon="⚠️")

# ── Empty state ───────────────────────────────────────────────────────────────
if "R" not in st.session_state:
    with tab1:
        st.markdown("""
        <div class='card' style='text-align:center;padding:56px 32px;'>
          <div style='font-size:72px;margin-bottom:16px;'>🫀</div>
          <div style='font-family:"Cormorant Garamond",serif;font-size:1.9rem;
          color:#f2e0e0;margin-bottom:12px;font-weight:600;'>
            Ready to assess cardiovascular risk
          </div>
          <div style='font-size:14px;color:#9a8080;line-height:1.85;
          max-width:440px;margin:0 auto;'>
            Enter patient measurements in the sidebar.<br>
            Every field has an <b style="color:#f2eaea;">ℹ tooltip</b> with full
            clinical context — normal ranges, what the measurement means, and
            why it predicts cardiac risk.
          </div>
        </div>""", unsafe_allow_html=True)
    st.stop()

R = st.session_state.R

# ════════════════════════════════════════════════════════════════════════════
# TAB 1 — Risk Overview
# ════════════════════════════════════════════════════════════════════════════
with tab1:
    c1,c2 = st.columns([1,1],gap="large")

    with c1:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown(f"""
        <div style='display:flex;align-items:center;
        justify-content:space-between;margin-bottom:6px;'>
          <div style='font-family:"Cormorant Garamond",serif;
          font-size:1.1rem;color:#f2e0e0;font-weight:600;'>
            Predicted Risk Score</div>
          <span class='badge'
            style='background:rgba(176,16,32,.12);
                   color:{R["tier"].color};
                   border:1px solid {R["tier"].color}40;'>
            {R["tier"].label}
          </span>
        </div>""", unsafe_allow_html=True)
        st.plotly_chart(make_gauge(R["proba"],R["tier"].color),
                        use_container_width=True, key="g1")

        lo,hi,pt = R["ivl"].lower*100, R["ivl"].upper*100, R["proba"]*100
        st.markdown(f"""
        <div style='padding:8px 4px;'>
          <div style='font-size:10px;color:#5c4040;text-transform:uppercase;
          letter-spacing:.12em;margin-bottom:7px;'>90% Prediction Interval</div>
          <div style='position:relative;height:7px;
          background:rgba(255,255,255,.05);border-radius:99px;'>
            <div style='position:absolute;left:{lo:.1f}%;
            width:{hi-lo:.1f}%;height:100%;
            background:linear-gradient(90deg,
              rgba(176,16,32,.25),rgba(176,16,32,.55));
            border-radius:99px;'></div>
            <div style='position:absolute;left:{pt:.1f}%;top:-4px;
            width:15px;height:15px;margin-left:-7px;
            background:{R["tier"].color};border-radius:50%;
            border:2px solid #070407;
            box-shadow:0 0 10px {R["tier"].color};'></div>
          </div>
          <div style='display:flex;justify-content:space-between;
          margin-top:6px;font-size:11px;color:#5c4040;
          font-family:"JetBrains Mono",monospace;'>
            <span>{lo:.1f}%</span>
            <span>Point: <b style='color:{R["tier"].color}'>{pt:.1f}%</b></span>
            <span>{hi:.1f}%</span>
          </div>
          <div style='font-size:11px;color:#3a2020;margin-top:5px;
          line-height:1.5;'>
            90% of future intervals contain the true risk (split-conformal guarantee)
          </div>
        </div>
        <div style='margin-top:14px;padding:14px 16px;
        background:rgba(176,16,32,.06);border-radius:12px;
        border-left:3px solid {R["tier"].color};'>
          <div style='font-size:10px;text-transform:uppercase;
          letter-spacing:.1em;color:{R["tier"].color};margin-bottom:4px;'>
            Clinical guidance</div>
          <div style='font-size:13px;line-height:1.6;color:#ccc;'>
            {R["tier"].advice}</div>
        </div>""", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

        # Azure score comparison if available
        if "azure_proba" in st.session_state:
            az = st.session_state.azure_proba
            st.markdown(f"""
            <div class='card' style='margin-top:0;'>
              <div style='font-size:10px;color:#5c4040;text-transform:uppercase;
              letter-spacing:.12em;margin-bottom:6px;'>☁️ Azure Endpoint Score</div>
              <div style='font-family:"JetBrains Mono",monospace;font-size:22px;
              color:#60a5fa;'>{az*100:.1f}%</div>
              <div style='font-size:11px;color:#5c4040;margin-top:4px;'>
              Scored via Azure ML Managed Online Endpoint</div>
            </div>""", unsafe_allow_html=True)

    with c2:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("""
        <div style='font-family:"Cormorant Garamond",serif;font-size:1.1rem;
        color:#f2e0e0;font-weight:600;margin-bottom:4px;'>
        Patient vs Healthy Baseline</div>
        <div style='font-size:11px;color:#5c4040;margin-bottom:6px;'>
        <span style='color:#b01020;'>■</span> Red = patient &nbsp;
        <span style='color:#15803d;'>■</span> Green dots = healthy reference.
        Values beyond green ring = elevated.
        </div>""", unsafe_allow_html=True)
        st.plotly_chart(make_radar(R["raw"]),use_container_width=True,key="r1")
        st.markdown("</div>", unsafe_allow_html=True)

    # ── Key vitals ────────────────────────────────────────────────────────────
    m=R["raw"]
    def flag(v,ok,warn): return "🔴" if v>=warn else("🟡" if v>=ok else "🟢")
    st.markdown(f"""
    <div class='mrow'>
      <div class='mpill'>
        <div class='v'>{flag(m["trestbps"],130,140)} {m["trestbps"]}</div>
        <div class='l'>Resting BP</div></div>
      <div class='mpill'>
        <div class='v'>{flag(m["chol"],200,240)} {m["chol"]}</div>
        <div class='l'>Cholesterol</div></div>
      <div class='mpill'>
        <div class='v'>{"🟡" if m["thalach"]<120 else "🟢"} {m["thalach"]}</div>
        <div class='l'>Max HR</div></div>
      <div class='mpill'>
        <div class='v'>{flag(m["oldpeak"],1.0,2.0)} {m["oldpeak"]:.1f}</div>
        <div class='l'>ST Depression</div></div>
      <div class='mpill'>
        <div class='v'>{"🔴" if m["exang"] else "🟢"} {"Yes" if m["exang"] else "No"}</div>
        <div class='l'>Exer. Angina</div></div>
      <div class='mpill'>
        <div class='v'>{flag(m["ca"],1,2)} {m["ca"]}</div>
        <div class='l'>Vessels Blocked</div></div>
    </div>""", unsafe_allow_html=True)

    # ── Insights ──────────────────────────────────────────────────────────────
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("""<div style='font-family:"Cormorant Garamond",serif;
    font-size:1.1rem;color:#f2e0e0;font-weight:600;margin-bottom:12px;'>
    What's driving this result?</div>""", unsafe_allow_html=True)
    icons=["🫀","🩸","📈","⚡"]
    for i,line in enumerate(R["insg"]):
        st.markdown(f"""
        <div class='irow' style='animation-delay:{i*.07}s;'>
          <div style='font-size:18px;flex-shrink:0;'>{icons[i%4]}</div>
          <div class='itext'>{line}</div>
        </div>""", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════════════
# TAB 2 — Why this result?
# ════════════════════════════════════════════════════════════════════════════
with tab2:
    st.markdown("""
    <div class='card' style='padding:18px 24px;'>
      <div style='font-family:"Cormorant Garamond",serif;font-size:1.2rem;
      color:#f2e0e0;font-weight:600;margin-bottom:6px;'>How to read these charts</div>
      <div style='font-size:13px;color:#9a8080;line-height:1.8;'>
        <b style='color:#b01020;'>Red bars</b> = raised your risk above average. &nbsp;
        <b style='color:#15803d;'>Green bars</b> = lowered your risk below average.<br>
        Bar length = strength of that feature's influence on <i>this</i> patient.<br>
        <b>SHAP + LIME use completely different maths</b> — agreement between them
        means the explanation is reliable.
      </div>
    </div>""", unsafe_allow_html=True)

    c1,c2 = st.columns([3,2],gap="large")
    with c1:
        st.markdown("<div class='card'>",unsafe_allow_html=True)
        st.plotly_chart(make_shap_bar(R["exp"].feature_names, R["exp"].values),
                        use_container_width=True, key="sb")

        # ── Deep clinical explanation cards ─────────────────────────────────
        st.markdown("""<div style='font-size:12px;color:#9a8080;
        margin-top:4px;margin-bottom:8px;'>
        <b style='color:#f2eaea;'>Top factors — full clinical reasoning:</b></div>""",
                    unsafe_allow_html=True)
        for a in R["deep"]:
            col = "#b01020" if a["direction"]=="raises" else "#15803d"
            sev_bg  = {"high":"rgba(176,16,32,.14)","moderate":"rgba(180,83,9,.1)",
                       "low":"rgba(21,128,61,.08)","neutral":"rgba(255,255,255,.03)"}
            sev_bdr = {"high":"#b01020","moderate":"#b45309",
                       "low":"#15803d","neutral":"#5c4040"}
            bg  = sev_bg.get(a["severity"],"rgba(255,255,255,.03)")
            bdr = sev_bdr.get(a["severity"],"#5c4040")
            shr = f"{a['share']*100:.0f}%"
            st.markdown(f"""
            <div style='background:{bg};border:1px solid {bdr}40;
            border-left:4px solid {bdr};border-radius:12px;
            padding:16px 18px;margin-top:10px;'>
              <div style='display:flex;justify-content:space-between;
              align-items:flex-start;margin-bottom:8px;'>
                <div style='font-size:13px;font-weight:600;color:{col};'>
                  &#9658; {a["what_it_is"]}</div>
                <span style='font-family:"JetBrains Mono",monospace;font-size:10px;
                color:#5c4040;background:rgba(0,0,0,.35);padding:2px 8px;
                border-radius:99px;white-space:nowrap;margin-left:8px;'>
                  {shr} of explanation</span>
              </div>
              <div style='font-size:12px;color:#e0cccc;line-height:1.7;margin-bottom:7px;'>
                <b style='color:#f2eaea;'>What this means: </b>{a["what_it_means"]}</div>
              <div style='font-size:12px;color:#ccbbbb;line-height:1.7;margin-bottom:7px;'>
                <b style='color:#f2eaea;'>How it affects the heart: </b>{a["mechanism"]}</div>
              <div style='font-size:12px;color:#ccbbbb;line-height:1.7;'>
                <b style='color:#f2eaea;'>What this leads to: </b>{a["consequence"]}</div>
            </div>""", unsafe_allow_html=True)
        st.markdown("</div>",unsafe_allow_html=True)

    with c2:
        st.markdown("<div class='card'>",unsafe_allow_html=True)
        st.plotly_chart(make_lime_bar(R["lime_r"]),
                        use_container_width=True, key="lb")
        st.markdown("""
        <div style='font-size:12px;color:#9a8080;margin-top:10px;line-height:1.7;'>
          <b style='color:#f2eaea;'>Why LIME?</b><br>
          LIME builds 5,000 modified patient copies and fits a simple rule model
          on the results. It's model-agnostic and independent from SHAP.
          When both point to the same features, confidence is high.
        </div>""", unsafe_allow_html=True)
        st.markdown("</div>",unsafe_allow_html=True)

    # ── Combined medical conclusion ───────────────────────────────────────────
    st.markdown(f"""
    <div class='card' style='border-left:4px solid #b01020;
    background:linear-gradient(135deg,rgba(176,16,32,.1),rgba(176,16,32,.04));'>
      <div style='font-family:"Cormorant Garamond",serif;font-size:1.15rem;
      font-weight:600;color:#f2e0e0;margin-bottom:10px;'>
        &#9672; Clinical Conclusion</div>
      <div style='font-size:14px;color:#e0cccc;line-height:1.85;'>
        {R["concl"]}</div>
      <div style='margin-top:12px;font-size:11px;color:#5c4040;font-style:italic;'>
        This conclusion is generated by synthesising SHAP attributions with
        evidence-based clinical knowledge for each feature value.
        It is for educational purposes only and does not constitute medical advice.
      </div>
    </div>""", unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════════════
# TAB 3 — What can change?
# ════════════════════════════════════════════════════════════════════════════
with tab3:
    st.markdown("""
    <div class='card' style='padding:18px 24px;'>
      <div style='font-family:"Cormorant Garamond",serif;font-size:1.2rem;
      color:#f2e0e0;font-weight:600;margin-bottom:6px;'>
        Counterfactual Analysis</div>
      <div style='font-size:13px;color:#9a8080;line-height:1.8;'>
        <i>"What is the smallest realistic change to this patient's measurements
        that would push predicted risk below 30%?"</i><br>
        The model searches over modifiable features (cholesterol, BP, heart rate,
        ST depression) and finds the minimum intervention.
      </div>
    </div>""", unsafe_allow_html=True)

    cf=R["cf"]
    rb,ra=R["proba"]*100, cf.target_proba*100
    delta=rb-ra

    if cf.deltas:
        col1,col2=st.columns(2,gap="large")
        with col1:
            fig=go.Figure(go.Indicator(
                mode="number+delta",
                value=ra,
                delta={"reference":rb,"increasing":{"color":"#dc2626"},
                       "decreasing":{"color":"#15803d"}},
                number={"suffix":"%","font":{"size":52,
                        "family":"Cormorant Garamond"},"valueformat":".1f"},
                title={"text":"Risk after changes","font":{"size":12,"color":"#9a8080"}},
            ))
            fig.update_layout(**PL,height=170)
            st.plotly_chart(fig,use_container_width=True,key="cf_ind")

        with col2:
            st.markdown(f"""
            <div style='padding:14px 0;'>
              <div style='font-size:11px;color:#5c4040;text-transform:uppercase;
              letter-spacing:.1em;'>Risk reduction</div>
              <div style='font-family:"JetBrains Mono",monospace;
              font-size:32px;color:#4ade80;margin:4px 0;'>
                −{delta:.1f}%</div>
              <div style='font-size:13px;color:#9a8080;'>
                from {rb:.1f}% → {ra:.1f}%</div>
            </div>""", unsafe_allow_html=True)

        st.markdown(f"""
        <div class='cfbox'>
          <h5>Recommended changes</h5>
          <div style='font-size:15px;line-height:1.9;color:#f2eaea;'>
            {cf.narrative}</div>
        </div>""", unsafe_allow_html=True)

        rows=[{"Feature":k,
               "From (scaled)":f"{v[0]:.3f}",
               "To (scaled)":f"{v[1]:.3f}",
               "Direction":"↓ reduce" if v[1]<v[0] else "↑ increase"}
              for k,v in cf.deltas.items()]
        st.dataframe(pd.DataFrame(rows),use_container_width=True,hide_index=True)
    else:
        st.markdown(f"""
        <div class='card' style='text-align:center;padding:32px;'>
          <div style='font-size:40px;'>✅</div>
          <div style='font-size:14px;color:#9a8080;margin-top:10px;'>
            {cf.narrative}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<hr style='border-color:rgba(176,16,32,.15);margin:18px 0;'>",
                unsafe_allow_html=True)
    if st.button("📄 Generate clinical PDF report"):
        rpath=REPO_ROOT/"reports"/f"{R['pid']}_report.pdf"
        build_report(
            output_path=rpath, patient_id=R["pid"],
            risk_proba=R["proba"], risk_tier=R["tier"].label,
            risk_color=R["tier"].color,
            feature_names=R["exp"].feature_names,
            shap_values=R["exp"].values,
            insights=R["insg"], counterfactual_text=cf.narrative,
            interval_lower=R["ivl"].lower, interval_upper=R["ivl"].upper,
        )
        with open(rpath,"rb") as f:
            st.download_button("⬇ Download PDF",data=f.read(),
                file_name=rpath.name,mime="application/pdf")


# ════════════════════════════════════════════════════════════════════════════
# TAB 4 — Patient History (MongoDB)
# ════════════════════════════════════════════════════════════════════════════
with tab4:
    if not MONGO_AVAILABLE:
        st.markdown("""
        <div class='card' style='text-align:center;padding:40px;'>
          <div style='font-size:40px;margin-bottom:12px;'>🗄</div>
          <div style='font-family:"Cormorant Garamond",serif;font-size:1.3rem;
          color:#f2e0e0;margin-bottom:10px;'>MongoDB not connected</div>
          <div style='font-size:13px;color:#9a8080;line-height:1.8;
          max-width:420px;margin:0 auto;'>
            Add your MongoDB Atlas URI to <code>.env</code> to enable
            patient history, tier analytics, and cohort insights.<br><br>
            <code style='color:#b01020;'>MONGO_URI=mongodb+srv://...</code>
          </div>
        </div>""", unsafe_allow_html=True)
    else:
        try:
            from mongo.analytics import (recent_predictions, tier_counts,
                                          common_high_risk_features)
            col1,col2=st.columns(2,gap="large")

            with col1:
                st.markdown("<div class='card'>",unsafe_allow_html=True)
                st.markdown("""<div style='font-family:"Cormorant Garamond",serif;
                font-size:1.1rem;color:#f2e0e0;font-weight:600;
                margin-bottom:10px;'>Recent Predictions</div>""",
                            unsafe_allow_html=True)
                recent=recent_predictions(limit=15)
                if recent:
                    df_r=pd.DataFrame(recent)
                    if "timestamp" in df_r.columns:
                        df_r["timestamp"]=pd.to_datetime(
                            df_r["timestamp"]).dt.strftime("%m/%d %H:%M")
                    st.dataframe(df_r,use_container_width=True,hide_index=True)
                else:
                    st.info("No predictions logged yet.")
                st.markdown("</div>",unsafe_allow_html=True)

            with col2:
                st.markdown("<div class='card'>",unsafe_allow_html=True)
                st.markdown("""<div style='font-family:"Cormorant Garamond",serif;
                font-size:1.1rem;color:#f2e0e0;font-weight:600;
                margin-bottom:10px;'>Risk Tier Distribution</div>""",
                            unsafe_allow_html=True)
                tiers=tier_counts()
                if tiers:
                    df_t=pd.DataFrame(tiers)
                    tier_colors={"Low":"#15803d","Moderate":"#b45309",
                                 "High":"#c2410c","Critical":"#b01020"}
                    fig=go.Figure(go.Bar(
                        x=df_t["tier"],y=df_t["count"],
                        marker_color=[tier_colors.get(t,"#5c4040")
                                      for t in df_t["tier"]],
                    ))
                    fig.update_layout(**PL,height=220,showlegend=False,
                        xaxis=dict(gridcolor="rgba(255,255,255,.04)"),
                        yaxis=dict(gridcolor="rgba(255,255,255,.04)"))
                    st.plotly_chart(fig,use_container_width=True,key="tier_bar")
                st.markdown("</div>",unsafe_allow_html=True)

            # Common high-risk features
            st.markdown("<div class='card'>",unsafe_allow_html=True)
            st.markdown("""<div style='font-family:"Cormorant Garamond",serif;
            font-size:1.1rem;color:#f2e0e0;font-weight:600;
            margin-bottom:10px;'>Top Risk Drivers — High/Critical Cohort</div>
            <div style='font-size:12px;color:#9a8080;margin-bottom:10px;'>
            Features with the highest mean positive SHAP value across
            all High and Critical risk predictions.</div>""",
                        unsafe_allow_html=True)
            drivers=common_high_risk_features(top_k=6)
            if drivers:
                df_d=pd.DataFrame(drivers)
                fig=go.Figure(go.Bar(
                    x=df_d["mean_contribution"],y=df_d["feature"],
                    orientation="h",
                    marker=dict(color="rgba(176,16,32,.80)",line=dict(width=0)),
                ))
                fig.update_layout(**PL,height=240,
                    xaxis_title="Mean SHAP contribution (High/Critical patients)",
                    xaxis=dict(gridcolor="rgba(255,255,255,.04)"),
                    yaxis=dict(gridcolor="rgba(255,255,255,.03)"))
                st.plotly_chart(fig,use_container_width=True,key="drivers")
            else:
                st.info("Score more patients to see cohort analytics.")
            st.markdown("</div>",unsafe_allow_html=True)

            # Patient search
            with st.expander("🔍 Search patient history"):
                pid_search=st.text_input("Patient ID","P-0001",key="pid_s")
                if st.button("Search",key="search_btn"):
                    from mongo.analytics import search_patient
                    results=search_patient(pid_search)
                    if results:
                        st.dataframe(pd.DataFrame(results),
                                     use_container_width=True,hide_index=True)
                    else:
                        st.info(f"No records found for {pid_search}")
        except Exception as e:
            st.error(f"MongoDB query failed: {e}")


# ════════════════════════════════════════════════════════════════════════════
# TAB 5 — Model Card
# ════════════════════════════════════════════════════════════════════════════
with tab5:
    mp=REPO_ROOT/"reports"/"metrics.json"
    if not mp.exists():
        st.info("Run `python -m src.train` to populate this tab.")
    else:
        metrics=json.loads(mp.read_text())
        st.markdown(f"""
        <div class='card' style='padding:18px 24px;'>
          <div style='font-family:"Cormorant Garamond",serif;font-size:1.3rem;
          color:#f2e0e0;font-weight:600;'>
            Champion: <span style='color:#b01020;'>
            {metrics["champion"].upper()}</span></div>
          <div style='font-size:12px;color:#9a8080;margin-top:4px;'>
            5-fold stratified CV AUC selection ·
            Isotonic calibration · UCI Cleveland 303 patients
          </div>
        </div>""", unsafe_allow_html=True)

        c1,c2=st.columns(2,gap="large")
        with c1:
            st.markdown("<div class='card'>",unsafe_allow_html=True)
            st.markdown("**Test-set performance**")
            tm=metrics["test_metrics"]
            perf=[
                {"Metric":"AUC-ROC","Value":f"{tm['auc_roc']:.4f}",
                 "What it means":"Ranks sick above healthy — 0.5=coin flip, 1.0=perfect"},
                {"Metric":"Accuracy","Value":f"{tm['accuracy']:.4f}",
                 "What it means":"% correctly classified"},
                {"Metric":"F1","Value":f"{tm['f1']:.4f}",
                 "What it means":"Balance of precision & recall"},
                {"Metric":"Precision","Value":f"{tm['precision']:.4f}",
                 "What it means":"Of flagged high-risk, how many truly had disease"},
                {"Metric":"Recall","Value":f"{tm['recall']:.4f}",
                 "What it means":"Of truly sick, how many did we catch (most critical)"},
                {"Metric":"Brier","Value":f"{tm['brier']:.4f}",
                 "What it means":"Calibration: 0=perfect, lower=more honest probabilities"},
            ]
            st.dataframe(pd.DataFrame(perf),use_container_width=True,hide_index=True)

            if "roc" in metrics:
                st.plotly_chart(make_roc(metrics["roc"]["fpr"],metrics["roc"]["tpr"]),
                                use_container_width=True,key="roc_c")
            st.markdown("</div>",unsafe_allow_html=True)

        with c2:
            st.markdown("<div class='card'>",unsafe_allow_html=True)
            st.markdown("**All candidates — CV AUC**")
            cv=pd.DataFrame(metrics["cv_results"])
            fig=go.Figure(go.Bar(
                x=cv["name"],y=cv["auc_mean"],
                error_y=dict(array=cv["auc_std"],visible=True,
                             color="rgba(255,255,255,.25)"),
                marker=dict(color=["#b01020" if n==metrics["champion"]
                                   else "#3a1a1e" for n in cv["name"]]),
            ))
            fig.update_layout(**PL,height=210,showlegend=False,
                yaxis=dict(range=[.5,1.],gridcolor="rgba(255,255,255,.04)"))
            st.plotly_chart(fig,use_container_width=True,key="cv_c")

            if metrics.get("fairness"):
                st.markdown("**Fairness audit — by sex**")
                st.dataframe(pd.DataFrame(metrics["fairness"]),
                             use_container_width=True,hide_index=True)
                st.markdown("""<div style='font-size:11px;color:#5c4040;
                margin-top:6px;line-height:1.6;'>
                A recall gap between groups means the model misses more
                real cases in one demographic — a critical clinical concern
                requiring investigation before deployment.
                </div>""", unsafe_allow_html=True)
            st.markdown("</div>",unsafe_allow_html=True)
