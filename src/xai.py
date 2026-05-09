"""
xai.py  —  CardioLens Explainability Layer (v2)
Deep clinical reasoning engine for every feature.
"""
from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np
import pandas as pd
import shap
from lime.lime_tabular import LimeTabularExplainer

FEATURE_ENGLISH = {
    "age":"age","sex":"sex","cp":"chest pain type",
    "trestbps":"resting blood pressure","chol":"serum cholesterol",
    "fbs":"fasting blood sugar","restecg":"resting ECG result",
    "thalach":"max heart rate achieved","exang":"exercise-induced angina",
    "oldpeak":"ST depression on exercise","slope":"ST segment slope",
    "ca":"number of major vessels (fluoroscopy)","thal":"thalassemia status",
}

@dataclass
class ShapExplanation:
    base_value: float
    values: np.ndarray
    feature_names: list[str]
    raw_features: dict[str, float]

@dataclass
class CounterfactualResult:
    original_proba: float
    target_proba: float
    deltas: dict[str, tuple[float, float]] = field(default_factory=dict)
    narrative: str = ""

# ── SHAP ─────────────────────────────────────────────────────────────────────
def build_shap_explainer(model, background: pd.DataFrame, max_background: int = 100):
    bg = background.sample(min(len(background), max_background), random_state=42)
    try:
        return shap.TreeExplainer(model)
    except Exception:
        return shap.KernelExplainer(model.predict_proba, bg)

def explain_instance(explainer, x: pd.DataFrame) -> ShapExplanation:
    shap_values = explainer.shap_values(x)
    if isinstance(shap_values, list):
        values = shap_values[1][0]
        base_value = float(np.atleast_1d(explainer.expected_value)[1])
    else:
        arr = np.array(shap_values)
        values = arr[0] if arr.ndim == 2 else arr[0, :, 1]
        base_value = float(np.atleast_1d(explainer.expected_value)[0])
    return ShapExplanation(
        base_value=base_value, values=np.array(values),
        feature_names=list(x.columns), raw_features=x.iloc[0].to_dict(),
    )

# ── Deep clinical reasoning engine ───────────────────────────────────────────
def _analyse_feature(feature: str, raw_val: float, shap_val: float) -> dict:
    d = {"feature": feature, "direction": "raises" if shap_val > 0 else "lowers",
         "raw_val": raw_val, "label": FEATURE_ENGLISH.get(feature, feature),
         "what_it_is": "", "what_it_means": "", "mechanism": "",
         "consequence": "", "severity": "neutral"}

    if feature == "age":
        age = int(raw_val)
        if age >= 65:
            d.update(dict(what_it_is=f"Age {age}",
                what_it_means="Advanced age is a major non-modifiable cardiac risk factor.",
                mechanism="Arterial walls stiffen (arteriosclerosis), coronary plaque accumulates over decades, and the heart's electrical conduction system degrades. The left ventricle thickens and becomes less efficient at filling and ejecting blood.",
                consequence="Older patients have less physiological reserve — the heart tolerates ischaemia less well. Combined with other risk factors, age amplifies total cardiovascular risk multiplicatively.",
                severity="high"))
        elif age >= 50:
            d.update(dict(what_it_is=f"Age {age}",
                what_it_means="Middle age carries moderately elevated baseline cardiac risk.",
                mechanism="Cumulative atherosclerotic plaque that began forming in the 20s can now be haemodynamically significant. Coronary reserve starts to diminish.",
                consequence="Risk roughly doubles every decade after 45. Any additional factor (high cholesterol, hypertension, smoking) has a compounding effect at this age.",
                severity="moderate"))
        else:
            d.update(dict(what_it_is=f"Age {age}",
                what_it_means="Younger age is protective — arterial walls are elastic and resilient.",
                mechanism="Endothelial function is typically preserved. The heart has greater physiological reserve and tolerance for stress.",
                consequence="Cardiac events at young age are less common and often linked to congenital defects or extreme risk factor burden.",
                severity="low"))

    elif feature == "cp":
        cp_map = {1:"Typical angina",2:"Atypical angina",3:"Non-anginal pain",4:"Asymptomatic"}
        label = cp_map.get(int(raw_val), "Unknown")
        if int(raw_val) == 4:
            d.update(dict(what_it_is=f"Chest pain: {label} (no chest pain)",
                what_it_means="Paradoxically, patients with NO chest pain have the highest disease prevalence — a phenomenon called silent ischaemia.",
                mechanism="Autonomic neuropathy (common in diabetics and elderly) blunts pain perception. The heart undergoes ischaemia — oxygen shortage — but the pain signal never reaches conscious awareness. The damage accumulates silently.",
                consequence="Silent ischaemia is the most dangerous presentation because it goes unnoticed until a major event. These patients often present late, after significant myocardial damage. It is the strongest chest-pain predictor in the Cleveland dataset.",
                severity="high"))
        elif int(raw_val) == 1:
            d.update(dict(what_it_is=f"Chest pain: {label}",
                what_it_means="Classic exertional chest tightness — the textbook presentation of stable coronary artery disease.",
                mechanism="During exertion, myocardial oxygen demand rises. In a narrowed coronary artery, supply cannot keep up. Ischaemia triggers pain via adenosine release stimulating cardiac afferent nerves.",
                consequence="Typical angina confirms clinical suspicion of obstructive CAD and warrants stress testing and likely angiography.",
                severity="moderate"))
        else:
            d.update(dict(what_it_is=f"Chest pain: {label}",
                what_it_means="Non-classic or non-cardiac chest pain pattern — lower pre-test probability of obstructive CAD.",
                mechanism="Pain not following the ischaemic pattern, may be musculoskeletal, gastrointestinal, or psychogenic.",
                consequence="Reduces pre-test probability of significant CAD but cannot be ruled out without investigation.",
                severity="low"))

    elif feature == "trestbps":
        bp = int(raw_val)
        if bp >= 140:
            d.update(dict(what_it_is=f"Resting BP: {bp} mm Hg (Stage 2 Hypertension)",
                what_it_means="Sustained high blood pressure at rest — a primary modifiable cardiac risk factor.",
                mechanism="Chronically elevated pressure damages the arterial endothelium, accelerating atherosclerosis. It forces the left ventricle to pump harder (increased afterload), causing left ventricular hypertrophy. Hypertensive arteries are stiffer and more prone to plaque rupture.",
                consequence="Stage 2 hypertension doubles the risk of major adverse cardiac events. LVH, if it develops, further increases arrhythmia risk and heart failure risk. This is one of the most treatable risk factors — medication can normalise BP within weeks.",
                severity="high"))
        elif bp >= 130:
            d.update(dict(what_it_is=f"Resting BP: {bp} mm Hg (Elevated / Stage 1 Hypertension)",
                what_it_means="Blood pressure above optimal range — early hypertensive changes underway.",
                mechanism="Even modest BP elevation causes slow endothelial injury over years. Inflammatory markers rise, promoting foam cell formation and early coronary plaque deposition.",
                consequence="Stage 1 hypertension increases 10-year ASCVD risk. Lifestyle changes at this stage can prevent progression to Stage 2.",
                severity="moderate"))
        else:
            d.update(dict(what_it_is=f"Resting BP: {bp} mm Hg (Normal)",
                what_it_means="Blood pressure within healthy range — not contributing to vascular damage.",
                mechanism="Normal vascular pressure preserves endothelial integrity and prevents the inflammatory cascade leading to atherosclerosis.",
                consequence="Protective. Normal BP reduces strain on the left ventricle and preserves coronary arterial compliance.",
                severity="low"))

    elif feature == "chol":
        ch = int(raw_val)
        if ch >= 240:
            d.update(dict(what_it_is=f"Cholesterol: {ch} mg/dL (High)",
                what_it_means="Clinically high serum cholesterol — the primary biochemical driver of atherosclerosis.",
                mechanism="Excess LDL cholesterol infiltrates the arterial intima and becomes oxidised. Macrophages engulf oxidised LDL, becoming foam cells that accumulate into atherosclerotic plaques. These plaques narrow coronary arteries (stenosis) and can rupture, triggering acute myocardial infarction.",
                consequence="Each 40 mg/dL rise in LDL increases ASCVD risk by approximately 20%. At 240+ mg/dL, significant coronary plaque burden is highly probable. Statin therapy is typically indicated at this level.",
                severity="high"))
        elif ch >= 200:
            d.update(dict(what_it_is=f"Cholesterol: {ch} mg/dL (Borderline High)",
                what_it_means="Cholesterol above the desirable threshold — slower but meaningful plaque accumulation.",
                mechanism="At borderline levels, LDL still deposits in arterial walls, compounding over decades especially when combined with hypertension or diabetes.",
                consequence="Combined with other risk factors, borderline cholesterol substantially raises total ASCVD risk. Dietary modification is the first-line intervention.",
                severity="moderate"))
        else:
            d.update(dict(what_it_is=f"Cholesterol: {ch} mg/dL (Desirable)",
                what_it_means="Cholesterol within healthy range — low substrate for atherosclerotic plaque formation.",
                mechanism="Lower LDL means less material for plaque formation. HDL can effectively perform reverse cholesterol transport from arterial walls.",
                consequence="Desirable cholesterol is one of the most important modifiable protective cardiovascular factors.",
                severity="low"))

    elif feature == "thalach":
        hr = int(raw_val)
        if hr < 120:
            d.update(dict(what_it_is=f"Max heart rate: {hr} bpm (Severely reduced)",
                what_it_means="A very low maximum heart rate during stress testing — a strong marker of cardiac limitation called chronotropic incompetence.",
                mechanism="Healthy hearts increase HR 2-3x during maximal exercise to meet oxygen demand. Failure to do so indicates sinus node dysfunction, significant ischaemia limiting cardiac output, or medication effects. The heart cannot respond to physiological stress.",
                consequence="Chronotropic incompetence is an independent predictor of all-cause mortality and cardiac events. It represents a fundamental failure of cardiovascular reserve.",
                severity="high"))
        elif hr < 150:
            d.update(dict(what_it_is=f"Max heart rate: {hr} bpm (Below average)",
                what_it_means="Below-average peak heart rate — partial chronotropic limitation.",
                mechanism="May reflect early conduction disease, mild ischaemia limiting exertion tolerance, or reduced cardiovascular fitness. The heart's stress response is blunted.",
                consequence="Associated with modestly increased cardiac risk. May indicate subclinical disease that doesn't cause symptoms at rest but limits the heart's ability to handle demand.",
                severity="moderate"))
        else:
            d.update(dict(what_it_is=f"Max heart rate: {hr} bpm (Good)",
                what_it_means="Healthy maximal heart rate response — good cardiovascular fitness and intact sinus node function.",
                mechanism="Adequate HR response to exercise indicates preserved cardiac output reserve and absence of significant ischaemia limiting exertion.",
                consequence="Higher max HR during stress testing is associated with lower cardiac mortality. The heart can meet physiological demands.",
                severity="low"))

    elif feature == "exang":
        if int(raw_val) == 1:
            d.update(dict(what_it_is="Exercise-induced angina: YES",
                what_it_means="Chest pain triggered by physical exertion — the clinical hallmark of obstructive coronary artery disease.",
                mechanism="During exercise, myocardial oxygen demand increases 3-5x. In a vessel with >70% stenosis, coronary flow reserve is exhausted — supply cannot meet demand. Resulting ischaemia triggers anginal pain via adenosine release stimulating cardiac C-fibres.",
                consequence="Exercise-induced angina confirms flow-limiting coronary disease with high specificity. It is a direct indication for coronary angiography to define anatomy and consider revascularisation (stenting or bypass surgery). Untreated, it predicts future MI.",
                severity="high"))
        else:
            d.update(dict(what_it_is="Exercise-induced angina: NO",
                what_it_means="No chest pain during physical exertion — absence of the hallmark symptom of obstructive CAD.",
                mechanism="Either coronary vessels adequately supply increased demand during exercise, or the patient has silent ischaemia with blunted pain perception.",
                consequence="Absence of exertional angina substantially reduces the pre-test probability of haemodynamically significant coronary stenosis.",
                severity="low"))

    elif feature == "oldpeak":
        op = float(raw_val)
        if op >= 2.0:
            d.update(dict(what_it_is=f"ST depression: {op:.1f} mm (Clinically significant)",
                what_it_means="Marked ST segment depression during exercise vs rest — strong objective evidence of myocardial ischaemia.",
                mechanism="During ischaemia, the subendocardial myocardium (innermost layer, most vulnerable to oxygen shortage) depolarises abnormally. This shifts the ECG's ST segment downward during exercise. Depression ≥2mm is a Class I indication for further cardiac investigation.",
                consequence="Significant ST depression correlates directly with the extent of myocardium at risk. It predicts multi-vessel coronary disease and is associated with 5-10x increased cardiac event risk compared to a negative stress test.",
                severity="high"))
        elif op >= 1.0:
            d.update(dict(what_it_is=f"ST depression: {op:.1f} mm (Borderline abnormal)",
                what_it_means="Mild ST depression during exercise — borderline abnormal, context-dependent significance.",
                mechanism="Mild subendocardial ischaemia during peak exercise. May represent early or single-vessel disease with limited haemodynamic impact at rest.",
                consequence="Moderate positive predictive value for CAD. Combined with other positive findings (exertional angina, reduced max HR), it significantly strengthens the clinical diagnosis.",
                severity="moderate"))
        else:
            d.update(dict(what_it_is=f"ST depression: {op:.1f} mm (Normal)",
                what_it_means="No significant ST depression during exercise — a negative stress test finding.",
                mechanism="Normal coronary flow reserve maintained during exercise. Myocardium receives adequate oxygenation even under peak physiological stress.",
                consequence="Normal ST response has a high negative predictive value — makes flow-limiting CAD substantially less likely.",
                severity="low"))

    elif feature == "ca":
        ca = int(raw_val)
        vessel_text = {0:"No blocked vessels",1:"1 vessel blocked",
                       2:"2 vessels blocked",3:"3 vessels blocked"}
        mechanisms = {
            0:"No significant coronary stenosis on fluoroscopy. Coronary blood flow is unobstructed in all major vessels.",
            1:"One major coronary artery has >50% stenosis. Single-vessel CAD limits perfusion to one myocardial territory.",
            2:"Two major coronary arteries critically narrowed. Bi-vessel CAD significantly limits total coronary reserve — a larger territory of myocardium is chronically underperfused.",
            3:"All three major coronary arteries critically narrowed (LAD, LCx, RCA). Triple-vessel CAD — the most severe anatomical pattern — leaves virtually no coronary reserve."
        }
        consequences = {
            0:"Absence of coronary stenosis is strongly protective. Low risk of ischaemic events.",
            1:"Single-vessel CAD: 2-3x increased cardiac event risk. Angioplasty or stenting of the culprit vessel typically restores normal prognosis.",
            2:"Bi-vessel CAD: 4-5x increased risk. Revascularisation strategy must address both vessels — CABG is often preferred over PCI.",
            3:"Triple-vessel CAD: highest-risk category. CABG is the gold standard treatment. Without intervention, 5-year mortality is substantially elevated. This is the most powerful predictor in the dataset."
        }
        d.update(dict(
            what_it_is=f"Coronary vessels blocked: {ca} ({vessel_text[ca]})",
            what_it_means=f"Fluoroscopy (contrast X-ray) found {ca} major coronary {'artery' if ca==1 else 'arteries'} with >50% narrowing.",
            mechanism=mechanisms[ca],
            consequence=consequences[ca],
            severity={0:"low",1:"moderate",2:"high",3:"high"}[ca]))

    elif feature == "thal":
        tv = int(raw_val)
        if tv == 7:
            d.update(dict(what_it_is="Thalassemia: Reversible defect",
                what_it_means="Thallium scan shows reduced blood flow to heart muscle under stress that recovers at rest — the direct imaging definition of myocardial ischaemia.",
                mechanism="During stress, a critically narrowed coronary artery cannot increase flow to meet demand. Thallium (a potassium analogue taken up by perfused cells) creates a cold spot in underperfused zones. At rest, flow recovers — the defect fills in (reversible). This directly images living but ischaemic myocardium.",
                consequence="A reversible perfusion defect is the most significant thal finding. It directly visualises tissue at immediate risk of infarction if the causative stenosis is untreated. This is a Class I indication for coronary angiography and consideration of revascularisation.",
                severity="high"))
        elif tv == 6:
            d.update(dict(what_it_is="Thalassemia: Fixed defect",
                what_it_means="Thallium scan shows a permanent cold spot — scar tissue from a previous myocardial infarction.",
                mechanism="A fixed defect represents infarcted (irreversibly damaged) myocardium from a prior event. Dead myocardium does not take up thallium at rest or under stress. The scar is electrically and mechanically silent.",
                consequence="Fixed defects indicate prior MI and reduced viable myocardium. The degree of LV function loss depends on infarct size. Remaining viable tissue adjacent to the scar may still be at risk if coronary disease persists.",
                severity="moderate"))
        else:
            d.update(dict(what_it_is="Thalassemia: Normal",
                what_it_means="Thallium scan shows normal, uniform perfusion throughout the myocardium at rest and under stress.",
                mechanism="All myocardial segments receive adequate blood flow during both rest and maximal stress. No areas of ischaemia or infarction detected.",
                consequence="A normal perfusion scan has very high negative predictive value for significant CAD. Annual cardiac event rate in patients with normal stress perfusion imaging is less than 1%.",
                severity="low"))

    elif feature == "slope":
        sl = int(raw_val)
        labels = {1:"Upsloping",2:"Flat",3:"Downsloping"}
        if sl == 3:
            d.update(dict(what_it_is=f"ST slope: Downsloping (Most abnormal pattern)",
                what_it_means="ST segment slopes downward during peak exercise — the most ominous ECG pattern in stress testing.",
                mechanism="Downsloping ST depression reflects widespread subendocardial ischaemia during peak demand. It is more specific for obstructive CAD than horizontal or upsloping depression. The degree of slope correlates with the haemodynamic severity of coronary stenosis.",
                consequence="Downsloping ST depression has the highest positive predictive value for obstructive multi-vessel CAD among all stress ECG findings. It is an indication for urgent further evaluation — angiography is typically the next step.",
                severity="high"))
        elif sl == 2:
            d.update(dict(what_it_is=f"ST slope: Flat (Borderline abnormal)",
                what_it_means="Flat ST segment at peak exercise — borderline abnormal and clinically significant in context.",
                mechanism="Flat ST depression suggests subendocardial ischaemia during peak demand. Less specific than downsloping but still associated with significant CAD, particularly when combined with other positive findings.",
                consequence="Warrants further evaluation, especially combined with other risk markers. Positive predictive value improves substantially when other test findings are also abnormal.",
                severity="moderate"))
        else:
            d.update(dict(what_it_is=f"ST slope: Upsloping (Normal pattern)",
                what_it_means="ST segment slopes upward during exercise — a normal physiological response.",
                mechanism="Upsloping ST change during exercise, particularly if mild, is considered benign. The heart's electrical recovery accelerates appropriately with increased heart rate.",
                consequence="Upsloping pattern reduces the probability of significant ischaemia. Combined with other normal findings, it suggests an adequate myocardial stress response.",
                severity="low"))

    elif feature == "fbs":
        if int(raw_val) == 1:
            d.update(dict(what_it_is="Fasting blood sugar: >120 mg/dL (Diabetic range)",
                what_it_means="Elevated fasting glucose indicates diabetes mellitus — a major independent cardiac risk factor that amplifies every other risk.",
                mechanism="Chronic hyperglycaemia damages vascular endothelium through multiple pathways: advanced glycation end-products (AGEs) stiffen arterial walls, oxidative stress promotes LDL oxidation, and systemic inflammation accelerates plaque formation. Diabetics also develop autonomic neuropathy — which masks cardiac symptoms, leading to silent ischaemia.",
                consequence="Diabetes increases cardiovascular mortality 2-4x. Diabetic patients tend to have more extensive, diffuse coronary disease and worse outcomes after MI. They are also far more likely to have completely silent ischaemia — their disease is underdiagnosed until a major event.",
                severity="high"))
        else:
            d.update(dict(what_it_is="Fasting blood sugar: Normal",
                what_it_means="No evidence of diabetes — removing a major risk amplifier from the equation.",
                mechanism="Normal glucose metabolism preserves endothelial function and prevents glycation-induced vascular damage.",
                consequence="Absence of diabetes significantly reduces background cardiovascular risk and improves prognosis across all other risk categories.",
                severity="low"))

    elif feature == "restecg":
        rv = int(raw_val)
        if rv == 2:
            d.update(dict(what_it_is="Resting ECG: Left Ventricular Hypertrophy (LVH)",
                what_it_means="The heart's main pumping chamber has abnormally thickened muscle walls — visible even at rest on ECG.",
                mechanism="LVH develops when the left ventricle is forced to pump against chronically elevated resistance (hypertension, aortic stenosis). The muscle thickens in response, but pathologically — it reduces chamber compliance, impairs diastolic filling, and dramatically increases myocardial oxygen demand. A thicker muscle needs more blood flow from already potentially diseased coronary arteries.",
                consequence="LVH is a powerful independent predictor of cardiac events. It represents years of haemodynamic overload. Thickened ventricles are prone to diastolic dysfunction, arrhythmias (especially atrial fibrillation), and heart failure even without obstructive coronary disease.",
                severity="high"))
        elif rv == 1:
            d.update(dict(what_it_is="Resting ECG: ST-T Wave Abnormality",
                what_it_means="Abnormal repolarisation pattern on the resting ECG — may indicate ischaemia or cardiomyopathy.",
                mechanism="ST-T wave changes at rest can reflect resting myocardial ischaemia (even without exertion), electrolyte abnormalities, medication effects, or structural heart disease. They indicate abnormal electrical recovery of the ventricular myocardium.",
                consequence="Resting ST-T changes increase pre-test probability of CAD and warrant further evaluation. They can also represent non-ischaemic cardiomyopathy.",
                severity="moderate"))
        else:
            d.update(dict(what_it_is="Resting ECG: Normal",
                what_it_means="Normal electrical activity at rest — no resting evidence of structural or ischaemic abnormality.",
                mechanism="Normal depolarisation and repolarisation indicates intact conduction pathways and no resting myocardial stress.",
                consequence="A normal resting ECG reduces (but does not eliminate) probability of significant structural heart disease.",
                severity="low"))

    elif feature == "sex":
        if int(raw_val) == 1:
            d.update(dict(what_it_is="Sex: Male",
                what_it_means="Male sex is an independent cardiac risk factor, particularly before age 65.",
                mechanism="Oestrogen in pre-menopausal women raises HDL, lowers LDL, and has vasodilatory effects on coronary arteries. Men lack this protection. Testosterone may promote pro-atherogenic lipid changes.",
                consequence="Men develop coronary artery disease approximately 10 years earlier than women on average. In the Cleveland dataset, male sex is one of the most prevalent risk markers.",
                severity="moderate"))
        else:
            d.update(dict(what_it_is="Sex: Female",
                what_it_means="Female sex is relatively protective before menopause, but risk equalises afterwards.",
                mechanism="Pre-menopausal oestrogen provides endothelial protection and favourable lipid profiles. After menopause, this protection is lost.",
                consequence="Women often present with atypical symptoms (fatigue, nausea) rather than classic angina — leading to underdiagnosis. The Cleveland dataset under-represents female patients.",
                severity="low"))

    return d


def deep_clinical_report(exp: ShapExplanation, raw_features: dict, top_k: int = 4) -> list[dict]:
    """
    Generate deep clinical analysis for top contributing features.
    raw_features must be the UNSCALED original patient values.
    """
    abs_vals = np.abs(exp.values)
    total = abs_vals.sum() or 1.0
    order = np.argsort(abs_vals)[::-1][:top_k]
    results = []
    for idx in order:
        feature  = exp.feature_names[idx]
        shap_val = float(exp.values[idx])
        raw_val  = raw_features.get(feature, 0.0)
        share    = float(abs_vals[idx] / total)
        analysis = _analyse_feature(feature, raw_val, shap_val)
        analysis["shap_val"] = shap_val
        analysis["share"]    = share
        results.append(analysis)
    return results


def combined_conclusion(analyses: list[dict], proba: float) -> str:
    """Single-paragraph medical conclusion synthesising all top factors."""
    risk_label = ("very high" if proba >= 0.75 else "high" if proba >= 0.50
                  else "moderate" if proba >= 0.20 else "low")
    high_sev = [a for a in analyses if a["severity"] == "high"]
    mod_sev  = [a for a in analyses if a["severity"] == "moderate"]
    parts = []
    if high_sev:
        labels = " and ".join(a["label"] for a in high_sev)
        parts.append(f"The dominant contributors to this {risk_label} risk prediction are "
            f"{labels} — each independently associated with significant coronary artery disease.")
    if mod_sev:
        labels = " and ".join(a["label"] for a in mod_sev)
        parts.append(f"Contributing further are {labels}, which compound the overall cardiovascular risk profile.")
    parts.append(
        f"Together, this combination of findings is consistent with a clinical picture that warrants "
        f"{'urgent' if proba >= 0.75 else 'prompt'} cardiovascular evaluation. "
        f"{'Coronary angiography should be strongly considered.' if proba >= 0.5 else 'A formal stress test and lipid panel are advisable next steps.'}"
    )
    return " ".join(parts)


def clinical_insight(exp: ShapExplanation, top_k: int = 4) -> list[str]:
    """Backward-compatible simple insight strings (used in reports/sidebar)."""
    abs_vals = np.abs(exp.values)
    total    = abs_vals.sum() or 1.0
    order    = np.argsort(abs_vals)[::-1][:top_k]
    sentences = []
    for idx in order:
        feature     = exp.feature_names[idx]
        contribution = float(exp.values[idx])
        share        = abs_vals[idx] / total
        direction    = "raised" if contribution > 0 else "lowered"
        magnitude    = ("substantially" if share > 0.30 else
                        "moderately"    if share > 0.15 else
                        "slightly"      if share > 0.05 else "marginally")
        english_name = FEATURE_ENGLISH.get(feature, feature)
        sentences.append(
            f"Your {english_name} {magnitude} {direction} the predicted risk "
            f"({share * 100:.0f}% of the total explanation)."
        )
    return sentences


# ── LIME ─────────────────────────────────────────────────────────────────────
def build_lime_explainer(X_train: pd.DataFrame, class_names=("no_disease","disease")):
    return LimeTabularExplainer(
        training_data=X_train.values,
        feature_names=list(X_train.columns),
        class_names=list(class_names),
        discretize_continuous=True, mode="classification",
    )

def lime_explain(lime_explainer, model, x: pd.DataFrame, num_features: int = 6):
    exp = lime_explainer.explain_instance(
        data_row=x.values[0], predict_fn=model.predict_proba,
        num_features=num_features,
    )
    return exp.as_list()


# ── Counterfactuals ───────────────────────────────────────────────────────────
def generate_counterfactual(model, x: pd.DataFrame, target_proba: float = 0.30,
    mutable_features: list[str] | None = None,
    step_grid: dict[str, list[float]] | None = None,
    max_changes: int = 2) -> CounterfactualResult:

    mutable_features = mutable_features or ["chol","trestbps","thalach","oldpeak"]
    step_grid = step_grid or {
        "chol":    [-0.5,-1.0,-1.5],
        "trestbps":[-0.3,-0.6,-1.0],
        "thalach": [0.3, 0.6, 1.0],
        "oldpeak": [-0.3,-0.6,-1.0],
    }
    baseline_proba = float(model.predict_proba(x)[0,1])
    best: CounterfactualResult | None = None

    def search(current: pd.DataFrame, used: list[str], depth: int) -> None:
        nonlocal best
        proba = float(model.predict_proba(current)[0,1])
        if proba <= target_proba:
            deltas = {f:(float(x.iloc[0][f]),float(current.iloc[0][f])) for f in used}
            cand = CounterfactualResult(
                original_proba=baseline_proba, target_proba=proba, deltas=deltas)
            if best is None or len(cand.deltas) < len(best.deltas):
                best = cand
            return
        if depth >= max_changes:
            return
        for feat in mutable_features:
            if feat in used:
                continue
            for step in step_grid.get(feat,[]):
                trial = current.copy()
                trial[feat] = trial[feat] + step
                search(trial, used+[feat], depth+1)

    search(x.copy(), [], 0)

    if best is None:
        return CounterfactualResult(
            original_proba=baseline_proba, target_proba=baseline_proba,
            narrative="No small modification within the search grid lowered risk to the target.")

    parts = []
    for feat,(orig,new) in best.deltas.items():
        english = FEATURE_ENGLISH.get(feat, feat)
        direction = "lower" if new < orig else "raise"
        parts.append(f"{direction} {english} from {orig:.2f} to {new:.2f}")
    best.narrative = (
        f"If you {' and '.join(parts)}, predicted risk drops from "
        f"{baseline_proba*100:.1f}% to {best.target_proba*100:.1f}%."
    )
    return best


# ── SHAP interaction values ───────────────────────────────────────────────────
def top_feature_interactions(explainer, X: pd.DataFrame, top_k: int = 5):
    try:
        interactions = explainer.shap_interaction_values(X)
    except Exception:
        return []
    arr = np.array(interactions)
    if arr.ndim == 4:
        arr = arr[1]
    mean_abs = np.mean(np.abs(arr), axis=0)
    np.fill_diagonal(mean_abs, 0)
    pairs = []
    n = mean_abs.shape[0]
    for i in range(n):
        for j in range(i+1,n):
            pairs.append((X.columns[i], X.columns[j], float(mean_abs[i,j])))
    pairs.sort(key=lambda t:t[2], reverse=True)
    return pairs[:top_k]
