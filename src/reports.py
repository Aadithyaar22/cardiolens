"""
reports.py
----------
Generates a single-page clinical-style PDF report per patient with:
  - Header (patient ID, timestamp, model version)
  - Risk score gauge text + tier
  - Top-5 SHAP contributions (bar chart embedded as PNG)
  - Counterfactual narrative
  - Conformal interval
  - Disclaimer footer
"""

from __future__ import annotations

import io
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import mm
from reportlab.platypus import (
    Image,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)


def _shap_bar_image(feature_names: list[str], shap_values: np.ndarray) -> io.BytesIO:
    order = np.argsort(np.abs(shap_values))[::-1][:6]
    feats = [feature_names[i] for i in order][::-1]
    vals = [shap_values[i] for i in order][::-1]
    cols = ["#dc2626" if v > 0 else "#16a34a" for v in vals]

    fig, ax = plt.subplots(figsize=(5.5, 3))
    ax.barh(feats, vals, color=cols)
    ax.axvline(0, color="#666", lw=0.8)
    ax.set_title("Top contributors (SHAP)", fontsize=10)
    ax.set_xlabel("Contribution to predicted risk")
    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=160, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return buf


def build_report(
    output_path: str | Path,
    patient_id: str,
    risk_proba: float,
    risk_tier: str,
    risk_color: str,
    feature_names: list[str],
    shap_values: np.ndarray,
    insights: list[str],
    counterfactual_text: str,
    interval_lower: float,
    interval_upper: float,
    model_name: str = "CardioLens v0.1",
) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    doc = SimpleDocTemplate(
        str(output_path),
        pagesize=A4,
        leftMargin=18 * mm,
        rightMargin=18 * mm,
        topMargin=18 * mm,
        bottomMargin=18 * mm,
    )

    styles = getSampleStyleSheet()
    h1 = ParagraphStyle("h1", parent=styles["Heading1"], fontSize=18, spaceAfter=4)
    sub = ParagraphStyle("sub", parent=styles["Normal"], textColor=colors.grey, fontSize=9)
    body = styles["BodyText"]
    body.spaceAfter = 4

    story = []
    story.append(Paragraph("CardioLens — Patient Risk Report", h1))
    story.append(Paragraph(
        f"Patient ID: <b>{patient_id}</b> &nbsp;&nbsp;|&nbsp;&nbsp; "
        f"Generated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')} "
        f"&nbsp;&nbsp;|&nbsp;&nbsp; Model: {model_name}",
        sub,
    ))
    story.append(Spacer(1, 8))

    summary = [
        ["Predicted risk", f"{risk_proba * 100:.1f}%"],
        ["Risk tier", risk_tier],
        ["90% prediction interval", f"{interval_lower * 100:.1f}% – {interval_upper * 100:.1f}%"],
    ]
    table = Table(summary, hAlign="LEFT", colWidths=[55 * mm, 70 * mm])
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#f1f5f9")),
        ("TEXTCOLOR", (1, 1), (1, 1), colors.HexColor(risk_color)),
        ("FONTNAME", (1, 1), (1, 1), "Helvetica-Bold"),
        ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
        ("BOX", (0, 0), (-1, -1), 0.4, colors.HexColor("#cbd5e1")),
        ("INNERGRID", (0, 0), (-1, -1), 0.3, colors.HexColor("#cbd5e1")),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("LEFTPADDING", (0, 0), (-1, -1), 8),
        ("RIGHTPADDING", (0, 0), (-1, -1), 8),
        ("TOPPADDING", (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
    ]))
    story.append(table)
    story.append(Spacer(1, 10))

    story.append(Paragraph("<b>Top contributors</b>", body))
    img_buf = _shap_bar_image(feature_names, shap_values)
    story.append(Image(img_buf, width=140 * mm, height=70 * mm))
    story.append(Spacer(1, 6))

    story.append(Paragraph("<b>Clinical insights</b>", body))
    for line in insights:
        story.append(Paragraph(f"• {line}", body))
    story.append(Spacer(1, 6))

    story.append(Paragraph("<b>Counterfactual</b>", body))
    story.append(Paragraph(counterfactual_text, body))
    story.append(Spacer(1, 10))

    story.append(Paragraph(
        "<i>Disclaimer: This report is for educational and decision-support purposes "
        "only and does not constitute medical advice. Consult a qualified clinician "
        "before making clinical decisions.</i>",
        sub,
    ))

    doc.build(story)
    return output_path
