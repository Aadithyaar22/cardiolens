"""
risk.py
-------
Translate a calibrated probability into a 4-tier clinical risk stratification.
Thresholds chosen to roughly correspond to standard cardiology risk bands.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class RiskTier:
    label: str
    color: str        # hex color for UI
    advice: str       # short clinician-style guidance


def stratify(prob: float) -> RiskTier:
    if prob < 0.20:
        return RiskTier(
            label="Low",
            color="#16a34a",
            advice="Routine follow-up. Maintain healthy lifestyle.",
        )
    if prob < 0.50:
        return RiskTier(
            label="Moderate",
            color="#eab308",
            advice="Discuss preventive measures with a clinician within 6–12 months.",
        )
    if prob < 0.75:
        return RiskTier(
            label="High",
            color="#f97316",
            advice="Schedule a clinical evaluation. Consider stress test and lipid panel.",
        )
    return RiskTier(
        label="Critical",
        color="#dc2626",
        advice="Urgent clinical review recommended.",
    )
