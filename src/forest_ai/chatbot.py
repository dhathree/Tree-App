from __future__ import annotations

from dataclasses import dataclass


@dataclass
class RiskContext:
    risk_score: float
    risk_label: str
    tree_detections: int
    inference_engine: str


def answer(question: str, context: RiskContext) -> str:
    q = question.lower().strip()

    if any(k in q for k in ["risk", "danger", "fire"]):
        return (
            f"Current estimated fire risk is {context.risk_score:.1f}/100 ({context.risk_label}). "
            "Recommended action: increase watchtower/drone frequency and create fireline checks in high-fuel zones."
        )

    if any(k in q for k in ["tree", "species", "detected"]):
        return (
            f"Detected {context.tree_detections} tree-related objects using {context.inference_engine}. "
            "For species-level precision, train YOLOv8 on your labeled local species dataset."
        )

    if any(k in q for k in ["why", "explain", "factors", "reason"]):
        return (
            "Top risk drivers generally include high temperature, low humidity, high wind speed, "
            "and low soil moisture. These variables are part of the prediction feature set."
        )

    return (
        "I can help with: current fire risk, detected tree count, and reasons behind the risk score. "
        "Try asking: 'What is the risk now?'"
    )
