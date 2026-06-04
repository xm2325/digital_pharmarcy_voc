"""Safe routing rules and deflection simulation."""
from __future__ import annotations

import re
from typing import Iterable

import numpy as np
import pandas as pd

from .taxonomy import BY_INTENT, SAFETY_TERMS


def contains_safety_language(text: str) -> bool:
    lowered = str(text).lower()
    return any(re.search(rf"\b{re.escape(term)}\b", lowered) for term in SAFETY_TERMS)


def assign_route(predicted_intent: str, confidence: float, clean_text: str, threshold: float = 0.72) -> str:
    spec = BY_INTENT.get(predicted_intent, BY_INTENT["unclear_or_out_of_scope"])
    if spec.safety_sensitive or contains_safety_language(clean_text):
        return "Clinical escalation"
    if confidence < threshold:
        return "Human adviser"
    if spec.default_route == "Proactive message":
        return "Proactive message"
    if spec.chatbot_eligible:
        return "Chatbot or FAQ self-service"
    return "Human adviser"


def add_routes(df: pd.DataFrame, threshold: float = 0.72) -> pd.DataFrame:
    out = df.copy()
    out["recommended_route"] = [
        assign_route(i, float(c), t, threshold)
        for i, c, t in zip(out["predicted_intent"], out["prediction_confidence"], out["clean_text"])
    ]
    return out


def deflection_scenario(df: pd.DataFrame, threshold: float = 0.72, contact_cost: float = 4.50, proactive_cost: float = 0.12) -> dict:
    routed = add_routes(df, threshold)
    counts = routed["recommended_route"].value_counts().to_dict()
    total = len(routed)
    auto = counts.get("Chatbot or FAQ self-service", 0)
    proactive = counts.get("Proactive message", 0)
    human = counts.get("Human adviser", 0)
    clinical = counts.get("Clinical escalation", 0)
    gross_avoided = (auto + proactive) * contact_cost
    comms_cost = proactive * proactive_cost
    return {
        "total_contacts": total,
        "automated_or_self_service": auto,
        "proactive_message": proactive,
        "human_adviser": human,
        "clinical_escalation": clinical,
        "potential_deflected_contacts": auto + proactive,
        "potential_deflection_rate": (auto + proactive) / total if total else 0.0,
        "assumed_gross_contact_value_gbp": gross_avoided,
        "assumed_proactive_message_cost_gbp": comms_cost,
        "assumed_net_value_gbp": gross_avoided - comms_cost,
    }
