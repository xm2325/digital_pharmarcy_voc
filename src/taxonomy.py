"""Intent taxonomy and safe-routing rules for the synthetic pharmacy VoC demo."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List


@dataclass(frozen=True)
class IntentSpec:
    intent: str
    category: str
    journey_stage: str
    root_cause: str
    default_route: str
    safety_sensitive: bool = False
    chatbot_eligible: bool = False
    proactive_candidate: bool = False


INTENT_SPECS: List[IntentSpec] = [
    IntentSpec("repeat_prescription_request", "Prescription request", "Request", "Routine repeat request", "FAQ self-service", chatbot_eligible=True),
    IntentSpec("early_repeat_request", "Prescription request", "Request", "Request submitted earlier than expected", "Human adviser"),
    IntentSpec("prescription_not_received", "Prescription request", "Request", "Prescription request not visible", "Human adviser"),
    IntentSpec("waiting_gp_approval", "GP approval", "Await GP approval", "GP approval pending", "Proactive message", chatbot_eligible=True, proactive_candidate=True),
    IntentSpec("gp_declined_request", "GP approval", "Await GP approval", "GP declined or changed request", "Human adviser"),
    IntentSpec("clinical_review_delay", "Clinical and dispensing", "Clinical review", "Pharmacist review pending", "Human adviser"),
    IntentSpec("medicine_unavailable", "Clinical and dispensing", "Dispensing", "Stock availability issue", "Human adviser"),
    IntentSpec("split_order_request", "Clinical and dispensing", "Dispensing", "One item delays a multi-item order", "Human adviser"),
    IntentSpec("not_dispatched", "Dispatch and delivery", "Packing", "Order packed but not dispatched", "Proactive message", chatbot_eligible=True, proactive_candidate=True),
    IntentSpec("delivery_tracking", "Dispatch and delivery", "Dispatched", "Customer requests tracking details", "FAQ self-service", chatbot_eligible=True),
    IntentSpec("delivery_delayed", "Dispatch and delivery", "Delivery", "Carrier delay after dispatch", "Proactive message", chatbot_eligible=True, proactive_candidate=True),
    IntentSpec("parcel_not_received", "Dispatch and delivery", "Delivery", "Parcel marked delivered or overdue", "Human adviser"),
    IntentSpec("change_delivery_address", "Account management", "Account", "Delivery address needs update", "FAQ self-service", chatbot_eligible=True),
    IntentSpec("login_issue", "Account management", "Account", "Account access problem", "FAQ self-service", chatbot_eligible=True),
    IntentSpec("reminder_missing", "Reminders and notifications", "Notifications", "Reorder reminder not received", "FAQ self-service", chatbot_eligible=True),
    IntentSpec("wrong_notification", "Reminders and notifications", "Notifications", "Incorrect or confusing message", "Human adviser"),
    IntentSpec("complaint_service", "Service access", "Service", "General service complaint", "Human adviser"),
    IntentSpec("refund_query", "Online Doctor and shop", "Shop or Online Doctor", "Refund progress query", "FAQ self-service", chatbot_eligible=True),
    IntentSpec("online_doctor_order_status", "Online Doctor and shop", "Shop or Online Doctor", "Private service order status", "FAQ self-service", chatbot_eligible=True),
    IntentSpec("urgent_ran_out_medicine", "Safety-sensitive", "Urgent support", "Customer reports no medicine remaining", "Clinical escalation", safety_sensitive=True),
    IntentSpec("possible_adverse_effect", "Safety-sensitive", "Urgent support", "Possible adverse effect or clinical concern", "Clinical escalation", safety_sensitive=True),
    IntentSpec("unclear_or_out_of_scope", "Out-of-scope", "Unknown", "Insufficient information", "Human adviser"),
]

BY_INTENT: Dict[str, IntentSpec] = {x.intent: x for x in INTENT_SPECS}
INTENTS = [x.intent for x in INTENT_SPECS]

SAFETY_TERMS = [
    "ran out", "no tablets left", "no medicine left", "urgent", "today", "breathing", "rash", "dizzy", "side effect",
    "adverse", "reaction", "chest pain", "very unwell", "need insulin", "need inhaler", "cannot wait"
]


def taxonomy_rows() -> List[dict]:
    return [x.__dict__.copy() for x in INTENT_SPECS]
