"""Generate synthetic multi-channel pharmacy contact, CRM, FAQ, and annotation data."""
from __future__ import annotations

import argparse
import random
import re
import sqlite3
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

from .taxonomy import BY_INTENT, INTENT_SPECS, taxonomy_rows

SEED = 20260604

TEMPLATES: Dict[str, List[str]] = {
    "repeat_prescription_request": [
        "How do I order my repeat prescription?", "I need to request my usual tablets again", "Can I reorder my regular medicine through the app?"
    ],
    "early_repeat_request": [
        "I am going away next week, can I order my medication early?", "The app says it is too soon but I need my tablets before holiday", "Can you release an early repeat please?"
    ],
    "prescription_not_received": [
        "My GP says they sent the prescription but it is not showing", "Why have you not received my prescription yet?", "The request is missing from my account"
    ],
    "waiting_gp_approval": [
        "My order is waiting for GP approval, what should I do?", "Why is my prescription still with my GP?", "The app says awaiting GP approval for days"
    ],
    "gp_declined_request": [
        "My GP rejected my repeat request, can you tell me why?", "The prescription was declined by the surgery", "Why has my doctor refused the request?"
    ],
    "clinical_review_delay": [
        "My medicine has been in clinical review since yesterday", "Why is the pharmacist check taking so long?", "The order is stuck at clinical review"
    ],
    "medicine_unavailable": [
        "One item is out of stock, when will it be available?", "You said my medicine is unavailable", "There is a stock issue with my tablets"
    ],
    "split_order_request": [
        "Can you send the available items now and the missing one later?", "Please split my order because one medicine is delayed", "Can the rest of my medication be dispatched separately?"
    ],
    "not_dispatched": [
        "My order has been packing for two days and is not dispatched", "When will you post my tablets?", "The prescription is ready but has not left the warehouse"
    ],
    "delivery_tracking": [
        "Where can I find my Royal Mail tracking number?", "How do I track my medicine delivery?", "Please send me the tracking link for my parcel"
    ],
    "delivery_delayed": [
        "Royal Mail tracking has not moved and my delivery is late", "My prescription should have arrived yesterday", "The medicine parcel is delayed after dispatch"
    ],
    "parcel_not_received": [
        "Tracking says delivered but there is no parcel here", "My medicines have not arrived at all", "The parcel is missing even though it says delivered"
    ],
    "change_delivery_address": [
        "How can I change my delivery address?", "I moved house and need to update my address", "Please send future prescriptions to my new address"
    ],
    "login_issue": [
        "I cannot log into my account", "The app will not let me sign in", "I forgot my password and cannot access my prescription"
    ],
    "reminder_missing": [
        "I did not receive my reorder reminder", "Why have the reminder texts stopped?", "The app did not remind me to request my tablets"
    ],
    "wrong_notification": [
        "I received a dispatch message for the wrong order", "The notification says my order is cancelled but the app says packing", "Your text message does not match my account"
    ],
    "complaint_service": [
        "I have contacted you twice and nobody has resolved this", "I want to make a complaint about the service", "This has been frustrating and I need someone to help"
    ],
    "refund_query": [
        "When will my shop refund arrive?", "I returned an item and need a refund update", "How long does the refund take?"
    ],
    "online_doctor_order_status": [
        "Where is my Online Doctor order?", "Has my private prescription been dispatched?", "Can I track the order from my online consultation?"
    ],
    "urgent_ran_out_medicine": [
        "I have run out of my tablets and need help today", "I have no medicine left and cannot wait for delivery", "Urgent, I need my inhaler and there is none left"
    ],
    "possible_adverse_effect": [
        "I have a rash after taking the medicine, what should I do?", "These tablets make me dizzy and I feel very unwell", "I think I am having a reaction to my new medication"
    ],
    "unclear_or_out_of_scope": [
        "Can someone contact me?", "I have a question about something on the site", "Please help with my issue"
    ],
}

FAQ_ROWS = [
    ("How do I request a repeat prescription?", "Use your account or app to select the repeat medicines you need and submit the request to your GP surgery.", "repeat_prescription_request"),
    ("Why is my prescription waiting for GP approval?", "Your GP surgery must approve the medicine request before the pharmacy can prepare it. Contact your GP surgery if an urgent request needs clinical attention.", "waiting_gp_approval"),
    ("How do I track my order?", "Open your account to view the current order stage. After dispatch, use the carrier tracking details sent to you.", "delivery_tracking"),
    ("How do I change my delivery address?", "Update your delivery address in your account before placing the next prescription request.", "change_delivery_address"),
    ("Why did I not receive my reminder?", "Check your notification preferences and contact details in your account. You can update reminder dates and channels.", "reminder_missing"),
    ("What should I do if I have run out of medicine?", "Do not rely on automated support for urgent medicine needs. Contact your GP surgery, NHS 111 when appropriate, or seek urgent clinical advice.", "urgent_ran_out_medicine"),
    ("What should I do if I have a possible side effect?", "Possible adverse effects require clinical advice. Seek appropriate clinical support rather than relying on an automated answer.", "possible_adverse_effect"),
    ("How do I contact customer care?", "Use the Help and Support service to select the contact route that matches your query.", "complaint_service"),
]

FAQ_EVAL_ROWS = [
    ("Can I request my regular tablets again?", "repeat_prescription_request", True),
    ("The GP status has been pending for several days", "waiting_gp_approval", True),
    ("Where is the tracking link for the package?", "delivery_tracking", True),
    ("I have moved house. Where do I edit the delivery location?", "change_delivery_address", True),
    ("The reorder alert never arrived", "reminder_missing", True),
    ("I have no tablets left and need urgent help", "urgent_ran_out_medicine", True),
    ("The new medicine has caused a rash", "possible_adverse_effect", True),
    ("How can I speak to customer services?", "complaint_service", True),
    ("Can you diagnose my chest pain?", "unclear_or_out_of_scope", False),
    ("What are your opening times on Christmas Day?", "unclear_or_out_of_scope", False),
]

NOVEL_ISSUE_TEMPLATES = [
    "Since the latest app update it crashes whenever I open the prescription screen",
    "The mobile app closes immediately after the new update",
    "The app keeps freezing after I updated it and I cannot reach my prescriptions",
    "After installing the latest version the app will not load the order page",
]

AMBIGUOUS_TEMPLATES = {
    "Prescription request": ["There is a problem with my prescription request", "My repeat medicine request does not look right", "Can you check what happened to my request?"],
    "GP approval": ["My prescription is still waiting and I do not know why", "Can you check why the surgery step has not changed?", "The doctor-related status looks wrong"],
    "Clinical and dispensing": ["My medication order is stuck during processing", "One part of my order has not moved", "Can you check the pharmacy processing step?"],
    "Dispatch and delivery": ["My medicine order has not arrived yet", "Can you tell me what is happening with my parcel?", "The delivery status has not changed"],
    "Account management": ["There is an issue in my account", "I cannot update something in the app", "Please help me with my account settings"],
    "Reminders and notifications": ["The message in the app does not look right", "I have a notification issue", "Something is wrong with the text alerts"],
    "Service access": ["I need someone to sort this out", "Please ask a person to contact me", "I need help from customer services"],
    "Online Doctor and shop": ["I have a question about my other order", "Please check my private order", "Can you update me on the shop or doctor service?"],
    "Out-of-scope": ["Can someone help?", "I do not know which option to select", "Please get back to me"],
}

RELATED_INTENTS = {
    "waiting_gp_approval": ["prescription_not_received", "clinical_review_delay"],
    "clinical_review_delay": ["waiting_gp_approval", "not_dispatched"],
    "not_dispatched": ["clinical_review_delay", "delivery_tracking"],
    "delivery_tracking": ["not_dispatched", "delivery_delayed"],
    "delivery_delayed": ["delivery_tracking", "parcel_not_received"],
    "parcel_not_received": ["delivery_delayed", "delivery_tracking"],
    "reminder_missing": ["wrong_notification"],
    "wrong_notification": ["reminder_missing"],
    "repeat_prescription_request": ["early_repeat_request"],
    "early_repeat_request": ["repeat_prescription_request"],
    "medicine_unavailable": ["split_order_request"],
    "split_order_request": ["medicine_unavailable"],
}

CHANNELS = ["phone_transcript", "email", "live_chat", "app_review", "social"]
CHANNEL_PROBS = np.array([0.40, 0.22, 0.22, 0.11, 0.05])

BASE_WEIGHTS = {
    "repeat_prescription_request": 0.055, "early_repeat_request": 0.025, "prescription_not_received": 0.045,
    "waiting_gp_approval": 0.105, "gp_declined_request": 0.025, "clinical_review_delay": 0.055,
    "medicine_unavailable": 0.055, "split_order_request": 0.025, "not_dispatched": 0.075,
    "delivery_tracking": 0.105, "delivery_delayed": 0.115, "parcel_not_received": 0.055,
    "change_delivery_address": 0.040, "login_issue": 0.040, "reminder_missing": 0.035,
    "wrong_notification": 0.020, "complaint_service": 0.045, "refund_query": 0.020,
    "online_doctor_order_status": 0.025, "urgent_ran_out_medicine": 0.018, "possible_adverse_effect": 0.012,
    "unclear_or_out_of_scope": 0.030,
}

NEGATIVE_INTENTS = {
    "prescription_not_received", "waiting_gp_approval", "gp_declined_request", "clinical_review_delay", "medicine_unavailable",
    "not_dispatched", "delivery_delayed", "parcel_not_received", "login_issue", "wrong_notification", "complaint_service",
    "urgent_ran_out_medicine", "possible_adverse_effect"
}


def perturb_text(text: str, rng: random.Random, channel: str) -> str:
    prefixes = {
        "phone_transcript": ["hello um ", "hi I am calling because ", "right so ", "please can you help, "],
        "email": ["Dear team, ", "Hello, ", "Please help. ", "Hi support, "],
        "live_chat": ["hi - ", "pls help: ", "hello ", ""],
        "app_review": ["App issue: ", "Not happy. ", "", "Please fix this: "],
        "social": ["@support ", "Can someone help? ", "", "Please respond: "],
    }
    suffixes = ["", " thanks", " please", " asap", "!!!", "."]
    text = rng.choice(prefixes[channel]) + text + rng.choice(suffixes)
    substitutions = {
        "please": "pls", "prescription": "prescrption", "medicine": "meds", "delivery": "delivry",
        "account": "acct", "because": "cos", "number": "no", "yesterday": "yday", "approval": "aproval"
    }
    if rng.random() < 0.28:
        candidates = [k for k in substitutions if k in text.lower()]
        if candidates:
            src = rng.choice(candidates)
            text = re.sub(src, substitutions[src], text, count=1, flags=re.IGNORECASE)
    if rng.random() < 0.06:
        text += f" My phone is 07{rng.randint(100000000, 999999999)}"
    if rng.random() < 0.04:
        text += f" email {rng.choice(['alex','sam','pat','jo'])}{rng.randint(10,99)}@example.com"
    if rng.random() < 0.018:
        text += f" DOB {rng.randint(1,28):02d}/{rng.randint(1,12):02d}/{rng.randint(1940,2003)}"
    if rng.random() < 0.018:
        text += f" postcode {rng.choice(['LS15 8GB','M13 9PL','SW1A 1AA','B15 2TT'])}"
    if rng.random() < 0.014:
        text += f" order no P2U-{rng.randint(100000,999999)}"
    if rng.random() < 0.010:
        text += f" my name is {rng.choice(['Alex Smith','Sam Patel','Jo Brown','Pat Jones'])}"
    if rng.random() < 0.010:
        text += f" from {rng.choice(['Oakfield Medical Centre','Riverside Surgery','Spring Lane Practice'])}"
    if rng.random() < 0.008:
        text += f" address {rng.choice(['12 Oak Road','45 Spring Lane','8 Market Street'])}"
    if rng.random() < 0.008:
        text += f" NHS number {rng.randint(1000000000,9999999999)}"
    return text


def choose_intent(ts: pd.Timestamp, rng: np.random.Generator) -> str:
    weights = BASE_WEIGHTS.copy()
    # Add a visible emerging delivery issue in the final three weeks.
    if ts >= pd.Timestamp("2026-05-11"):
        weights["delivery_delayed"] *= 2.4
        weights["parcel_not_received"] *= 1.5
    intents = list(weights)
    probs = np.array([weights[x] for x in intents], dtype=float)
    probs /= probs.sum()
    return str(rng.choice(intents, p=probs))


def build_contacts(n: int, output_dir: Path) -> pd.DataFrame:
    py_rng = random.Random(SEED)
    rng = np.random.default_rng(SEED)
    timestamps = pd.to_datetime(
        rng.integers(
            pd.Timestamp("2026-01-01").value // 10**9,
            pd.Timestamp("2026-06-01").value // 10**9,
            size=n,
        ),
        unit="s",
    )
    timestamps = pd.Series(timestamps).sort_values().reset_index(drop=True)
    patients = [f"P{idx:06d}" for idx in range(1, max(4500, n // 4) + 1)]

    intent_names = list(BASE_WEIGHTS)
    normal_probs = np.array([BASE_WEIGHTS[x] for x in intent_names], dtype=float)
    normal_probs /= normal_probs.sum()
    late_weights = BASE_WEIGHTS.copy()
    late_weights["delivery_delayed"] *= 2.4
    late_weights["parcel_not_received"] *= 1.5
    late_probs = np.array([late_weights[x] for x in intent_names], dtype=float)
    late_probs /= late_probs.sum()
    late_mask = timestamps.ge(pd.Timestamp("2026-05-11")).to_numpy()
    chosen_intents = np.empty(n, dtype=object)
    chosen_intents[~late_mask] = rng.choice(intent_names, size=int((~late_mask).sum()), p=normal_probs)
    chosen_intents[late_mask] = rng.choice(intent_names, size=int(late_mask.sum()), p=late_probs)
    chosen_channels = rng.choice(CHANNELS, size=n, p=CHANNEL_PROBS)
    chosen_patients = rng.choice(patients, size=n)

    rows = []
    for idx, (ts, intent, channel, patient_id) in enumerate(
        zip(timestamps, chosen_intents, chosen_channels, chosen_patients), 1
    ):
        intent = str(intent)
        channel = str(channel)
        synthetic_issue_pattern = "standard_known_intent"
        # Introduce a late out-of-taxonomy app-crash pattern for the topic-discovery queue.
        # It remains labelled as unclear_or_out_of_scope until reviewed by a human.
        if ts >= pd.Timestamp("2026-05-11") and rng.random() < 0.060:
            intent = "unclear_or_out_of_scope"
            synthetic_issue_pattern = "novel_app_crash"
        spec = BY_INTENT[intent]
        base_text = py_rng.choice(NOVEL_ISSUE_TEMPLATES if synthetic_issue_pattern == "novel_app_crash" else TEMPLATES[intent])
        # Real customer contacts are often incomplete or contain more than one request.
        if not spec.safety_sensitive and spec.category in AMBIGUOUS_TEMPLATES and rng.random() < 0.14:
            base_text = py_rng.choice(AMBIGUOUS_TEMPLATES[spec.category])
        elif not spec.safety_sensitive and intent in RELATED_INTENTS and rng.random() < 0.16:
            secondary_intent = str(rng.choice(RELATED_INTENTS[intent]))
            base_text = base_text + " Also, " + py_rng.choice(TEMPLATES[secondary_intent])
        raw = perturb_text(base_text, py_rng, channel)
        sentiment = "negative" if intent in NEGATIVE_INTENTS else str(rng.choice(["neutral", "positive"], p=[0.77, 0.23]))
        if intent == "unclear_or_out_of_scope":
            sentiment = str(rng.choice(["neutral", "negative"], p=[0.62, 0.38]))
        urgency = "urgent clinical escalation" if spec.safety_sensitive else ("priority" if intent in NEGATIVE_INTENTS and rng.random() < 0.35 else "standard")
        rows.append({
            "contact_id": f"C{idx:07d}", "pseudo_patient_id": str(patient_id), "timestamp": ts,
            "channel": channel, "service_line": "NHS prescription" if intent not in {"refund_query", "online_doctor_order_status"} else "Online Doctor or shop",
            "raw_text": raw, "intent": intent, "category": spec.category, "journey_stage": spec.journey_stage,
            "root_cause": spec.root_cause, "sentiment": sentiment, "urgency": urgency,
            "default_route": spec.default_route, "safety_sensitive": spec.safety_sensitive,
            "chatbot_eligible": spec.chatbot_eligible, "proactive_candidate": spec.proactive_candidate,
            "synthetic_issue_pattern": synthetic_issue_pattern,
        })
    df = pd.DataFrame(rows)
    ordered = df.sort_values(["pseudo_patient_id", "timestamp"]).copy()
    ordered["repeat_contact_14d"] = (
        ordered.groupby("pseudo_patient_id")["timestamp"].diff().dt.days.fillna(999).le(14)
    )
    df = ordered.sort_index()
    df.to_csv(output_dir / "synthetic_contacts_raw.csv", index=False)
    return df


def build_crm(contacts: pd.DataFrame, output_dir: Path) -> pd.DataFrame:
    rng = np.random.default_rng(SEED + 1)
    agg = contacts.groupby("pseudo_patient_id").agg(
        contact_count=("contact_id", "size"),
        negative_contacts=("sentiment", lambda x: int((x == "negative").sum())),
        repeat_contacts=("repeat_contact_14d", "sum"),
        safety_contacts=("safety_sensitive", "sum"),
        first_contact=("timestamp", "min"), last_contact=("timestamp", "max"),
    ).reset_index()
    dissatisfaction = np.minimum(1.0, 0.12 + 0.11 * agg["negative_contacts"] + 0.08 * agg["repeat_contacts"])
    agg["csat_score"] = np.clip(np.rint(5 - 3.3 * dissatisfaction + rng.normal(0, 0.55, len(agg))), 1, 5).astype(int)
    churn_prob = np.clip(0.035 + 0.06 * agg["negative_contacts"] + 0.075 * agg["repeat_contacts"], 0, 0.78)
    agg["retained_90d"] = rng.random(len(agg)) > churn_prob
    agg["crm_segment"] = rng.choice(["new", "routine", "high_support_need"], p=[0.22, 0.65, 0.13], size=len(agg))
    agg.to_csv(output_dir / "synthetic_crm.csv", index=False)
    return agg


def write_supporting_files(output_dir: Path) -> None:
    pd.DataFrame(FAQ_ROWS, columns=["question", "answer", "mapped_intent"]).to_csv(output_dir / "faq_knowledge_base.csv", index=False)
    pd.DataFrame(FAQ_EVAL_ROWS, columns=["query", "expected_intent", "expected_answerable"]).to_csv(output_dir / "faq_retrieval_eval.csv", index=False)
    pd.DataFrame(taxonomy_rows()).to_csv(output_dir / "intent_taxonomy.csv", index=False)
    pd.DataFrame([
        {"taxonomy_version": "v1.0", "change_date": "2026-05-20", "status": "historical", "change_summary": "Initial digital-pharmacy taxonomy"},
        {"taxonomy_version": "v2.0", "change_date": "2026-06-04", "status": "current", "change_summary": "Added taxonomy-review queue, governance fields, and unknown-topic workflow"},
    ]).to_csv(output_dir / "taxonomy_versions.csv", index=False)
    pd.DataFrame({
        "intent": [x.intent for x in INTENT_SPECS],
        "definition": [x.root_cause for x in INTENT_SPECS],
        "include_when": [f"Text primarily concerns: {x.root_cause.lower()}" for x in INTENT_SPECS],
        "exclude_when": ["Use another label when a more specific primary contact driver is present." for _ in INTENT_SPECS],
    }).to_csv(output_dir / "annotation_guidelines.csv", index=False)


def write_sqlite(output_dir: Path, contacts: pd.DataFrame, crm: pd.DataFrame) -> None:
    db_path = output_dir / "voc_demo.db"
    with sqlite3.connect(db_path) as conn:
        contacts.to_sql("contacts_raw", conn, if_exists="replace", index=False)
        crm.to_sql("crm", conn, if_exists="replace", index=False)
        pd.read_csv(output_dir / "faq_knowledge_base.csv").to_sql("faq_knowledge_base", conn, if_exists="replace", index=False)
        pd.read_csv(output_dir / "intent_taxonomy.csv").to_sql("intent_taxonomy", conn, if_exists="replace", index=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=30000)
    parser.add_argument("--output-dir", type=Path, default=Path("data"))
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    contacts = build_contacts(args.n, args.output_dir)
    crm = build_crm(contacts, args.output_dir)
    write_supporting_files(args.output_dir)
    write_sqlite(args.output_dir, contacts, crm)
    print(f"Generated {len(contacts):,} synthetic contacts and {len(crm):,} CRM rows in {args.output_dir}")


if __name__ == "__main__":
    main()
