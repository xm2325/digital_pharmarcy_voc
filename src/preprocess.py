"""Text normalisation and rule-based personal-data redaction.

The default redactor is intentionally local and auditable. It is not a replacement
for a reviewed production de-identification service. An optional Microsoft Presidio
adapter is documented separately for production-oriented experimentation.
"""
from __future__ import annotations

import re
from collections import Counter
from typing import Dict, Tuple

import pandas as pd

PII_PATTERNS: Dict[str, re.Pattern[str]] = {
    "email": re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"),
    "phone": re.compile(r"\b(?:\+?44\s?\(?(?:0\)?\s?)?|0)(?:7\d{3}|1\d{2,4}|2\d{1,2})[\s-]?\d{3,4}[\s-]?\d{3,4}\b"),
    "date_of_birth": re.compile(r"\b(?:0?[1-9]|[12]\d|3[01])[/-](?:0?[1-9]|1[0-2])[/-](?:19|20)\d{2}\b"),
    "postcode": re.compile(r"\b(?:GIR\s?0AA|[A-Z]{1,2}\d[A-Z\d]?\s?\d[A-Z]{2})\b", re.IGNORECASE),
    "nhs_number_like": re.compile(r"\b(?:\d[ -]?){9}\d\b"),
    "order_reference": re.compile(r"\b(?:order|reference|ref)\s*(?:number|no\.?|#|:)??\s*[A-Z0-9]{0,6}[- ]?\d{5,10}\b", re.IGNORECASE),
    "named_person": re.compile(r"\b(?:my name is|this is)\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,2}\b", re.IGNORECASE),
    "gp_surgery": re.compile(r"\b[A-Z][A-Za-z&' -]{2,45}\s(?:Medical Centre|Health Centre|Surgery|Practice)\b", re.IGNORECASE),
    "street_address": re.compile(r"\b\d{1,4}\s+[A-Z][A-Za-z' -]{2,35}\s(?:Road|Street|Lane|Avenue|Close|Drive|Way|Gardens|Court)\b", re.IGNORECASE),
}

PLACEHOLDERS = {
    "email": "[EMAIL]", "phone": "[PHONE]", "date_of_birth": "[DOB]", "postcode": "[POSTCODE]",
    "nhs_number_like": "[NHS_NUMBER]", "order_reference": "[ORDER_REF]", "named_person": "[NAME]",
    "gp_surgery": "[GP_SURGERY]", "street_address": "[ADDRESS]",
}

MULTISPACE_RE = re.compile(r"\s+")

NORMALISATION = {
    "pls": "please", "prescrption": "prescription", "delivry": "delivery", "aproval": "approval",
    "acct": "account", "meds": "medicine", "cos": "because", "yday": "yesterday", "asap": "as soon as possible",
    "appt": "appointment", "msg": "message", "gp's": "gp", "doc": "doctor", "rx": "prescription",
}


def redact_personal_data_detailed(text: str) -> Tuple[str, Dict[str, int]]:
    """Redact rule-based PII patterns and return counts by entity type."""
    redacted = str(text)
    counts: Counter[str] = Counter()
    # More specific patterns run before number-only patterns.
    for entity in [
        "email", "phone", "date_of_birth", "postcode", "order_reference", "named_person",
        "gp_surgery", "street_address", "nhs_number_like",
    ]:
        redacted, n = PII_PATTERNS[entity].subn(PLACEHOLDERS[entity], redacted)
        counts[entity] += int(n)
    return redacted, dict(counts)


def redact_personal_data(text: str) -> tuple[str, int]:
    redacted, counts = redact_personal_data_detailed(text)
    return redacted, int(sum(counts.values()))


def clean_text_detailed(text: str) -> tuple[str, Dict[str, int]]:
    text, redaction_counts = redact_personal_data_detailed(str(text))
    text = text.lower()
    for src, dst in NORMALISATION.items():
        text = re.sub(rf"\b{re.escape(src)}\b", dst, text)
    text = re.sub(r"[^a-z0-9_\[\]@'\s-]", " ", text)
    text = MULTISPACE_RE.sub(" ", text).strip()
    return text, redaction_counts


def clean_text(text: str) -> tuple[str, int]:
    cleaned, counts = clean_text_detailed(text)
    return cleaned, int(sum(counts.values()))


def preprocess_contacts(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    cleaned = out["raw_text"].fillna("").map(clean_text_detailed)
    out["clean_text"] = cleaned.map(lambda x: x[0])
    out["redaction_count"] = cleaned.map(lambda x: int(sum(x[1].values())))
    for entity in PII_PATTERNS:
        out[f"redaction_{entity}"] = cleaned.map(lambda x, key=entity: int(x[1].get(key, 0)))
    out["text_length"] = out["clean_text"].str.len()
    out["token_count"] = out["clean_text"].str.split().str.len()
    out["is_short_text"] = out["text_length"].lt(24)
    out["is_empty_text"] = out["clean_text"].eq("")
    return out
