"""Data-quality and governance summaries for the portfolio dashboard."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict

import pandas as pd

from .preprocess import PII_PATTERNS


def governance_summary(df: pd.DataFrame) -> Dict[str, float | int]:
    duplicate_mask = df.duplicated(subset=["pseudo_patient_id", "timestamp", "raw_text"], keep=False)
    return {
        "contacts": int(len(df)),
        "missing_raw_text_rate": float(df["raw_text"].isna().mean()),
        "empty_clean_text_rate": float(df["is_empty_text"].mean()),
        "short_text_rate": float(df["is_short_text"].mean()),
        "duplicate_contact_rate": float(duplicate_mask.mean()),
        "contacts_with_redaction_rate": float(df["redaction_count"].gt(0).mean()),
        "total_redactions": int(df["redaction_count"].sum()),
        "channels": int(df["channel"].nunique()),
        "service_lines": int(df["service_line"].nunique()),
    }


def redaction_counts(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for entity in PII_PATTERNS:
        column = f"redaction_{entity}"
        rows.append({
            "entity_type": entity,
            "redactions": int(df[column].sum()),
            "contacts_affected": int(df[column].gt(0).sum()),
            "affected_contact_rate": float(df[column].gt(0).mean()),
        })
    return pd.DataFrame(rows).sort_values("redactions", ascending=False)


def channel_quality(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.groupby("channel")
        .agg(
            contacts=("contact_id", "size"),
            mean_text_length=("text_length", "mean"),
            short_text_rate=("is_short_text", "mean"),
            redaction_rate=("redaction_count", lambda x: float(x.gt(0).mean())),
            mean_redactions=("redaction_count", "mean"),
        )
        .reset_index()
        .sort_values("contacts", ascending=False)
    )


def write_governance_reports(df: pd.DataFrame, reports_dir: Path) -> None:
    reports_dir.mkdir(parents=True, exist_ok=True)
    with open(reports_dir / "governance_summary.json", "w", encoding="utf-8") as f:
        json.dump(governance_summary(df), f, indent=2)
    redaction_counts(df).to_csv(reports_dir / "entity_redaction_counts.csv", index=False)
    channel_quality(df).to_csv(reports_dir / "data_quality_by_channel.csv", index=False)
    sample_cols = ["contact_id", "channel", "raw_text", "clean_text", "redaction_count"]
    sample = df[df["redaction_count"].gt(0)][sample_cols].head(100)
    sample.to_csv(reports_dir / "redaction_audit_sample.csv", index=False)
