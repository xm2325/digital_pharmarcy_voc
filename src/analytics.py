"""Trend, CRM, and dashboard analytics."""
from __future__ import annotations

import numpy as np
import pandas as pd


def weekly_intent_trends(df: pd.DataFrame) -> pd.DataFrame:
    work = df.copy()
    work["week"] = pd.to_datetime(work["timestamp"]).dt.to_period("W-MON").dt.start_time
    grouped = work.groupby(["week", "predicted_intent"]).size().rename("contact_count").reset_index()
    totals = grouped.groupby("week")["contact_count"].transform("sum")
    grouped["share"] = grouped["contact_count"] / totals
    return grouped


def emerging_issues(df: pd.DataFrame, min_contacts: int = 15, recent_weeks: int = 3) -> pd.DataFrame:
    """Compare the recent multi-week share with the earlier baseline.

    A multi-week window is less sensitive to a partial final week and is more useful
    for operational review than a single-week spike.
    """
    trends = weekly_intent_trends(df)
    rows = []
    for intent, grp in trends.groupby("predicted_intent"):
        grp = grp.sort_values("week")
        if len(grp) < recent_weeks + 3:
            continue
        latest = grp.iloc[-1]
        recent = grp.iloc[-recent_weeks:]
        historical = grp.iloc[:-recent_weeks]["share"]
        mean = historical.mean()
        std = historical.std(ddof=0)
        recent_mean = recent["share"].mean()
        z = (recent_mean - mean) / (std if std > 1e-9 else 1e-9)
        rows.append({
            "intent": intent, "latest_week": latest["week"], "latest_contacts": int(latest["contact_count"]),
            "latest_share": float(latest["share"]), "recent_mean_share": float(recent_mean),
            "historical_mean_share": float(mean), "z_score": float(z),
            "alert": bool(int(recent["contact_count"].sum()) >= min_contacts and z >= 2.0),
        })
    columns = [
        "intent", "latest_week", "latest_contacts", "latest_share", "recent_mean_share",
        "historical_mean_share", "z_score", "alert",
    ]
    result = pd.DataFrame(rows, columns=columns)
    if result.empty:
        return result
    return result.sort_values(["alert", "z_score"], ascending=[False, False])


def crm_outcomes(predictions: pd.DataFrame, crm: pd.DataFrame) -> pd.DataFrame:
    patient_contacts = predictions.groupby("pseudo_patient_id").agg(
        contacts=("contact_id", "size"),
        negative_share=("sentiment", lambda x: float((x == "negative").mean())),
        low_confidence_share=("prediction_confidence", lambda x: float((x < 0.72).mean())),
        clinical_escalations=("recommended_route", lambda x: int((x == "Clinical escalation").sum())),
    ).reset_index()
    return patient_contacts.merge(crm, on="pseudo_patient_id", how="left")
