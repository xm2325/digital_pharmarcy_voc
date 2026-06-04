"""Annotation-review queue and taxonomy-version helpers."""
from __future__ import annotations

from pathlib import Path

import pandas as pd


def build_annotation_queue(predictions: pd.DataFrame, taxonomy_version: str = "v2.0", limit: int = 700) -> pd.DataFrame:
    work = predictions.copy()
    work["review_priority"] = (
        (1 - work["prediction_confidence"]) * 0.55
        + work["predicted_intent"].eq("unclear_or_out_of_scope").astype(float) * 0.30
        + work["recommended_route"].eq("Clinical escalation").astype(float) * 0.15
    )
    columns = [
        "contact_id", "timestamp", "channel", "raw_text", "clean_text", "predicted_intent",
        "prediction_confidence", "predicted_sentiment", "sentiment_confidence", "recommended_route",
        "review_priority",
    ]
    queue = work.sort_values("review_priority", ascending=False).head(limit)[columns].copy()
    queue["human_intent_label"] = queue["predicted_intent"]
    queue["human_sentiment_label"] = queue["predicted_sentiment"]
    queue["review_status"] = "unreviewed"
    queue["reviewer_id"] = ""
    queue["review_notes"] = ""
    queue["taxonomy_version"] = taxonomy_version
    return queue


def write_annotation_queue(predictions: pd.DataFrame, data_dir: Path) -> None:
    data_dir.mkdir(parents=True, exist_ok=True)
    build_annotation_queue(predictions).to_csv(data_dir / "annotation_review_queue.csv", index=False)
