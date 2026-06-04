import pandas as pd
from src.annotation import build_annotation_queue


def test_annotation_queue_contains_review_fields():
    frame = pd.DataFrame([{
        "contact_id": "C1", "timestamp": pd.Timestamp("2026-01-01"), "channel": "email",
        "raw_text": "help", "clean_text": "help", "predicted_intent": "unclear_or_out_of_scope",
        "prediction_confidence": 0.4, "predicted_sentiment": "neutral", "sentiment_confidence": 0.6,
        "recommended_route": "Human adviser",
    }])
    out = build_annotation_queue(frame)
    assert {"human_intent_label", "human_sentiment_label", "review_status", "taxonomy_version"}.issubset(out.columns)
    assert out.loc[0, "review_status"] == "unreviewed"
