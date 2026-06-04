import pandas as pd
from src.topic_discovery import discover_topics


def test_unknown_topic_discovery_surfaces_recent_app_update_cluster():
    rows = []
    for idx in range(36):
        rows.append({
            "contact_id": f"N{idx}",
            "timestamp": pd.Timestamp("2026-05-20") + pd.Timedelta(hours=idx),
            "clean_text": "latest app update crashes when open prescription screen",
            "raw_text": "Latest app update crashes when I open my prescription screen",
            "predicted_intent": "unclear_or_out_of_scope",
            "prediction_confidence": 0.87,
            "synthetic_issue_pattern": "novel_app_crash",
        })
    for idx in range(90):
        rows.append({
            "contact_id": f"B{idx}",
            "timestamp": pd.Timestamp("2026-01-01") + pd.Timedelta(days=idx % 30),
            "clean_text": "please contact me about issue site help",
            "raw_text": "Please contact me about an issue on the site",
            "predicted_intent": "unclear_or_out_of_scope",
            "prediction_confidence": 0.75,
            "synthetic_issue_pattern": "standard_known_intent",
        })
    clusters, examples = discover_topics(pd.DataFrame(rows), n_clusters=3)
    assert not clusters.empty
    strongest = clusters.sort_values("synthetic_novel_pattern_rate", ascending=False).iloc[0]
    assert strongest["synthetic_novel_pattern_rate"] >= 0.9
    assert "app" in strongest["top_terms"] or "update" in strongest["top_terms"]
    assert strongest["recommended_review_action"] == "Create or refine intent"
    assert not examples.empty
