import pandas as pd
from src.analytics import emerging_issues

def test_emerging_issue_output_columns():
    rows = []
    for week, count in zip(pd.date_range("2026-01-05", periods=7, freq="W-MON"), [5, 5, 5, 5, 5, 35, 50]):
        rows.extend({"timestamp": week, "predicted_intent": "delivery_delayed"} for _ in range(count))
        rows.extend({"timestamp": week, "predicted_intent": "other"} for _ in range(100))
    out = emerging_issues(pd.DataFrame(rows), min_contacts=15)
    assert {"intent", "z_score", "alert"}.issubset(out.columns)
    assert out.iloc[0]["intent"] == "delivery_delayed"


def test_emerging_issue_insufficient_history_returns_empty_frame():
    frame = pd.DataFrame([
        {"timestamp": pd.Timestamp("2026-01-05"), "predicted_intent": "delivery_delayed"},
        {"timestamp": pd.Timestamp("2026-01-12"), "predicted_intent": "delivery_delayed"},
    ])
    out = emerging_issues(frame)
    assert out.empty
    assert {"intent", "z_score", "alert"}.issubset(out.columns)
