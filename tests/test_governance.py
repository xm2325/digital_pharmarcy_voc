import pandas as pd
from src.governance import governance_summary, redaction_counts
from src.preprocess import preprocess_contacts


def test_governance_summary_and_entity_counts():
    frame = pd.DataFrame({
        "contact_id": ["1", "2"],
        "pseudo_patient_id": ["P1", "P2"],
        "timestamp": pd.to_datetime(["2026-01-01", "2026-01-02"]),
        "channel": ["email", "live_chat"],
        "service_line": ["NHS prescription", "NHS prescription"],
        "raw_text": ["email sam@example.com", "plain issue"],
    })
    clean = preprocess_contacts(frame)
    summary = governance_summary(clean)
    counts = redaction_counts(clean)
    assert summary["contacts"] == 2
    assert summary["total_redactions"] == 1
    assert counts.loc[counts["entity_type"] == "email", "redactions"].iloc[0] == 1
