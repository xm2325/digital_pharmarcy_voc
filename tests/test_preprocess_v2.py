from src.preprocess import clean_text_detailed, preprocess_contacts
import pandas as pd


def test_expanded_rule_based_redaction_entities():
    text = (
        "My name is Alex Smith DOB 12/05/1975. Send order no P2U-123456 to 12 Oak Road, LS15 8GB. "
        "Email alex@example.com or call 07123 456 789 from Riverside Surgery. NHS number 1234567890"
    )
    cleaned, counts = clean_text_detailed(text)
    assert "[name]" in cleaned
    assert "[dob]" in cleaned
    assert "[order_ref]" in cleaned
    assert "[address]" in cleaned
    assert "[postcode]" in cleaned
    assert "[email]" in cleaned
    assert "[phone]" in cleaned
    assert "[gp_surgery]" in cleaned
    assert "[nhs_number]" in cleaned
    assert sum(counts.values()) >= 9


def test_preprocess_adds_governance_columns():
    frame = pd.DataFrame({"raw_text": ["Email me on jo@example.com", "plain text"]})
    out = preprocess_contacts(frame)
    assert {"clean_text", "redaction_email", "redaction_count", "token_count", "is_empty_text"}.issubset(out.columns)
    assert out.loc[0, "redaction_email"] == 1
    assert out.loc[1, "redaction_count"] == 0
