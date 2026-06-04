from src.preprocess import clean_text

def test_redacts_email_and_phone():
    cleaned, n = clean_text("Call me on 07123 456 789 or email sam@example.com pls")
    assert "[phone]" in cleaned
    assert "[email]" in cleaned
    assert "please" in cleaned
    assert n == 2
