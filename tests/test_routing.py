from src.routing import assign_route

def test_safety_language_overrides_classification():
    route = assign_route("delivery_tracking", 0.99, "I have run out of tablets and need help today", threshold=0.72)
    assert route == "Clinical escalation"

def test_low_confidence_goes_to_human():
    route = assign_route("delivery_tracking", 0.45, "where is my parcel", threshold=0.72)
    assert route == "Human adviser"

def test_safe_high_confidence_uses_self_service():
    route = assign_route("delivery_tracking", 0.94, "where is my parcel", threshold=0.72)
    assert route == "Chatbot or FAQ self-service"
