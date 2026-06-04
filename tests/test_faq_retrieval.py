import pandas as pd
from src.faq_retrieval import FAQRetriever, evaluate_retriever


def sample_faq():
    return pd.DataFrame([
        {"question": "How do I track my order?", "answer": "Use tracking", "mapped_intent": "delivery_tracking"},
        {"question": "How do I request a repeat prescription?", "answer": "Use the app", "mapped_intent": "repeat_prescription_request"},
        {"question": "How do I change my address?", "answer": "Update account", "mapped_intent": "change_delivery_address"},
    ])


def test_tfidf_retriever_returns_tracking_match():
    retriever = FAQRetriever(sample_faq(), mode="tfidf", answerability_threshold=0.1)
    result = retriever.search("Where is my order tracking?", top_k=1)[0]
    assert result.mapped_intent == "delivery_tracking"
    assert retriever.is_answerable("Where is my order tracking?")


def test_retriever_metrics_are_bounded():
    evaluation = pd.DataFrame([
        {"query": "track order", "expected_intent": "delivery_tracking", "expected_answerable": True},
        {"query": "repeat prescription", "expected_intent": "repeat_prescription_request", "expected_answerable": True},
    ])
    metrics = evaluate_retriever(FAQRetriever(sample_faq(), mode="lsa", answerability_threshold=0.05), evaluation)
    assert 0 <= metrics["recall_at_1"] <= 1
    assert 0 <= metrics["mean_reciprocal_rank"] <= 1
