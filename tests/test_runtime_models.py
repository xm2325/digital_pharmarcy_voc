from pathlib import Path

import joblib

from src.runtime_models import load_or_rebuild_models, rebuild_runtime_models

ROOT = Path(__file__).resolve().parents[1]


def test_runtime_seed_rebuild_predicts_probabilities():
    bundle = rebuild_runtime_models(ROOT / "data")
    text = ["my GP has not approved the prescription yet"]
    assert bundle.fallback_used is True
    assert bundle.intent_model.predict_proba(text).shape[0] == 1
    assert bundle.sentiment_model.predict_proba(text).shape[0] == 1


def test_runtime_loader_falls_back_when_joblib_is_incompatible(monkeypatch):
    def broken_load(*args, **kwargs):
        raise ModuleNotFoundError("simulated incompatible serialized module")

    monkeypatch.setattr(joblib, "load", broken_load)
    bundle = load_or_rebuild_models(ROOT / "models", ROOT / "data")
    assert bundle.fallback_used is True
    assert "ModuleNotFoundError" in bundle.detail
