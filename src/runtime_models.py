"""Portable runtime loading for the interactive chatbot demonstration.

The repository contains fitted scikit-learn ``joblib`` artefacts so the normal
path is fast. Pickled estimators are not guaranteed to remain portable across
Python, NumPy, SciPy, and scikit-learn combinations. Streamlit Community Cloud
may select a newer Python version than the environment used for the packaged
build. This module therefore provides an in-memory fallback trained from the
small labelled seed dataset included in the repository.

The fallback is used only for the interactive chatbot demonstration. The
reported benchmark metrics remain the offline build outputs stored in
``reports/`` and are not recomputed or replaced at dashboard start-up.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import FeatureUnion, Pipeline

SEED = 20260604


@dataclass(frozen=True)
class RuntimeModelBundle:
    """Models and transparent information about how they were obtained."""

    intent_model: Any
    sentiment_model: Any
    source: str
    fallback_used: bool
    detail: str


def _intent_pipeline() -> Pipeline:
    return Pipeline([
        (
            "features",
            FeatureUnion([
                (
                    "word",
                    TfidfVectorizer(
                        ngram_range=(1, 2), min_df=1, max_features=9000,
                        sublinear_tf=True,
                    ),
                ),
                (
                    "char",
                    TfidfVectorizer(
                        analyzer="char_wb", ngram_range=(3, 5), min_df=2,
                        max_features=9000, sublinear_tf=True,
                    ),
                ),
            ]),
        ),
        (
            "classifier",
            SGDClassifier(
                loss="log_loss", alpha=3e-5, max_iter=140, tol=1e-4,
                class_weight="balanced", random_state=SEED,
            ),
        ),
    ])


def _sentiment_pipeline() -> Pipeline:
    return Pipeline([
        (
            "features",
            FeatureUnion([
                (
                    "word",
                    TfidfVectorizer(
                        ngram_range=(1, 2), min_df=1, max_features=7000,
                        sublinear_tf=True,
                    ),
                ),
                (
                    "char",
                    TfidfVectorizer(
                        analyzer="char_wb", ngram_range=(3, 5), min_df=2,
                        max_features=6000, sublinear_tf=True,
                    ),
                ),
            ]),
        ),
        (
            "classifier",
            SGDClassifier(
                loss="log_loss", alpha=4e-5, max_iter=130, tol=1e-4,
                class_weight="balanced", random_state=SEED + 1,
            ),
        ),
    ])


def rebuild_runtime_models(data_dir: Path) -> RuntimeModelBundle:
    """Fit small in-memory models from the labelled seed CSV.

    This path is intentionally lightweight for Community Cloud. It does not
    modify the packaged artefacts and it does not change the reported offline
    benchmark metrics.
    """

    seed_path = Path(data_dir) / "annotation_seed.csv"
    seed = pd.read_csv(seed_path)
    required = {"clean_text", "intent", "sentiment"}
    missing = required.difference(seed.columns)
    if missing:
        raise ValueError(f"Fallback seed data is missing columns: {sorted(missing)}")

    train = seed.dropna(subset=["clean_text", "intent", "sentiment"]).copy()
    if train.empty:
        raise ValueError("Fallback seed data contains no usable labelled rows")

    intent_model = _intent_pipeline().fit(train["clean_text"].astype(str), train["intent"].astype(str))
    sentiment_model = _sentiment_pipeline().fit(train["clean_text"].astype(str), train["sentiment"].astype(str))
    return RuntimeModelBundle(
        intent_model=intent_model,
        sentiment_model=sentiment_model,
        source="in-memory fallback rebuilt from data/annotation_seed.csv",
        fallback_used=True,
        detail=f"Rebuilt portable chatbot models from {len(train):,} labelled seed contacts.",
    )


def load_or_rebuild_models(models_dir: Path, data_dir: Path) -> RuntimeModelBundle:
    """Load packaged models and fall back to a portable rebuild when needed."""

    try:
        intent_model = joblib.load(Path(models_dir) / "intent_classifier.joblib")
        sentiment_model = joblib.load(Path(models_dir) / "sentiment_classifier.joblib")
        return RuntimeModelBundle(
            intent_model=intent_model,
            sentiment_model=sentiment_model,
            source="packaged joblib models",
            fallback_used=False,
            detail="Loaded the fitted offline dashboard artefacts.",
        )
    except Exception as exc:  # noqa: BLE001 - compatibility failures vary by stack version
        rebuilt = rebuild_runtime_models(Path(data_dir))
        return RuntimeModelBundle(
            intent_model=rebuilt.intent_model,
            sentiment_model=rebuilt.sentiment_model,
            source=rebuilt.source,
            fallback_used=True,
            detail=(
                f"Packaged model load failed with {type(exc).__name__}; "
                f"{rebuilt.detail}"
            ),
        )
