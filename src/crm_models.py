"""Synthetic CRM linkage and interpretable outcome-driver models."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, average_precision_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

SEED = 20260604

NUMERIC_FEATURES = [
    "contacts", "negative_share", "repeat_flag_share", "low_confidence_share",
    "clinical_escalations", "human_adviser_share", "proactive_message_share",
]
CATEGORICAL_FEATURES = ["dominant_channel", "dominant_predicted_intent", "crm_segment"]


def build_patient_features(predictions: pd.DataFrame, crm: pd.DataFrame) -> pd.DataFrame:
    work = predictions.copy()
    grouped = work.groupby("pseudo_patient_id")
    features = grouped.agg(
        contacts=("contact_id", "size"),
        negative_share=("predicted_sentiment", lambda x: float((x == "negative").mean())),
        repeat_flag_share=("repeat_contact_14d", "mean"),
        low_confidence_share=("prediction_confidence", lambda x: float((x < 0.72).mean())),
        clinical_escalations=("recommended_route", lambda x: int((x == "Clinical escalation").sum())),
        human_adviser_share=("recommended_route", lambda x: float((x == "Human adviser").mean())),
        proactive_message_share=("recommended_route", lambda x: float((x == "Proactive message").mean())),
    ).reset_index()
    dominant_channel = grouped["channel"].agg(lambda x: str(x.mode().iloc[0])).rename("dominant_channel").reset_index()
    dominant_intent = grouped["predicted_intent"].agg(lambda x: str(x.mode().iloc[0])).rename("dominant_predicted_intent").reset_index()
    features = features.merge(dominant_channel, on="pseudo_patient_id").merge(dominant_intent, on="pseudo_patient_id")
    out = features.merge(crm, on="pseudo_patient_id", how="inner")
    out["low_csat"] = out["csat_score"].le(2)
    out["not_retained_90d"] = ~out["retained_90d"].astype(bool)
    return out


def _pipeline() -> Pipeline:
    transformer = ColumnTransformer([
        ("numeric", Pipeline([("imputer", SimpleImputer(strategy="median")), ("scale", StandardScaler())]), NUMERIC_FEATURES),
        ("categorical", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("onehot", OneHotEncoder(handle_unknown="ignore"))]), CATEGORICAL_FEATURES),
    ])
    return Pipeline([("features", transformer), ("classifier", LogisticRegression(max_iter=800, class_weight="balanced", random_state=SEED))])


def _feature_names(model: Pipeline) -> list[str]:
    transformer = model.named_steps["features"]
    return list(transformer.get_feature_names_out())


def _fit_one(frame: pd.DataFrame, outcome: str) -> Tuple[Pipeline, Dict[str, float | int], pd.DataFrame]:
    x = frame[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    y = frame[outcome].astype(int)
    train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=0.28, stratify=y, random_state=SEED)
    model = _pipeline()
    model.fit(train_x, train_y)
    proba = model.predict_proba(test_x)[:, 1]
    pred = proba >= 0.5
    metrics: Dict[str, float | int] = {
        "rows": int(len(frame)), "positive_rate": float(y.mean()), "test_rows": int(len(test_y)),
        "roc_auc": float(roc_auc_score(test_y, proba)),
        "average_precision": float(average_precision_score(test_y, proba)),
        "accuracy_at_0_5": float(accuracy_score(test_y, pred)),
    }
    coefs = pd.DataFrame({"feature": _feature_names(model), "coefficient": model.named_steps["classifier"].coef_[0]})
    coefs["outcome"] = outcome
    coefs["absolute_coefficient"] = coefs["coefficient"].abs()
    return model, metrics, coefs.sort_values("absolute_coefficient", ascending=False)


def train_crm_models(predictions: pd.DataFrame, crm: pd.DataFrame, models_dir: Path, reports_dir: Path) -> pd.DataFrame:
    models_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)
    linked = build_patient_features(predictions, crm)
    all_metrics: Dict[str, Dict[str, float | int]] = {}
    all_coefs = []
    for outcome in ["low_csat", "not_retained_90d"]:
        model, metrics, coefs = _fit_one(linked, outcome)
        joblib.dump(model, models_dir / f"crm_{outcome}_classifier.joblib")
        all_metrics[outcome] = metrics
        all_coefs.append(coefs)
    pd.concat(all_coefs, ignore_index=True).to_csv(reports_dir / "crm_driver_coefficients.csv", index=False)
    linked.to_csv(reports_dir / "crm_linked_patient_features.csv", index=False)
    with open(reports_dir / "crm_driver_metrics.json", "w", encoding="utf-8") as f:
        json.dump(all_metrics, f, indent=2)
    return linked
