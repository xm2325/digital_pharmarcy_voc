"""Train V2 NLP benchmarks and write dashboard-ready reports.

The packaged workflow runs without external model downloads. It compares lexical
and local dense-embedding classifiers, trains an independent sentiment model, writes
calibration and channel-level checks, builds a taxonomy-review queue, and trains
interpretable synthetic CRM outcome-driver models.
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys

# Keep laptop rebuilds predictable and avoid BLAS thread oversubscription.
_THREADS = os.environ.get("VOC_NUM_THREADS", "1")
for _name in ["OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"]:
    os.environ[_name] = _THREADS
from pathlib import Path
from typing import Callable, Dict, Tuple

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import Normalizer

from .analytics import emerging_issues
from .annotation import write_annotation_queue
from .crm_models import train_crm_models
from .faq_retrieval import FAQRetriever, evaluate_retriever
from .governance import write_governance_reports
from .preprocess import preprocess_contacts
from .routing import add_routes, deflection_scenario
from .topic_discovery import write_topic_reports

SEED = 20260604


def noisy_intent_labels(train: pd.DataFrame, rate: float = 0.045) -> np.ndarray:
    """Add synthetic annotation noise to training labels only."""
    rng = np.random.default_rng(SEED)
    labels = train["intent"].astype(str).to_numpy(copy=True)
    mask = rng.random(len(labels)) < rate
    options = np.unique(labels)
    labels[mask] = rng.choice(options, size=int(mask.sum()))
    return labels


def word_tfidf_pipeline() -> Pipeline:
    return Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=18000, sublinear_tf=True)),
        ("classifier", SGDClassifier(loss="log_loss", alpha=2e-5, max_iter=180, tol=1e-4, class_weight="balanced", random_state=SEED)),
    ])


def robust_char_word_pipeline() -> Pipeline:
    features = FeatureUnion([
        ("word", TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=15000, sublinear_tf=True)),
        ("char", TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=3, max_features=15000, sublinear_tf=True)),
    ])
    return Pipeline([
        ("features", features),
        ("classifier", SGDClassifier(loss="log_loss", alpha=2.5e-5, max_iter=190, tol=1e-4, class_weight="balanced", random_state=SEED)),
    ])


def local_dense_embedding_pipeline() -> Pipeline:
    """Offline dense embedding baseline: TF-IDF -> SVD -> normalisation -> classifier."""
    return Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=14000, sublinear_tf=True, stop_words="english")),
        ("svd", TruncatedSVD(n_components=48, random_state=SEED)),
        ("normalise", Normalizer(copy=False)),
        ("classifier", SGDClassifier(loss="log_loss", alpha=7e-5, max_iter=170, tol=1e-4, class_weight="balanced", random_state=SEED)),
    ])


def sentiment_pipeline() -> Pipeline:
    features = FeatureUnion([
        ("word", TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=11000, sublinear_tf=True)),
        ("char", TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=3, max_features=9000, sublinear_tf=True)),
    ])
    return Pipeline([
        ("features", features),
        ("classifier", SGDClassifier(loss="log_loss", alpha=3e-5, max_iter=170, tol=1e-4, class_weight="balanced", random_state=SEED + 3)),
    ])


def _fit_intent_model(factory: Callable[[], Pipeline], train: pd.DataFrame) -> Pipeline:
    model = factory()
    model.fit(train["clean_text"], noisy_intent_labels(train))
    return model


def split_frames(df: pd.DataFrame, split_mode: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if split_mode == "random_stratified":
        return train_test_split(df, test_size=0.24, random_state=SEED, stratify=df["intent"])
    if split_mode == "temporal_holdout":
        ordered = df.sort_values("timestamp")
        cut = int(len(ordered) * 0.76)
        return ordered.iloc[:cut].copy(), ordered.iloc[cut:].copy()
    raise ValueError(f"Unsupported split mode: {split_mode}")


def basic_metrics(y_true: pd.Series, y_pred: np.ndarray) -> Dict[str, float | int]:
    return {
        "rows": int(len(y_true)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted")),
    }


def benchmark_intent_models(df: pd.DataFrame) -> Tuple[pd.DataFrame, Pipeline, pd.DataFrame, pd.DataFrame]:
    factories: Dict[str, Callable[[], Pipeline]] = {
        "tfidf_word_sgd": word_tfidf_pipeline,
        "tfidf_char_word_sgd": robust_char_word_pipeline,
        "local_dense_embedding_lsa": local_dense_embedding_pipeline,
    }
    rows = []
    # Keep repository rebuild time practical while evaluating all benchmark variants.
    # The selected model is then fitted again on the full random training split.
    benchmark_df = df.sample(min(8000, len(df)), random_state=SEED).copy()
    random_train, random_test = split_frames(df, "random_stratified")
    for split_mode in ["random_stratified", "temporal_holdout"]:
        train, test = split_frames(benchmark_df, split_mode)
        for model_name, factory in factories.items():
            model = _fit_intent_model(factory, train)
            pred = model.predict(test["clean_text"])
            row = {"task": "intent_classification", "model": model_name, "split": split_mode, "benchmark_train_rows": int(len(train)), **basic_metrics(test["intent"], pred)}
            rows.append(row)
    benchmark = pd.DataFrame(rows).sort_values(["split", "macro_f1"], ascending=[True, False])
    # Use the robust sparse model for the packaged dashboard. It handles spelling noise,
    # rebuilds quickly on a laptop, and avoids forcing an SVD fit at deployment time.
    packaged_name = "tfidf_char_word_sgd"
    best_model = _fit_intent_model(factories[packaged_name], random_train)
    return benchmark, best_model, random_train, random_test


def probability_diagnostics(y_true: pd.Series, pred: np.ndarray, proba: np.ndarray, labels: list[str]) -> Tuple[dict, pd.DataFrame]:
    confidence = proba.max(axis=1)
    correct = (np.asarray(y_true.astype(str)) == np.asarray(pred).astype(str)).astype(float)
    bins = np.linspace(0.0, 1.0, 11)
    bucket = np.clip(np.digitize(confidence, bins, right=True) - 1, 0, 9)
    rows = []
    ece = 0.0
    for b in range(10):
        mask = bucket == b
        if not mask.any():
            continue
        mean_conf = float(confidence[mask].mean())
        accuracy = float(correct[mask].mean())
        share = float(mask.mean())
        ece += share * abs(mean_conf - accuracy)
        rows.append({
            "bin_lower": float(bins[b]), "bin_upper": float(bins[b + 1]), "contacts": int(mask.sum()),
            "mean_confidence": mean_conf, "observed_accuracy": accuracy, "share": share,
        })
    label_to_index = {label: idx for idx, label in enumerate(labels)}
    y_onehot = np.zeros_like(proba)
    for row_idx, label in enumerate(y_true.astype(str)):
        if label in label_to_index:
            y_onehot[row_idx, label_to_index[label]] = 1.0
    multiclass_brier = float(np.mean(np.sum((proba - y_onehot) ** 2, axis=1)))
    return {"expected_calibration_error": float(ece), "multiclass_brier_score": multiclass_brier}, pd.DataFrame(rows)


def channel_metrics(predictions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for channel, group in predictions.groupby("channel"):
        rows.append({"channel": channel, **basic_metrics(group["intent"], group["predicted_intent"].to_numpy()), "mean_confidence": float(group["prediction_confidence"].mean())})
    return pd.DataFrame(rows).sort_values("macro_f1", ascending=False)


def coverage_curve(pred: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for threshold in np.round(np.arange(0.40, 0.96, 0.04), 2):
        covered = pred[pred["prediction_confidence"] >= threshold]
        rows.append({
            "confidence_threshold": threshold, "coverage": len(covered) / len(pred),
            "accuracy_on_covered": covered["prediction_correct"].mean() if len(covered) else np.nan,
            "contacts_covered": len(covered),
        })
    return pd.DataFrame(rows)


def evaluate_selected_intent_model(model: Pipeline, test: pd.DataFrame, selected_name: str) -> Tuple[dict, pd.DataFrame, pd.DataFrame]:
    pred = model.predict(test["clean_text"])
    proba = model.predict_proba(test["clean_text"])
    conf = proba.max(axis=1)
    labels = list(model.classes_)
    out = test.copy()
    out["predicted_intent"] = pred
    out["prediction_confidence"] = conf
    out["prediction_correct"] = out["intent"].astype(str) == out["predicted_intent"].astype(str)
    out = add_routes(out, threshold=0.72)
    urgent_mask = out["safety_sensitive"].astype(bool)
    urgent_recall = float((out.loc[urgent_mask, "recommended_route"] == "Clinical escalation").mean()) if urgent_mask.any() else 0.0
    diag, reliability = probability_diagnostics(test["intent"], pred, proba, labels)
    metrics = {
        "selected_intent_model": selected_name,
        **basic_metrics(test["intent"], pred),
        "urgent_route_recall": urgent_recall, "test_rows": int(len(test)),
        "mean_confidence": float(conf.mean()), "low_confidence_rate_at_0_72": float((conf < 0.72).mean()),
        **diag,
        "labels": labels,
        "classification_report": classification_report(test["intent"], pred, labels=labels, output_dict=True, zero_division=0),
        "confusion_matrix": confusion_matrix(test["intent"], pred, labels=labels).tolist(),
    }
    cm = pd.DataFrame(metrics["confusion_matrix"], index=labels, columns=labels)
    return metrics, out, reliability


def train_sentiment_model(train: pd.DataFrame, test: pd.DataFrame) -> Tuple[Pipeline, dict, pd.DataFrame]:
    model = sentiment_pipeline()
    model.fit(train["clean_text"], train["sentiment"].astype(str))
    pred = model.predict(test["clean_text"])
    proba = model.predict_proba(test["clean_text"])
    out = test.copy()
    out["predicted_sentiment"] = pred
    out["sentiment_confidence"] = proba.max(axis=1)
    metrics = {
        **basic_metrics(test["sentiment"], pred),
        "mean_confidence": float(proba.max(axis=1).mean()),
        "labels": list(model.classes_),
        "classification_report": classification_report(test["sentiment"], pred, output_dict=True, zero_division=0),
    }
    return model, metrics, out[["contact_id", "predicted_sentiment", "sentiment_confidence"]]


def write_optional_model_status(reports_dir: Path) -> None:
    status = {
        "packaged_default": "offline local models executed",
        "sentence_transformer_classifier": "not executed in packaged build; run src.optional_transformer_benchmark after downloading a model",
        "huggingface_fine_tuned_classifier": "not executed in packaged build; run src.optional_hf_finetune after downloading a model",
        "reason": "The public repository remains runnable without external model downloads. Optional scripts are included for a connected environment.",
    }
    with open(reports_dir / "optional_transformer_status.json", "w", encoding="utf-8") as f:
        json.dump(status, f, indent=2)


def write_faq_metrics(data_dir: Path, reports_dir: Path) -> None:
    faq = pd.read_csv(data_dir / "faq_knowledge_base.csv")
    faq_eval = pd.read_csv(data_dir / "faq_retrieval_eval.csv")
    rows = []
    for mode in ["tfidf", "lsa"]:
        rows.append(evaluate_retriever(FAQRetriever(faq, mode=mode), faq_eval))
    pd.DataFrame(rows).to_csv(reports_dir / "faq_retrieval_metrics.csv", index=False)


def write_model_metadata(reports_dir: Path, selected_model: str) -> None:
    metadata = {
        "project_version": "2.0.0", "selected_intent_model": selected_model,
        "python": sys.version.split()[0], "platform": platform.platform(), "scikit_learn": sklearn.__version__,
        "random_seed": SEED, "build_mode": "offline_reproducible_default",
    }
    with open(reports_dir / "model_metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)


def _load_preprocessed(input_path: Path) -> pd.DataFrame:
    raw = pd.read_csv(input_path, parse_dates=["timestamp"])
    return preprocess_contacts(raw)


def run_sentiment_stage(args: argparse.Namespace) -> None:
    df = _load_preprocessed(args.input)
    train, test = split_frames(df, "random_stratified")
    sentiment_model, sentiment_metrics, sentiment_pred = train_sentiment_model(train, test)
    joblib.dump(sentiment_model, args.models_dir / "sentiment_classifier.joblib")
    sentiment_pred.to_csv(args.data_dir / "sentiment_prediction_sample.csv", index=False)
    with open(args.reports_dir / "sentiment_metrics.json", "w", encoding="utf-8") as f:
        json.dump(sentiment_metrics, f, indent=2)
    print(json.dumps({"stage": "sentiment", "macro_f1": sentiment_metrics["macro_f1"], "rows": sentiment_metrics["rows"]}, indent=2))


def run_intent_stage(args: argparse.Namespace) -> None:
    df = _load_preprocessed(args.input)
    write_governance_reports(df, args.reports_dir)
    benchmark, intent_model, train, test = benchmark_intent_models(df)
    selected_name = "tfidf_char_word_sgd"
    metrics, pred, reliability = evaluate_selected_intent_model(intent_model, test, selected_name)
    joblib.dump(intent_model, args.models_dir / "intent_classifier.joblib")
    pred.to_csv(args.data_dir / "prediction_sample.csv", index=False)
    train.sample(min(3000, len(train)), random_state=SEED).to_csv(args.data_dir / "annotation_seed.csv", index=False)
    benchmark.to_csv(args.reports_dir / "intent_model_benchmark.csv", index=False)
    channel_metrics(pred).to_csv(args.reports_dir / "intent_metrics_by_channel.csv", index=False)
    reliability.to_csv(args.reports_dir / "intent_reliability_diagram.csv", index=False)
    coverage_curve(pred).to_csv(args.reports_dir / "coverage_accuracy_curve.csv", index=False)
    pd.DataFrame(metrics["confusion_matrix"], index=metrics["labels"], columns=metrics["labels"]).to_csv(args.reports_dir / "intent_confusion_matrix.csv")
    emerging_issues(pred).to_csv(args.reports_dir / "emerging_issues.csv", index=False)
    with open(args.reports_dir / "model_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print(json.dumps({"stage": "intent", **{k: v for k, v in metrics.items() if k not in {"classification_report", "confusion_matrix", "labels"}}}, indent=2))


def run_finalize_stage(args: argparse.Namespace) -> None:
    pred = pd.read_csv(args.data_dir / "prediction_sample.csv", parse_dates=["timestamp"])
    sentiment_pred = pd.read_csv(args.data_dir / "sentiment_prediction_sample.csv")
    pred = pred.drop(columns=["predicted_sentiment", "sentiment_confidence"], errors="ignore").merge(sentiment_pred, on="contact_id", how="left")
    pred.to_csv(args.data_dir / "prediction_sample.csv", index=False)
    emerging_issues(pred).to_csv(args.reports_dir / "emerging_issues.csv", index=False)
    write_topic_reports(pred, args.reports_dir)
    write_annotation_queue(pred, args.data_dir)
    write_faq_metrics(args.data_dir, args.reports_dir)
    crm = pd.read_csv(args.data_dir / "synthetic_crm.csv")
    train_crm_models(pred, crm, args.models_dir, args.reports_dir)
    with open(args.reports_dir / "deflection_scenario_default.json", "w", encoding="utf-8") as f:
        json.dump(deflection_scenario(pred), f, indent=2)
    write_optional_model_status(args.reports_dir)
    write_model_metadata(args.reports_dir, "tfidf_char_word_sgd")
    print(json.dumps({"stage": "finalize", "prediction_rows": len(pred)}, indent=2))


def _stage_command(stage: str, args: argparse.Namespace) -> list[str]:
    return [
        sys.executable, "-m", "src.train_models", "--stage", stage,
        "--input", str(args.input), "--data-dir", str(args.data_dir),
        "--models-dir", str(args.models_dir), "--reports-dir", str(args.reports_dir),
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=Path("data/synthetic_contacts_raw.csv"))
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--models-dir", type=Path, default=Path("models"))
    parser.add_argument("--reports-dir", type=Path, default=Path("reports"))
    parser.add_argument("--stage", choices=["all", "sentiment", "intent", "finalize"], default="all")
    args = parser.parse_args()
    for path in [args.data_dir, args.models_dir, args.reports_dir]:
        path.mkdir(parents=True, exist_ok=True)
    if args.stage == "sentiment":
        run_sentiment_stage(args)
        sys.stdout.flush(); sys.stderr.flush(); os._exit(0)
    elif args.stage == "intent":
        run_intent_stage(args)
        sys.stdout.flush(); sys.stderr.flush(); os._exit(0)
    elif args.stage == "finalize":
        run_finalize_stage(args)
        sys.stdout.flush(); sys.stderr.flush(); os._exit(0)
    else:
        # Replace this Python process with a lightweight shell orchestrator. Each
        # numerical stage then runs in a fresh process, which avoids retaining a
        # second imported scientific-Python stack in memory during laptop rebuilds.
        script = Path(__file__).resolve().parents[1] / "scripts" / "run_training_stages.sh"
        os.execvpe("bash", ["bash", str(script), str(args.input), str(args.data_dir), str(args.models_dir), str(args.reports_dir)], os.environ.copy())


if __name__ == "__main__":
    main()
