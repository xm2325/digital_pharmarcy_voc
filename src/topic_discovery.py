"""Unknown-topic discovery using local text embeddings and c-TF-IDF-style summaries.

This default implementation is deliberately offline: TF-IDF vectors are projected
with TruncatedSVD and clustered with MiniBatchKMeans. An optional transformer-based
extension can replace the embedding function without changing the review workflow.
"""
from __future__ import annotations

from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import Normalizer

SEED = 20260604


def _top_terms_by_cluster(texts: pd.Series, clusters: np.ndarray, top_n: int = 8) -> dict[int, str]:
    grouped = pd.DataFrame({"text": texts.astype(str).to_numpy(), "cluster_id": clusters}).groupby("cluster_id")["text"].apply(lambda x: " ".join(x))
    vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), min_df=1, max_features=7000)
    matrix = vectorizer.fit_transform(grouped)
    terms = np.asarray(vectorizer.get_feature_names_out())
    result: dict[int, str] = {}
    for row_idx, cluster_id in enumerate(grouped.index):
        weights = matrix[row_idx].toarray().ravel()
        top = terms[np.argsort(weights)[::-1][:top_n]]
        result[int(cluster_id)] = ", ".join(top.tolist())
    return result


def discover_topics(
    predictions: pd.DataFrame,
    n_clusters: int = 6,
    confidence_threshold: float = 0.72,
    recent_days: int = 21,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Cluster review-worthy text and return cluster summaries and examples."""
    work = predictions.copy()
    work["timestamp"] = pd.to_datetime(work["timestamp"])
    recent_cutoff = work["timestamp"].max() - pd.Timedelta(days=recent_days)
    # Taxonomy evolution should start with contacts that the current catalogue does
    # not explain. If that queue is very small, add low-confidence cases so the
    # review page still remains useful on a small demonstration dataset.
    queue = work[work["predicted_intent"].eq("unclear_or_out_of_scope")].copy()
    queue["topic_queue_source"] = "out_of_scope_prediction"
    if len(queue) < 80:
        supplement = work[
            work["prediction_confidence"].lt(confidence_threshold)
            & ~work.index.isin(queue.index)
        ].copy()
        supplement["topic_queue_source"] = "low_confidence_supplement"
        queue = pd.concat([queue, supplement], ignore_index=False)
    queue = queue[queue["clean_text"].str.len().ge(8)].reset_index(drop=True)
    if len(queue) < 12:
        return pd.DataFrame(), pd.DataFrame()
    n_clusters = max(2, min(int(n_clusters), len(queue) // 5))
    vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), min_df=2, max_features=9000, sublinear_tf=True)
    sparse = vectorizer.fit_transform(queue["clean_text"])
    max_components = max(2, min(90, sparse.shape[0] - 1, sparse.shape[1] - 1))
    dense = TruncatedSVD(n_components=max_components, random_state=SEED).fit_transform(sparse)
    dense = Normalizer(copy=False).fit_transform(dense)
    clusterer = MiniBatchKMeans(n_clusters=n_clusters, random_state=SEED, n_init=10, batch_size=512)
    queue["cluster_id"] = clusterer.fit_predict(dense)
    top_terms = _top_terms_by_cluster(queue["clean_text"], queue["cluster_id"].to_numpy())

    all_recent_rate = float(work["timestamp"].ge(recent_cutoff).mean())
    rows = []
    example_rows = []
    for cluster_id, group in queue.groupby("cluster_id"):
        idx = group.index.to_numpy()
        centre = clusterer.cluster_centers_[int(cluster_id)].reshape(1, -1)
        scores = cosine_similarity(dense[idx], centre).ravel()
        representative_positions = idx[np.argsort(scores)[::-1][:5]]
        dominant_intent = str(group["predicted_intent"].mode().iloc[0])
        dominant_share = float(group["predicted_intent"].eq(dominant_intent).mean())
        recent_rate = float(group["timestamp"].ge(recent_cutoff).mean())
        out_of_scope_rate = float(group["predicted_intent"].eq("unclear_or_out_of_scope").mean())
        synthetic_novel_rate = float(group.get("synthetic_issue_pattern", pd.Series(index=group.index, dtype=str)).eq("novel_app_crash").mean())
        action = "Create or refine intent" if recent_rate >= min(0.95, all_recent_rate * 1.45) else "Review taxonomy coverage"
        rows.append({
            "cluster_id": int(cluster_id), "contacts": int(len(group)), "top_terms": top_terms[int(cluster_id)],
            "dominant_predicted_intent": dominant_intent, "dominant_intent_share": dominant_share,
            "out_of_scope_rate": out_of_scope_rate, "recent_contact_rate": recent_rate,
            "baseline_recent_contact_rate": all_recent_rate, "synthetic_novel_pattern_rate": synthetic_novel_rate,
            "recommended_review_action": action,
        })
        for rank, pos in enumerate(representative_positions, start=1):
            r = queue.loc[int(pos)]
            example_rows.append({
                "cluster_id": int(cluster_id), "rank": rank, "contact_id": r["contact_id"],
                "timestamp": r["timestamp"], "predicted_intent": r["predicted_intent"],
                "prediction_confidence": float(r["prediction_confidence"]), "raw_text": r["raw_text"],
            })
    clusters = pd.DataFrame(rows).sort_values(["recommended_review_action", "recent_contact_rate", "contacts"], ascending=[True, False, False])
    examples = pd.DataFrame(example_rows).sort_values(["cluster_id", "rank"])
    return clusters, examples


def write_topic_reports(predictions: pd.DataFrame, reports_dir: Path) -> None:
    reports_dir.mkdir(parents=True, exist_ok=True)
    clusters, examples = discover_topics(predictions)
    clusters.to_csv(reports_dir / "topic_discovery_clusters.csv", index=False)
    examples.to_csv(reports_dir / "topic_discovery_examples.csv", index=False)
