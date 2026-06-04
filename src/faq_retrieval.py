"""FAQ semantic-search helper with offline and optional transformer modes.

Modes:
- ``tfidf``: lexical baseline.
- ``lsa``: local dense latent-semantic embeddings from TF-IDF + TruncatedSVD.
- ``sentence_transformer``: optional Sentence Transformers embeddings. This needs
  the optional dependency and a downloaded model.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import Normalizer


@dataclass
class RetrievalResult:
    question: str
    answer: str
    mapped_intent: str
    score: float


class FAQRetriever:
    def __init__(
        self,
        faq: pd.DataFrame,
        mode: str = "lsa",
        model_name: str = "all-MiniLM-L6-v2",
        answerability_threshold: float = 0.22,
        use_sentence_transformer: bool | None = None,
    ):
        if use_sentence_transformer is not None:
            mode = "sentence_transformer" if use_sentence_transformer else "tfidf"
        self.faq = faq.reset_index(drop=True).copy()
        self.requested_mode = mode
        self.mode = mode
        self.model_name = model_name
        self.answerability_threshold = float(answerability_threshold)
        self.vectorizer = None
        self.reducer = None
        self.model = None
        self.matrix = None
        questions = self.faq["question"].astype(str).tolist()
        if mode == "sentence_transformer":
            try:
                from sentence_transformers import SentenceTransformer
                self.model = SentenceTransformer(model_name)
                self.matrix = self.model.encode(questions, normalize_embeddings=True)
                self.mode = f"sentence-transformer: {model_name}"
            except Exception:
                self.mode = "lsa fallback (sentence-transformer unavailable)"
                self.model = None
                self._fit_lsa(questions)
        elif mode == "lsa":
            self._fit_lsa(questions)
            self.mode = "local latent-semantic embeddings (TF-IDF + SVD)"
        elif mode == "tfidf":
            self.vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=1)
            self.matrix = self.vectorizer.fit_transform(questions)
            self.mode = "TF-IDF lexical baseline"
        else:
            raise ValueError(f"Unsupported retrieval mode: {mode}")

    def _fit_lsa(self, questions: list[str]) -> None:
        self.vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=1, stop_words="english")
        sparse = self.vectorizer.fit_transform(questions)
        n_components = max(2, min(6, sparse.shape[0] - 1, sparse.shape[1] - 1))
        self.reducer = TruncatedSVD(n_components=n_components, random_state=20260604)
        self.matrix = Normalizer(copy=False).fit_transform(self.reducer.fit_transform(sparse))

    def _scores(self, query: str) -> np.ndarray:
        if self.model is not None:
            q = self.model.encode([query], normalize_embeddings=True)
            return np.asarray(q @ self.matrix.T).ravel()
        if self.reducer is not None:
            sparse = self.vectorizer.transform([query])
            q = Normalizer(copy=False).fit_transform(self.reducer.transform(sparse))
            return cosine_similarity(q, self.matrix).ravel()
        q = self.vectorizer.transform([query])
        return cosine_similarity(q, self.matrix).ravel()

    def search(self, query: str, top_k: int = 3) -> List[RetrievalResult]:
        scores = self._scores(query)
        indices = np.argsort(scores)[::-1][:top_k]
        return [
            RetrievalResult(
                question=str(self.faq.loc[i, "question"]), answer=str(self.faq.loc[i, "answer"]),
                mapped_intent=str(self.faq.loc[i, "mapped_intent"]), score=float(scores[i]),
            )
            for i in indices
        ]

    def is_answerable(self, query: str) -> bool:
        results = self.search(query, top_k=1)
        return bool(results and results[0].score >= self.answerability_threshold)


def evaluate_retriever(retriever: FAQRetriever, evaluation: pd.DataFrame) -> dict[str, float | int | str]:
    reciprocal_ranks = []
    recall_at_1 = []
    recall_at_3 = []
    answerable_accuracy = []
    for row in evaluation.itertuples(index=False):
        results = retriever.search(str(row.query), top_k=min(3, len(retriever.faq)))
        intents = [r.mapped_intent for r in results]
        expected = str(row.expected_intent)
        reciprocal_ranks.append(1.0 / (intents.index(expected) + 1) if expected in intents else 0.0)
        recall_at_1.append(float(bool(intents) and intents[0] == expected))
        recall_at_3.append(float(expected in intents))
        expected_answerable = bool(row.expected_answerable)
        answerable_accuracy.append(float(retriever.is_answerable(str(row.query)) == expected_answerable))
    return {
        "mode": retriever.mode, "queries": int(len(evaluation)),
        "recall_at_1": float(np.mean(recall_at_1)), "recall_at_3": float(np.mean(recall_at_3)),
        "mean_reciprocal_rank": float(np.mean(reciprocal_ranks)),
        "answerability_accuracy": float(np.mean(answerable_accuracy)),
    }
