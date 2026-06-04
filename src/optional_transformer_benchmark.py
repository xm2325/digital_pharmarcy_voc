"""Optional Sentence Transformers intent benchmark.

Run only in a connected environment after installing requirements-optional-transformers.txt.
The script downloads or loads the specified Sentence Transformers model, encodes the
cleaned contacts, trains a linear classifier, and writes an honest benchmark row.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split

from .preprocess import preprocess_contacts

SEED = 20260604


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=Path("data/synthetic_contacts_raw.csv"))
    parser.add_argument("--model-name", default="all-MiniLM-L6-v2")
    parser.add_argument("--reports-dir", type=Path, default=Path("reports"))
    args = parser.parse_args()
    args.reports_dir.mkdir(parents=True, exist_ok=True)
    df = preprocess_contacts(pd.read_csv(args.input, parse_dates=["timestamp"]))
    train, test = train_test_split(df, test_size=0.24, stratify=df["intent"], random_state=SEED)
    encoder = SentenceTransformer(args.model_name)
    train_x = encoder.encode(train["clean_text"].tolist(), normalize_embeddings=True, show_progress_bar=True)
    test_x = encoder.encode(test["clean_text"].tolist(), normalize_embeddings=True, show_progress_bar=True)
    clf = LogisticRegression(max_iter=900, class_weight="balanced", random_state=SEED)
    clf.fit(train_x, train["intent"])
    pred = clf.predict(test_x)
    result = pd.DataFrame([{
        "task": "intent_classification", "model": f"sentence_transformer::{args.model_name}", "split": "random_stratified",
        "rows": len(test), "accuracy": accuracy_score(test["intent"], pred),
        "macro_f1": f1_score(test["intent"], pred, average="macro"),
        "weighted_f1": f1_score(test["intent"], pred, average="weighted"),
    }])
    result.to_csv(args.reports_dir / "optional_sentence_transformer_benchmark.csv", index=False)
    print(result.to_string(index=False))


if __name__ == "__main__":
    main()
