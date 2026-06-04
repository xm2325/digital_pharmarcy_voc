"""Optional Hugging Face sequence-classification fine-tuning entry point.

This script is intentionally separate from the offline build because it downloads a
pre-trained checkpoint and needs additional packages. It is a runnable extension for
a connected environment rather than a claimed packaged benchmark result.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from datasets import Dataset
from sklearn.model_selection import train_test_split
from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments

from .preprocess import preprocess_contacts

SEED = 20260604


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=Path("data/synthetic_contacts_raw.csv"))
    parser.add_argument("--checkpoint", default="distilbert-base-uncased")
    parser.add_argument("--output-dir", type=Path, default=Path("models/hf_intent_classifier"))
    parser.add_argument("--epochs", type=float, default=1.0)
    args = parser.parse_args()
    df = preprocess_contacts(pd.read_csv(args.input, parse_dates=["timestamp"]))
    labels = sorted(df["intent"].astype(str).unique())
    label2id = {label: idx for idx, label in enumerate(labels)}
    id2label = {idx: label for label, idx in label2id.items()}
    train, test = train_test_split(df[["clean_text", "intent"]], test_size=0.24, stratify=df["intent"], random_state=SEED)
    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint)
    model = AutoModelForSequenceClassification.from_pretrained(args.checkpoint, num_labels=len(labels), label2id=label2id, id2label=id2label)

    def make_dataset(frame: pd.DataFrame) -> Dataset:
        ds = Dataset.from_pandas(pd.DataFrame({"text": frame["clean_text"], "label": frame["intent"].map(label2id)}), preserve_index=False)
        return ds.map(lambda batch: tokenizer(batch["text"], padding="max_length", truncation=True, max_length=128), batched=True)

    train_ds, test_ds = make_dataset(train), make_dataset(test)
    training_args = TrainingArguments(output_dir=str(args.output_dir), num_train_epochs=args.epochs, per_device_train_batch_size=16, per_device_eval_batch_size=32, eval_strategy="epoch", save_strategy="epoch", report_to=[])
    trainer = Trainer(model=model, args=training_args, train_dataset=train_ds, eval_dataset=test_ds, processing_class=tokenizer)
    trainer.train()
    trainer.save_model(str(args.output_dir))
    tokenizer.save_pretrained(str(args.output_dir))
    predictions = trainer.predict(test_ds).predictions.argmax(axis=1)
    accuracy = float(np.mean(predictions == np.asarray(test_ds["label"])))
    print({"checkpoint": args.checkpoint, "test_accuracy": accuracy, "labels": len(labels)})


if __name__ == "__main__":
    main()
