# Technical notes for optional extensions

The default project remains runnable without external model downloads. These optional paths can be tested in a connected environment.

## Sentence Transformers

Sentence Transformers can map text to dense vector representations and support semantic-search workflows. The repository includes `src/optional_transformer_benchmark.py` for an optional intent-classification comparison and a `sentence_transformer` mode in `src/faq_retrieval.py`.

Official documentation:

- https://sbert.net/
- https://www.sbert.net/examples/sentence_transformer/applications/semantic-search/README.html

## Hugging Face Transformers

The optional `src/optional_hf_finetune.py` script provides an entry point for sequence-classification fine-tuning after downloading a checkpoint.

Official documentation:

- https://huggingface.co/docs/transformers/en/tasks/sequence_classification

## Topic summaries

The local topic-discovery module uses latent-semantic embeddings and c-TF-IDF-style cluster summaries. It does not claim to run BERTopic. BERTopic is a possible connected-environment comparison.

Official documentation:

- https://maartengr.github.io/BERTopic/index.html
- https://maartengr.github.io/BERTopic/getting_started/ctfidf/ctfidf.html

## Redaction

The packaged redactor uses explicit local rules so that its actions can be audited. Microsoft Presidio is listed as an optional production-oriented experiment because it provides analyser and anonymiser components for PII handling.

Official documentation:

- https://microsoft.github.io/presidio/
- https://microsoft.github.io/presidio/text_anonymization/
