# Model card: V2 digital-pharmacy VoC workbench

## Intended use

This repository is a portfolio demonstration of customer-contact analytics for a digital-pharmacy setting. It uses generated data only. It is intended for code review, dashboard review, and discussion of an operational NLP workflow.

It is not intended for clinical decision-making, direct patient communication, or deployment without reviewed data and governance controls.

## Intent-classification model

The packaged dashboard uses a character-and-word TF-IDF representation with a probability-producing linear SGD classifier. Character features improve robustness to generated spelling errors and abbreviations while keeping local rebuild time reasonable.

### Held-out synthetic test result

| Metric | Value |
|---|---:|
| Test contacts | 7,200 |
| Accuracy | 0.932 |
| Macro-F1 | 0.933 |
| Weighted F1 | 0.932 |
| Mean confidence | 0.831 |
| Low-confidence rate at threshold 0.72 | 0.149 |
| Expected Calibration Error | 0.101 |
| Multiclass Brier score | 0.096 |
| Urgent-route recall | 1.000 |

The full per-intent classification report is stored in `model_metrics.json`. The confusion matrix is stored in `intent_confusion_matrix.csv`.

## Intent benchmark

`intent_model_benchmark.csv` compares:

- Word TF-IDF + linear SGD classifier
- Character-and-word TF-IDF + linear SGD classifier
- Local dense LSA embeddings + linear SGD classifier

Each is tested on a random stratified split and a temporal holdout split. The temporal split is weaker, which is expected because the final period includes a recent issue shift.

The local LSA model is a dense embedding baseline. It is not a transformer model.

## Independent sentiment model

The sentiment model uses separate character-and-word TF-IDF features and a linear SGD classifier.

| Metric | Value |
|---|---:|
| Test contacts | 7,200 |
| Accuracy | 0.879 |
| Macro-F1 | 0.614 |
| Weighted F1 | 0.860 |

The generated positive class is smaller and has much lower recall than the other classes. The dashboard reports this limitation. It should be addressed through better labelling, more positive examples, and model comparison before operational use.

## Routing logic

The model prediction is not the only routing input. Rule-based safety language and safety-sensitive intent labels override automated routing and send the contact to clinical escalation. Low-confidence cases are routed to a human adviser.

## Synthetic CRM driver models

Two interpretable logistic-regression models are fitted on synthetic patient-level linked features:

| Outcome | ROC-AUC | Average precision |
|---|---:|---:|
| Low CSAT | 0.698 | 0.478 |
| Not retained at 90 days | 0.581 | 0.397 |

These are association models for demonstration. They are not causal models and they are not company estimates.

## Optional transformer experiments

`optional_transformer_status.json` records that transformer models were not executed in the packaged offline build. The repository includes scripts for Sentence Transformers classification and Hugging Face fine-tuning in a connected environment. Add results only after running those scripts.

## Risks and controls

Main risks include incorrect intent prediction, weak probability calibration, insufficient redaction, taxonomy drift, missed safety language, and inappropriate automation. The dashboard includes a human-review queue, calibration report, safety override, redaction audit output, and unknown-topic review page to make these issues visible.
