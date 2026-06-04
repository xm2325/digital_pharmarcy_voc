# V2 run summary

## Build status

The V2 repository was rebuilt locally from generated data and checked through unit tests and Streamlit page execution tests.

| Check | Result |
|---|---:|
| Generated contacts | 30,000 |
| Generated CRM rows | 7,348 |
| Held-out prediction rows | 7,200 |
| Unit tests | 13 passed |
| Streamlit pages executed | 10 passed |

## Main model outputs

| Metric | Value |
|---|---:|
| Packaged intent model | Character-and-word TF-IDF + linear SGD classifier |
| Intent accuracy | 0.932 |
| Intent Macro-F1 | 0.933 |
| Intent Weighted F1 | 0.932 |
| Temporal benchmark Macro-F1 | 0.892 |
| Expected Calibration Error | 0.101 |
| Low-confidence review rate at 0.72 | 14.9% |
| Urgent-route recall | 100.0% |
| Independent sentiment Macro-F1 | 0.614 |

## Governance outputs

| Metric | Value |
|---|---:|
| Contacts with at least one local rule-based redaction | 17.8% |
| Total redactions | 5,810 |
| Short-text rate | 0.3% |
| Duplicate-contact rate | 0.0% |

## Synthetic routing scenario

At confidence threshold 0.72:

| Route | Contacts |
|---|---:|
| Chatbot or FAQ self-service | 1,908 |
| Proactive message | 1,753 |
| Human adviser | 3,338 |
| Clinical escalation | 201 |
| Potentially deflected | 3,661 |
| Potential deflection rate | 50.8% |

The financial scenario uses demonstration assumptions only. It is not a Pharmacy2U estimate.

## Emerging issues and taxonomy evolution

The known-intent monitor flags a recent rise in `delivery_delayed` contacts and an increase in `unclear_or_out_of_scope` contacts. The unknown-topic workflow identifies a 27-contact pure cluster with terms such as `latest app`, `update`, and `open prescription`, then recommends creating or refining an intent.

## CRM driver models

| Outcome | ROC-AUC | Average precision |
|---|---:|---:|
| Low CSAT | 0.698 | 0.478 |
| Not retained at 90 days | 0.581 | 0.397 |

These are association models on generated data, not causal estimates.
