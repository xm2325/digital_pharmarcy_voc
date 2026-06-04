# Data card: synthetic digital-pharmacy contacts and CRM records

## Summary

The repository contains generated demonstration data only.

| Table | Rows | Purpose |
|---|---:|---|
| `synthetic_contacts_raw.csv` | 30,000 | Multi-channel customer-contact text and generated labels |
| `synthetic_crm.csv` | 7,348 | Generated patient-level satisfaction and retention outcomes |
| `prediction_sample.csv` | 7,200 | Held-out contacts with model predictions and routing results |
| `annotation_review_queue.csv` | 700 | Prioritised contacts for human label review |
| `faq_knowledge_base.csv` | 8 | Small demonstration FAQ table |
| `faq_retrieval_eval.csv` | 10 | Retrieval-quality test queries |

## Channels

- Phone transcript
- Email
- Live chat
- App review
- Social text

## Generated text conditions

The generator adds channel-specific phrasing, abbreviations, misspellings, vague questions, multi-intent distractors, and personal-data-like strings. It also adds a known delivery-delay rise and an unknown app-update/app-crash pattern in the final period so that monitoring workflows can be checked.

## Personal-data-like strings

The generated text may contain fake email addresses, phone numbers, dates of birth, postcodes, NHS-number-like strings, order references, named-person phrases, GP surgery phrases, and street addresses. These values exist only to test the local redaction workflow.

## CRM outcomes

Customer Satisfaction (CSAT) and 90-day retention are generated outcomes. They support pipeline testing and dashboard design. They are not observations from Pharmacy2U or another organisation.

## Use boundary

The dataset is suitable for portfolio review, testing, and teaching. It is not suitable for estimating live service performance, patient behaviour, financial value, or clinical risk.
