# V2.1 deployment-fix verification summary

- Version: `2.1.0`
- Root cause class: packaged `joblib` model deserialisation failed during module import in the Streamlit Cloud Python 3.14 environment.
- Fix: lazy model loading plus in-memory fallback rebuild from `data/annotation_seed.csv`.
- Unit tests: `15 passed`.
- Packaged-model chatbot page check: passed.
- Simulated incompatible-model chatbot page check: passed.
- Fallback rebuild input: `3,000` labelled seed contacts.
- Fallback example output: intent `waiting_gp_approval`; route `Proactive message`.
- Analytics-page Streamlit smoke checks: passed.
- Chatbot-page Streamlit smoke check: passed.
- Local Streamlit health endpoint: `ok`.
