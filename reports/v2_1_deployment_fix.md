# V2.1 Streamlit Community Cloud compatibility fix

## Observed deployment error

The Streamlit Community Cloud deployment raised `ModuleNotFoundError` while `joblib.load()` was deserialising a fitted scikit-learn pipeline. The dashboard artefacts were fitted under Python 3.13 and scikit-learn 1.8. The reported Cloud traceback used Python 3.14.

## Changes

- Moved fitted-model loading from global application startup to the chatbot page.
- Added `src/runtime_models.py`.
- The runtime first attempts to load the packaged `joblib` pipelines.
- If loading fails, the chatbot page rebuilds small in-memory models from `data/annotation_seed.csv`.
- Added visible fallback messaging so the runtime model source is transparent.
- Added dependency ranges for Python 3.13 and Python 3.14 compatible runtime wheels.
- Added regression tests for the fallback path.
- Updated deployment instructions to recommend Python 3.13 when creating a new Streamlit Community Cloud app.

## Interpretation

The offline benchmark metrics shown on the dashboard remain the stored V2 build outputs. A fallback rebuild supports the interactive text box only. It does not silently replace or recalculate the reported benchmark results.
