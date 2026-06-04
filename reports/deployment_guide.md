# Streamlit deployment guide

## Recommended Streamlit Community Cloud deployment

1. Create a GitHub repository.
2. Upload the **contents** of this project folder so that `app.py`, `requirements.txt`, `data/`, `models/`, `reports/`, and `src/` are in the repository root.
3. In Streamlit Community Cloud, choose **Create app** and select the GitHub repository.
4. Set the main file path to `app.py`.
5. In **Advanced settings**, select Python **3.13** when available. The packaged offline model artefacts were generated under Python 3.13 and scikit-learn 1.8.
6. Deploy the app.

The V2.1 dashboard also supports Python 3.14 deployments. If a packaged `joblib` model cannot be deserialised because the Python, NumPy, SciPy, or scikit-learn stack differs, only the interactive chatbot page rebuilds lightweight in-memory models from `data/annotation_seed.csv`. The nine analytics pages are loaded independently and remain available.

## Existing deployment created with Python 3.14

You can deploy V2.1 without changing the Python version because of the runtime fallback. For the closest match to the packaged offline artefacts, redeploy with Python 3.13.

Streamlit Community Cloud does not allow the Python version of an existing app to be changed in place. Record the current settings, delete the app, create it again, and choose Python 3.13 in **Advanced settings**.

## Local run

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

## Full rebuild

```bash
make all
```

The rebuild regenerates data, runs isolated training stages, writes reports, rebuilds previews, and runs tests.

## Optional connected-environment models

Transformer checkpoints are not downloaded by the default deployment. Install `requirements-optional-transformers.txt` and run the optional scripts only in a suitable connected environment.
