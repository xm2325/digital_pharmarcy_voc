# Streamlit deployment guide

## Streamlit Community Cloud

1. Create a GitHub repository.
2. Upload the contents of this project folder while keeping `app.py`, `requirements.txt`, `data/`, `models/`, and `reports/` in the repository root.
3. In Streamlit Community Cloud, choose **Create app** and select the GitHub repository.
4. Set the main file path to `app.py`.
5. Deploy the app.

The packaged synthetic data and fitted local models are included, so the public dashboard does not retrain during startup.

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
