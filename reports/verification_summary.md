# Verification summary

## Full rebuild

Command:

```bash
make all
```

Result: passed with exit status `0`.

Observed local rebuild time in the packaging environment: approximately 1 minute 35 seconds.

The full rebuild performs:

1. Synthetic data generation
2. Independent sentiment-model training
3. Intent benchmark and packaged intent-model training
4. Final report generation, taxonomy discovery, FAQ evaluation, and synthetic CRM driver analysis
5. Portable HTML preview generation
6. PNG preview generation
7. Unit tests

## Unit tests

Command:

```bash
pytest -q
```

Result: `13 passed`.

## Streamlit page smoke test

Command:

```bash
make smoke
```

Result: `10 Streamlit pages passed`.

## Local Streamlit service health

Command:

```bash
streamlit run app.py --server.headless true --server.port 8517
curl http://127.0.0.1:8517/_stcore/health
```

Result: `ok`.
