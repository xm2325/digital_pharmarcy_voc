#!/usr/bin/env bash
set -euo pipefail
INPUT_PATH="${1:-data/synthetic_contacts_raw.csv}"
DATA_DIR="${2:-data}"
MODELS_DIR="${3:-models}"
REPORTS_DIR="${4:-reports}"
export VOC_NUM_THREADS="${VOC_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
python -m src.train_models --stage sentiment --input "$INPUT_PATH" --data-dir "$DATA_DIR" --models-dir "$MODELS_DIR" --reports-dir "$REPORTS_DIR"
python -m src.train_models --stage intent --input "$INPUT_PATH" --data-dir "$DATA_DIR" --models-dir "$MODELS_DIR" --reports-dir "$REPORTS_DIR"
python -m src.train_models --stage finalize --input "$INPUT_PATH" --data-dir "$DATA_DIR" --models-dir "$MODELS_DIR" --reports-dir "$REPORTS_DIR"
