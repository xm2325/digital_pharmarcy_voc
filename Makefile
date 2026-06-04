.PHONY: setup data train preview preview-image test smoke run all clean-reports
setup:
	python -m pip install -r requirements.txt

data:
	python -m src.generate_data --n 30000 --output-dir data

train:
	VOC_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m src.train_models --input data/synthetic_contacts_raw.csv --data-dir data --models-dir models --reports-dir reports

preview:
	python -m src.build_static_dashboard

preview-image:
	python -m src.build_preview_image

test:
	pytest -q

smoke:
	python scripts/check_streamlit_pages.py

run:
	streamlit run app.py

all: data train preview preview-image test
