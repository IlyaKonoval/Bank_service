.PHONY: train test lint coverage docker-build docker-run run clean

train:
	python train.py

test:
	pytest tests/ -v --tb=short

lint:
	ruff check .

coverage:
	pytest tests/ -v --cov=pipeline --cov-report=term-missing

docker-build:
	docker build -t bank-service .

docker-run:
	docker-compose up --build

run:
	streamlit run app.py

clean:
	rm -rf artifacts/*.png artifacts/*.json artifacts/training.log
	rm -rf __pycache__ pipeline/__pycache__ tests/__pycache__
	rm -rf .pytest_cache catboost_info
