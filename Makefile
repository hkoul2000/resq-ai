.PHONY: setup smoke test lint data train evaluate optimize paper app docker all clean

PYTHON_CMD := $(if $(OS),py -3.13,python3)

setup:
	$(PYTHON_CMD) -m venv .venv
	.venv/Scripts/pip install -r requirements.txt
	.venv/Scripts/pip install -e .[dev]

smoke:
	.venv/Scripts/pytest -m smoke

test:
	.venv/Scripts/pytest

lint:
	.venv/Scripts/ruff check src tests
	.venv/Scripts/black --check src tests

data:
	@echo "Downloading and processing data..."

train:
	@echo "Training models..."

evaluate:
	@echo "Running evaluations..."

optimize:
	@echo "Running allocation optimization..."

paper:
	@echo "Compiling paper..."

app:
	@echo "Running app..."

docker:
	docker build -t resq-ai .

all: data train evaluate optimize paper

clean:
	rm -rf .venv
	rm -rf data/cache/*
	rm -rf __pycache__
	rm -rf .pytest_cache
