# Makefile for IRST Library
# Professional development workflow automation

.PHONY: help install install-dev install-docs test test-fast test-gpu lint format clean build docs serve-docs deploy publish clean-build clean-pyc clean-test benchmark profile security

# Default target
help:
	@echo "IRST Library - Development Commands"
	@echo "==================================="
	@echo "Installation:"
	@echo "  install       Install package"
	@echo "  install-dev   Install development dependencies"
	@echo "  install-docs  Install documentation dependencies"
	@echo ""
	@echo "Testing:"
	@echo "  test          Run all tests"
	@echo "  test-fast     Run fast tests only"
	@echo "  test-gpu      Run GPU tests"
	@echo "  benchmark     Run benchmarks"
	@echo ""
	@echo "Code Quality:"
	@echo "  lint          Run linting"
	@echo "  format        Format code"
	@echo "  security      Run security checks"
	@echo "  profile       Profile code performance"
	@echo ""
	@echo "Documentation:"
	@echo "  docs          Build documentation"
	@echo "  serve-docs    Serve documentation locally"
	@echo ""
	@echo "Building & Publishing:"
	@echo "  build         Build package"
	@echo "  publish       Publish to PyPI"
	@echo "  clean         Clean all build artifacts"

# Installation
install:
	pip install -e .

install-dev:
	pip install -e ".[dev]"
	pre-commit install

install-docs:
	pip install -e ".[docs]"

install-all:
	pip install -e ".[all]"

# Testing
test:
	pytest -v --cov=irst_library --cov-report=html --cov-report=term-missing

test-fast:
	pytest -v -m "not slow and not gpu" --cov=irst_library

test-gpu:
	pytest -v -m "gpu" --cov=irst_library

test-integration:
	pytest -v -m "integration" --cov=irst_library

benchmark:
	pytest -v -m "benchmark" --benchmark-only --benchmark-html=benchmark_results.html

# Code Quality
lint:
	flake8 irst_library tests examples
	mypy irst_library
	bandit -r irst_library
	black --check irst_library tests examples
	isort --check-only irst_library tests examples

format:
	black irst_library tests examples
	isort irst_library tests examples

security:
	bandit -r irst_library
	safety check
	pip-audit

profile:
	python -m cProfile -o profile_results.prof examples/profile_model.py
	python -c "import pstats; pstats.Stats('profile_results.prof').sort_stats('cumulative').print_stats(20)"

# Documentation
docs:
	cd docs && make html

serve-docs:
	cd docs/_build/html && python -m http.server 8000

docs-clean:
	cd docs && make clean

# Building & Publishing
build: clean
	python -m build

publish-test: build
	python -m twine upload --repository testpypi dist/*

publish: build
	python -m twine upload dist/*

# Docker
docker-build:
	docker build -t irst-library:latest .

docker-run:
	docker run -it --rm --gpus all -v $(PWD):/workspace irst-library:latest

docker-compose-up:
	docker-compose up --build

# Model Export
export-onnx:
	python scripts/export_models.py --format onnx --output models/

export-tensorrt:
	python scripts/export_models.py --format tensorrt --output models/

# Dataset Management
download-datasets:
	python scripts/download_datasets.py --all

prepare-datasets:
	python scripts/prepare_datasets.py --dataset all

# Training
train-serank:
	irst-train --config configs/experiments/serank_sirst.yaml

train-acm:
	irst-train --config configs/experiments/acm_nuaa_sirst.yaml

train-all:
	python scripts/train_all_models.py

# Evaluation
evaluate-all:
	python scripts/evaluate_all_models.py --output results/

benchmark-models:
	python scripts/benchmark_models.py --output benchmark_results/

# Deployment
deploy-api:
	docker-compose -f docker-compose.prod.yml up -d

deploy-monitoring:
	docker-compose -f docker-compose.monitoring.yml up -d

# Cleaning
clean: clean-build clean-pyc clean-test clean-docs

clean-build:
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	rm -rf wheels/

clean-pyc:
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	find . -type f -name "*.pyo" -delete

clean-test:
	rm -rf .pytest_cache/
	rm -rf .coverage
	rm -rf htmlcov/
	rm -rf .tox/
	rm -rf coverage.xml
	rm -rf *.prof

clean-docs:
	rm -rf docs/_build/

# Development helpers
setup-dev: install-dev
	pip install -e ".[all]"
	pre-commit install
	@echo "Development environment ready!"

check-all: lint test security
	@echo "All checks passed!"

release-check: clean build
	python -m twine check dist/*
	@echo "Release check completed!"

# CI/CD helpers
ci-install:
	pip install -e ".[dev]"

ci-test:
	pytest --cov=irst_library --cov-report=xml --junitxml=junit.xml

ci-lint:
	flake8 irst_library tests --format=junit-xml --output-file=flake8.xml
	mypy irst_library --junit-xml mypy.xml || true

# Performance testing
stress-test:
	python scripts/stress_test.py --duration 300 --concurrent 10

memory-test:
	python -m memory_profiler examples/memory_test.py

# Model validation
validate-models:
	python scripts/validate_pretrained_models.py

check-model-compatibility:
	python scripts/check_model_compatibility.py --all

# Documentation helpers
api-docs:
	sphinx-apidoc -o docs/source/api irst_library

update-requirements:
	pip-compile requirements.in
	pip-compile requirements-dev.in

# GitHub Actions helpers
gh-setup:
	gh workflow run ci.yml
	gh workflow run docs.yml

# Local development server
dev-server:
	python -m irst_library.server --dev --reload
