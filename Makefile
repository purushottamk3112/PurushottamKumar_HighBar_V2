.PHONY: help setup install test run clean lint format

# Default target
help:
	@echo "Kasparro V2 Production - Makefile Commands"
	@echo ""
	@echo "Available commands:"
	@echo "  make setup      - Create virtual environment and install dependencies"
	@echo "  make install    - Install dependencies only"
	@echo "  make test       - Run unit tests"
	@echo "  make run        - Run analysis with default settings"
	@echo "  make clean      - Clean generated files"
	@echo "  make lint       - Run code linting (requires pylint)"
	@echo "  make format     - Format code (requires black)"
	@echo ""

# Setup virtual environment and install
setup:
	@echo "Setting up virtual environment..."
	python3 -m venv venv
	@echo "Activating and installing dependencies..."
	. venv/bin/activate && pip install --upgrade pip
	. venv/bin/activate && pip install -r requirements.txt
	@echo ""
	@echo "✓ Setup complete!"
	@echo "Activate with: source venv/bin/activate"

# Install dependencies only
install:
	pip install -r requirements.txt

# Run unit tests
test:
	@echo "Running unit tests..."
	python -m pytest test_evaluator.py -v --tb=short
	@echo ""
	@echo "✓ Tests complete!"

# Run with coverage
test-coverage:
	@echo "Running tests with coverage..."
	python -m pytest test_evaluator.py -v --cov=. --cov-report=term-missing
	@echo ""

# Run analysis
run:
	@echo "Running analysis..."
	python main.py
	@echo ""

# Run with custom CSV
run-custom:
	@echo "Running analysis with custom CSV..."
	@read -p "Enter CSV path: " csv_path; \
	python main.py --csv $$csv_path
	@echo ""

# Clean generated files
clean:
	@echo "Cleaning generated files..."
	rm -rf outputs/
	rm -rf logs/
	rm -rf __pycache__/
	rm -rf .pytest_cache/
	rm -rf *.pyc
	rm -rf .coverage
	@echo "✓ Clean complete!"

# Lint code (requires pylint)
lint:
	@echo "Running linter..."
	pylint *.py --disable=C0103,R0913,R0914

# Format code (requires black)
format:
	@echo "Formatting code..."
	black *.py --line-length 100

# Quick validation
validate:
	@echo "Validating setup..."
	@python -c "import pandas; import numpy; import scipy; import yaml; print('✓ All imports OK')"
	@python -c "from schema import DataSchema; print('✓ Schema OK')"
	@python -c "from utils import ConfigLoader; print('✓ Utils OK')"
	@echo "✓ Validation complete!"
