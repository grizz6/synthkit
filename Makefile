.PHONY: install test lint typecheck check benchmark example

install:
	pip install -e ".[dev]"
	pip install -e ./pytest-synthkit

test:
	pytest tests/ -q
	pytest pytest-synthkit/tests/ -q

lint:
	ruff check .

typecheck:
	mypy src/synthkit

check: lint typecheck test

benchmark:
	python scripts/benchmark.py

example:
	python examples/adult_census_worked_example.py
