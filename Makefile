.PHONY: install test lint typecheck check benchmark examples

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

examples:
	python examples/adult_census_worked_example.py
	python examples/titanic_example.py
	python examples/wine_quality_example.py
	python examples/bike_sharing_example.py
