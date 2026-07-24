.PHONY: install test lint typecheck check

install:
	python -m pip install -e ".[dev]"

test:
	python -m pytest

lint:
	python -m ruff check .

typecheck:
	python -m mypy

check: lint typecheck test
