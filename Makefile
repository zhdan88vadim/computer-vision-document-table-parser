.PHONY: check test run clean

check:
	mypy main.py

test:
	pytest tests/

run:
	python main.py

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type d -name ".mypy_cache" -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -exec rm -rf {} +

all: check test