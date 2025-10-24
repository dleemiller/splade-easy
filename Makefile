.PHONY: schema test clean install format lint pre-commit

# Compile FlatBuffers schema and fix imports
schema:
	@echo "Compiling FlatBuffers schema..."
	flatc --python -o src/splade_easy src/splade_easy/schema.fbs
	@echo "Fixing imports..."
	@sed -i 's/from SpladeEasy\./from ./g' src/splade_easy/SpladeEasy/*.py
	@echo "Done! Generated code in src/splade_easy/SpladeEasy/"

# Run tests
test:
	uv run pytest tests/ -v

# Run tests with coverage
test-cov:
	uv run pytest tests/ -v --cov=splade_easy --cov-report=html --cov-report=term

# Format code with ruff
format:
	uv run ruff format src/ tests/

# Lint code with ruff
lint:
	uv run ruff check src/ tests/

# Lint and fix
lint-fix:
	uv run ruff check --fix src/ tests/

# Install pre-commit hooks
pre-commit:
	uv run pre-commit install

# Run pre-commit on all files
pre-commit-all:
	uv run pre-commit run --all-files

# Clean generated files
clean:
	rm -rf src/splade_easy/SpladeEasy/
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	rm -rf .pytest_cache
	rm -rf htmlcov
	rm -rf .coverage

# Install in dev mode
install:
	uv sync
