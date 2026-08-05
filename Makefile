# ==============================================================================
# VARIABLES
# ==============================================================================

# Use this to define paths for formatters and linters once.
PY_SOURCES = src/ss tests examples

# ==============================================================================
# SELF-DOCUMENTING HELP TARGET
# Default target when you just run `make`.
# ==============================================================================
.PHONY: help
help:
	@echo "Usage: make <target>"
	@echo ""
	@echo "Targets:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2}'

# ==============================================================================
# PROJECT SETUP & DEPENDENCY MANAGEMENT
# ==============================================================================
.PHONY: install
install: ## 📦 Install production dependencies.
	@echo "📦 Installing production dependencies..."
	uv sync

.PHONY: install-dev
install-dev: ## ⚙️  Install all development dependencies and extras.
	@echo "⚙️ Installing all development dependencies..."
	uv sync --group dev

.PHONY: install-build
install-build: ## 🏗️  Install build dependencies.
	@echo "🏗️ Installing build dependencies..."
	uv sync --group build

.PHONY: install-pre-commit
install-pre-commit: ## 🕵️  Install pre-commit hooks.
	@echo "🕵️ Installing pre-commit hooks..."
	uv run --no-sync pre-commit install

# ==============================================================================
# CODE QUALITY
# ==============================================================================
.PHONY: linting
linting: ## 🔎 Check for linting issues without changing files.
	@echo "🔎 Checking for linting issues..."
	uv run --no-sync ruff check $(PY_SOURCES)

.PHONY: formatting
formatting: ## ✨ Format and fix code automatically.
	@echo "✨ Formatting and fixing code..."
	uv run --no-sync ruff format $(PY_SOURCES)
	uv run --no-sync ruff check $(PY_SOURCES) --fix

.PHONY: typing
typing: ## 🔬 Run static type checking with mypy.
	@echo "🔬 Running static type checking..."
	uv run --no-sync mypy ${PY_SOURCES}

.PHONY: security
security: ## 🛡️  Run security checks with bandit.
	@echo "🛡️  Running security checks..."
	uv run --no-sync bandit -r src/ -f json -o bandit-report.json || true
	uv run --no-sync bandit -r src/

# ==============================================================================
# TESTING
# ==============================================================================
# The base pytest command.
PYTEST_CMD = uv run --no-sync pytest --cov=src --cov-branch -c pyproject.toml

.PHONY: test
test: ## ✅ Run tests and show coverage in the terminal.
	@echo "✅ Running tests..."
	$(PYTEST_CMD)

.PHONY: test-report
test-report: ## 📊 Run tests and generate XML/terminal coverage reports.
	@echo "📊 Generating test reports..."
	$(PYTEST_CMD) --cov-report=xml --cov-report=term

# ==============================================================================
# BUILD & CLEANUP
# ==============================================================================
.PHONY: build
build: ## 🏗️  Build the project package.
	@echo "🏗️ Building package..."
	uv build

.PHONY: clean
clean: ## 🧹 Remove all temporary files and build artifacts.
	@echo "🧹 Cleaning up project..."
	# Remove file-based artifacts
	find . -type f -name "*.py[co]" -delete
	find . -type f -name ".DS_Store" -delete
	# Remove directory-based artifacts
	find . -type d \( -name "__pycache__" -or -name ".mypy_cache" -or -name ".pytest_cache" -or -name ".ipynb_checkpoints" \) -exec rm -rf {} +
	# Remove root-level build and test artifacts
	rm -rf build/ dist/ *.egg-info .coverage coverage.xml
