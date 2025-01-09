#* Variables
PYTHON := python3
PYTHONPATH := `pwd`

#* Installation
.PHONY: install

install:
	python -m pip install --upgrade pip
	python -m pip install -e ".[dev]"


#* Installation of pre-commit: tool of Git hook scripts
.PHONY: install-pre-commit
install-pre-commit:
	pre-commit install

#* Unittests
.PHONY: test
test:
	pytest -c pyproject.toml --cov=src --cov-report=xml --cov-report=term

#* Formatters
.PHONY: formatting
formatting:
	pyupgrade --exit-zero-even-if-changed --py38-plus src/**/*.py
	isort --settings-path pyproject.toml ./
	black --config pyproject.toml ./
	mypy --config-file pyproject.toml ./

#* Linting
.PHONY: check-test
check-test:
	poetry run pytest -c pyproject.toml --cov=src

.PHONY: check-formatting
check-formatting:
	poetry run isort --diff --check-only --settings-path pyproject.toml ./
	poetry run black --diff --check --config pyproject.toml ./
	poetry run mypy --config-file pyproject.toml src

.PHONY: lint
lint: check-formatting check-test

#* Update developer tools
.PHONY: update-dev
update-dev:
	pip install --upgrade \
    "isort[colors]" \
    mypy \
    pre-commit \
    pytest \
    pyupgrade \
    coverage \
    pytest-html \
    pytest-cov \
    black \
    pydocstyle \
    pylint

#* Cleaning
.PHONY: pycache-remove
pycache-remove:
	find . | grep -E "(__pycache__|\.pyc|\.pyo$$)" | xargs rm -rf

.PHONY: dsstore-remove
dsstore-remove:
	find . | grep -E ".DS_Store" | xargs rm -rf

.PHONY: mypycache-remove
mypycache-remove:
	find . | grep -E ".mypy_cache" | xargs rm -rf

.PHONY: ipynbcheckpoints-remove
ipynbcheckpoints-remove:
	find . | grep -E ".ipynb_checkpoints" | xargs rm -rf

.PHONY: pytestcache-remove
pytestcache-remove:
	find . | grep -E ".pytest_cache" | xargs rm -rf

.PHONY: build-remove
build-remove:
	rm -rf build/

.PHONY: cleanup
cleanup: pycache-remove dsstore-remove mypycache-remove ipynbcheckpoints-remove pytestcache-remove
