.DEFAULT_GOAL := help
.PHONY: help install install-docs lint lint-check format format-check pre-commit \
        test test-cov docs docs-clean docs-serve clean build check \
        publish publish-test

# Colours
BOLD  := \033[1m
RESET := \033[0m
CYAN  := \033[36m

# Overridable tool paths
RUFF       ?= ruff
PYTEST     ?= pytest
PIP        ?= pip
PRECOMMIT  ?= pre-commit
TWINE      ?= twine
BUILD      ?= python -m build

# Paths
SRC   := combatlearn
TESTS := tests

help:  ## Show this help message
	@printf "$(BOLD)combatlearn - available targets$(RESET)\n\n"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
	  | awk 'BEGIN {FS = ":.*?## "}; {printf "  $(CYAN)%-18s$(RESET) %s\n", $$1, $$2}'

# Installation

install:  ## Install package with development dependencies
	@$(PIP) install -e ".[dev]"

install-docs:  ## Install package with documentation dependencies
	@$(PIP) install -e ".[docs]"

# Code quality

lint:  ## Run ruff linter (auto-fix)
	@$(RUFF) check --fix $(SRC)/ $(TESTS)/

lint-check:  ## Run ruff linter in check-only mode (no fixes)
	@$(RUFF) check $(SRC)/ $(TESTS)/

format:  ## Run ruff formatter (applies changes)
	@$(RUFF) format $(SRC)/ $(TESTS)/

format-check:  ## Run ruff formatter in check-only mode (no changes)
	@$(RUFF) format --check $(SRC)/ $(TESTS)/

pre-commit:  ## Run the full pre-commit suite on all files
	@$(PRECOMMIT) run --all-files

# Tests

test:  ## Run tests
	@$(PYTEST) $(TESTS)/

test-cov:  ## Run tests with coverage report
	@$(PYTEST) $(TESTS)/ --cov=$(SRC) --cov-report=term-missing

# Composite targets

check: lint-check format-check test  ## Run lint + format-check + tests (CI gate)

# Documentation

docs:  ## Build Sphinx HTML documentation
	@$(MAKE) -C docs html

docs-clean:  ## Remove Sphinx build artefacts
	@$(MAKE) -C docs clean

docs-serve:  ## Serve built docs locally on http://localhost:8080
	@python -m http.server --directory docs/_build/html 8080

# Housekeeping

clean:  ## Remove build artefacts and caches
	@find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	@find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	@find . -type d -name "*.egg-info"   -exec rm -rf {} + 2>/dev/null || true
	@rm -rf dist/ build/

# Build & Release

build: clean  ## Build distribution packages
	@$(BUILD)

publish: build  ## Build and publish package to PyPI
	@$(TWINE) upload dist/*

publish-test: build  ## Build and publish package to TestPyPI
	@$(TWINE) upload --repository testpypi dist/*
