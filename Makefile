.DEFAULT_GOAL := help
.PHONY: help install install-docs lint format format-check pre-commit \
        test test-cov typecheck docs docs-clean docs-serve clean publish publish-test

# Colours
BOLD  := \033[1m
RESET := \033[0m
CYAN  := \033[36m

# Paths
SRC   := combatlearn
TESTS := tests

help:  ## Show this help message
	@printf "$(BOLD)combatlearn - available targets$(RESET)\n\n"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
	  | awk 'BEGIN {FS = ":.*?## "}; {printf "  $(CYAN)%-18s$(RESET) %s\n", $$1, $$2}'

# Installation

install:  ## Install package with development dependencies
	@pip install -e ".[dev]"

install-docs:  ## Install package with documentation dependencies
	@pip install -e ".[docs]"

# Code quality

lint:  ## Run ruff linter
	@ruff check --fix $(SRC)/ $(TESTS)/

format:  ## Run ruff formatter (applies changes)
	@ruff format $(SRC)/ $(TESTS)/

format-check:  ## Run ruff formatter in check-only mode (no changes)
	@ruff format --check $(SRC)/ $(TESTS)/

pre-commit:  ## Run the full pre-commit suite on all files
	@pre-commit run --all-files

typecheck:  ## Run mypy type checking
	@mypy $(SRC)/

# Tests

test:  ## Run tests
	@pytest $(TESTS)/

test-cov:  ## Run tests with coverage report
	@pytest $(TESTS)/ --cov=$(SRC) --cov-report=term-missing

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

# Release

publish: clean  ## Build and publish package to PyPI
	@read -p "Publish to PyPI? [y/N] " ans && [ "$$ans" = "y" ]
	@python -m build
	@twine upload dist/*

publish-test: clean  ## Build and publish package to TestPyPI
	@python -m build
	@twine upload --repository testpypi dist/*
