# Contributing

Thank you for considering contributing to combatlearn! This document provides guidelines for contributing.

## Getting Started

1. Fork the repository on GitHub
2. Clone your fork locally
3. Create a new branch for your feature or bugfix

```bash
git clone https://github.com/EttoreRocchi/combatlearn.git
cd combatlearn
git checkout -b feature/your-feature-name
```

## Development Setup

Install combatlearn in development mode with dev dependencies:

```bash
pip install -e ".[dev]"
```

This installs:
- pytest for testing
- pytest-cov for coverage
- ruff for linting
- mypy for type checking

## Running Tests

Run the test suite:

```bash
pytest
```

Run with coverage:

```bash
pytest --cov=combatlearn --cov-report=html
```

## Code Style

We use `ruff` for linting and formatting:

```bash
# Check code style
ruff check .

# Auto-fix issues
ruff check --fix .

# Format code
ruff format .
```

## Type Checking

Run type checking with mypy:

```bash
mypy combatlearn
```

## Adding Features

1. Write tests first (TDD approach)
2. Implement the feature
3. Ensure all tests pass
4. Add documentation
5. Submit a pull request

## Documentation

Documentation is built with MkDocs. To preview docs locally:

```bash
pip install mkdocs mkdocs-material mkdocstrings[python]
mkdocs serve
```

Then visit `http://127.0.0.1:8000`

## Pull Request Process

1. Update documentation for new features
2. Add tests for bug fixes
3. Ensure all tests pass
4. Update CHANGELOG.md
5. Submit PR with clear description

## Reporting Bugs

Use GitHub Issues to report bugs. Please include:

- Python version
- combatlearn version
- Minimal code to reproduce
- Expected vs actual behavior
- Full error traceback

## Feature Requests

Feature requests are welcome! Please:

- Check if feature already exists
- Provide clear use case
- Explain expected behavior
- Consider submitting a PR

## Code of Conduct

Be respectful and constructive in all interactions.

## License

By contributing, you agree that your contributions will be licensed under the MIT License.
