# Installation

## Requirements

combatlearn requires Python 3.10 or later.

## Install from PyPI

The easiest way to install combatlearn is via pip:

```bash
pip install combatlearn
```

This will install combatlearn and all required dependencies:

- pandas >= 1.3
- numpy >= 1.21
- scikit-learn >= 1.2
- matplotlib >= 3.4
- plotly >= 5.0
- umap-learn >= 0.5
- nbformat >= 4.2

## Install from Source

To install the latest development version from GitHub:

```bash
git clone https://github.com/EttoreRocchi/combatlearn.git
cd combatlearn
pip install -e .
```

## Development Installation

If you want to contribute to combatlearn, install the development dependencies:

```bash
git clone https://github.com/EttoreRocchi/combatlearn.git
cd combatlearn
pip install -e ".[dev]"
```

This installs additional tools for testing and code quality:

- pytest >= 7
- pytest-cov >= 4.0
- ruff >= 0.1
- mypy >= 1.0

## Verify Installation

To verify that combatlearn is installed correctly:

```python
import combatlearn
print(combatlearn.__version__)
```

You should see the version number printed (e.g., `1.0.0`).

## Troubleshooting

### ImportError: No module named 'combatlearn'

Make sure you've installed the package:

```bash
pip install combatlearn
```

### UMAP or Plotly not found

These are required dependencies and should be installed automatically. If you encounter issues, install them manually:

```bash
pip install umap-learn plotly
```

### Version Conflicts

If you have conflicting package versions, create a fresh virtual environment:

```bash
python -m venv combatlearn-env
source combatlearn-env/bin/activate  # On Windows: combatlearn-env\Scripts\activate
pip install combatlearn
```

## Next Steps

- [Quick Start Tutorial](quickstart.md)
- [User Guide](user-guide/overview.md)
- [Examples](examples/basic-usage.md)
