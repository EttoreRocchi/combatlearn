# combatlearn Examples

This directory contains practical examples demonstrating common use cases for the **combatlearn** package.

## Examples

### 1. [basic_usage.py](basic_usage.py)
Simple demonstration of ComBat batch correction with the three available methods (Johnson, Fortin, Chen).

### 2. [cross_validation.py](cross_validation.py)
Shows how to use ComBat within scikit-learn's cross-validation framework to prevent data leakage.

### 3. [grid_search.py](grid_search.py)
Demonstrates hyperparameter tuning with GridSearchCV, including ComBat parameters.

### 4. [visualization.py](visualization.py)
Examples of visualizing batch effects before and after ComBat correction using the `plot_transformation` method.

## Running the Examples

Each example is a standalone Python script. To run an example:

```bash
python examples/basic_usage.py
```

## Requirements

All examples require the combatlearn package and its dependencies:

```bash
pip install combatlearn
```

Some examples may require additional packages for demonstration purposes (e.g., seaborn for enhanced plotting).
