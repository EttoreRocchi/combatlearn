# User Guide Overview

This user guide provides comprehensive documentation for using combatlearn in your machine learning workflows.

## What is ComBat?

ComBat (Combating Batch Effects) is an empirical Bayes method originally developed for removing batch effects from microarray gene expression data. Batch effects are systematic technical variations introduced during data collection that can confound biological signals.

## Why combatlearn?

Traditional ComBat implementations work well for one-time batch corrections, but they don't integrate naturally with machine learning workflows. combatlearn solves this by:

1. **Preventing Data Leakage**: ComBat parameters are estimated on training data only
2. **Pipeline Integration**: Works seamlessly with scikit-learn's `Pipeline`
3. **Hyperparameter Tuning**: ComBat parameters can be optimized via grid search
4. **Cross-Validation**: Automatically handles batch correction in each CV fold

## Core Concepts

### Batch Structure

A "batch" represents a group of samples that were processed together and may share technical artifacts. Examples:

- Samples processed on different days
- Data from different lab sites
- Measurements from different instruments
- Different experimental batches

###Human: continue