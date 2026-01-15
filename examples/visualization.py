"""
Visualization Example
======================

This example demonstrates how to visualize batch effects before and after
ComBat correction using the built-in plot_transformation method.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from combatlearn import ComBat

# Set random seed
np.random.seed(42)

# Generate synthetic data with strong batch effects
n_samples = 200
n_features = 100
batches = np.repeat(["Batch_1", "Batch_2", "Batch_3", "Batch_4"], [50, 50, 50, 50])

# Create data with batch effects
X = np.random.randn(n_samples, n_features)
for i, batch in enumerate(batches):
    if batch == "Batch_1":
        X[i] += 3.0
    elif batch == "Batch_2":
        X[i] -= 2.5
    elif batch == "Batch_3":
        X[i] += 1.5
    elif batch == "Batch_4":
        X[i] -= 1.0

# Convert to DataFrame
X = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(n_features)])
batch = pd.Series(batches, index=X.index, name="batch")

print("=" * 70)
print("Batch Effect Visualization Example")
print("=" * 70)
print(f"\nDataset: {X.shape[0]} samples, {X.shape[1]} features")
print(f"Batches: {batch.value_counts().to_dict()}")

# Fit ComBat
print("\nFitting ComBat model...")
combat = ComBat(batch=batch, method="johnson").fit(X)

# Example 1: 2D PCA visualization (static)
print("\n" + "=" * 70)
print("1. 2D PCA Visualization (Static)")
print("=" * 70)

fig = combat.plot_transformation(
    X,
    reduction_method="pca",
    n_components=2,
    plot_type="static",
    figsize=(14, 6),
    alpha=0.7,
    point_size=60,
    cmap="Set2",
    title="ComBat Correction Effect - 2D PCA",
)
plt.savefig("combat_2d_pca.png", dpi=300, bbox_inches="tight")
print("Saved: combat_2d_pca.png")
plt.close()

# Example 2: 3D PCA visualization (static)
print("\n" + "=" * 70)
print("2. 3D PCA Visualization (Static)")
print("=" * 70)

fig = combat.plot_transformation(
    X,
    reduction_method="pca",
    n_components=3,
    plot_type="static",
    figsize=(14, 6),
    alpha=0.6,
    point_size=40,
    cmap="tab10",
    title="ComBat Correction Effect - 3D PCA",
)
plt.savefig("combat_3d_pca.png", dpi=300, bbox_inches="tight")
print("Saved: combat_3d_pca.png")
plt.close()

# Example 3: t-SNE visualization
print("\n" + "=" * 70)
print("3. t-SNE Visualization")
print("=" * 70)

fig = combat.plot_transformation(
    X,
    reduction_method="tsne",
    n_components=2,
    plot_type="static",
    figsize=(14, 6),
    alpha=0.7,
    point_size=60,
    perplexity=40,  # t-SNE specific parameter
)
plt.savefig("combat_tsne.png", dpi=300, bbox_inches="tight")
print("Saved: combat_tsne.png")
plt.close()

# Example 4: Return embeddings for custom plotting
print("\n" + "=" * 70)
print("4. Custom Plotting with Returned Embeddings")
print("=" * 70)

fig, embeddings = combat.plot_transformation(
    X,
    reduction_method="pca",
    n_components=2,
    plot_type="static",
    return_embeddings=True,
)
plt.close()

# Create custom plot
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# Plot original
for batch_name in batch.unique():
    mask = batch == batch_name
    ax1.scatter(
        embeddings["original"][mask, 0],
        embeddings["original"][mask, 1],
        label=batch_name,
        alpha=0.7,
        s=60,
        edgecolors="black",
        linewidth=0.5,
    )

ax1.set_title("Before ComBat (Custom Plot)", fontsize=12, fontweight="bold")
ax1.set_xlabel("PC1")
ax1.set_ylabel("PC2")
ax1.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
ax1.grid(True, alpha=0.3)

# Plot corrected
for batch_name in batch.unique():
    mask = batch == batch_name
    ax2.scatter(
        embeddings["transformed"][mask, 0],
        embeddings["transformed"][mask, 1],
        label=batch_name,
        alpha=0.7,
        s=60,
        edgecolors="black",
        linewidth=0.5,
    )

ax2.set_title("After ComBat (Custom Plot)", fontsize=12, fontweight="bold")
ax2.set_xlabel("PC1")
ax2.set_ylabel("PC2")
ax2.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("combat_custom_plot.png", dpi=300, bbox_inches="tight")
print("Saved: combat_custom_plot.png")
plt.close()

# Example 5: Interactive plot
print("\n" + "=" * 70)
print("5. Interactive Plotly Visualization")
print("=" * 70)

fig = combat.plot_transformation(
    X,
    reduction_method="pca",
    n_components=2,
    plot_type="interactive",
    title="ComBat Correction - Interactive",
)
fig.write_html("combat_interactive.html")
print("Saved: combat_interactive.html")
print("Open this file in a web browser for interactive exploration!")

print("\n" + "=" * 70)
print("Visualization examples completed!")
print("\nGenerated files:")
print("  - combat_2d_pca.png")
print("  - combat_3d_pca.png")
print("  - combat_tsne.png")
print("  - combat_custom_plot.png")
print("  - combat_interactive.html")
print("=" * 70)
