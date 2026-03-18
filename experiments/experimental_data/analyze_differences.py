import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from sklearn.mixture import GaussianMixture
import pandas as pd

# Load the uploaded file to inspect its contents
file_path = "differences.csv"
data = pd.read_csv(file_path)

# Columns to analyze
columns_to_analyze = ["1-3", "1-2", "2-3"]

# Prepare subplots
fig, axes = plt.subplots(1, len(columns_to_analyze), figsize=(15, 5))

# Loop through each column for analysis
for i, column in enumerate(columns_to_analyze):
    if column not in data.columns:
        print(f"Column '{column}' not found in the data.")
        continue

    print(f"Analyzing column: {column}")
    # Extract the data, drop NaN values, and filter by absolute value
    column_data = data[column].dropna()
    column_data = column_data[np.abs(column_data) <= 3]

    # Fit a Gaussian Mixture Model (GMM) to the data
    gmm = GaussianMixture(n_components=2, random_state=42)
    gmm.fit(column_data.values.reshape(-1, 1))

    # Extract parameters from the fitted GMM
    means = gmm.means_.flatten()
    stds = np.sqrt(gmm.covariances_).flatten()
    weights = gmm.weights_

    # Create a histogram of the data
    x_values = np.linspace(column_data.min(), column_data.max(), 1000)
    pdf_1 = weights[0] * norm.pdf(x_values, means[0], stds[0])
    pdf_2 = weights[1] * norm.pdf(x_values, means[1], stds[1])

    # Plot the histogram and the two Gaussian fits
    ax = axes[i]
    ax.hist(
        column_data,
        bins=30,
        density=True,
        alpha=0.6,
        color="gray",
        edgecolor="black",
        label="Data Histogram",
    )
    ax.plot(x_values, pdf_1, label=f"Gaussian 1 (μ={means[0]:.2f}, σ={stds[0]:.2f})")
    ax.plot(x_values, pdf_2, label=f"Gaussian 2 (μ={means[1]:.2f}, σ={stds[1]:.2f})")
    ax.plot(x_values, pdf_1 + pdf_2, "k--", label="Combined Fit")

    ax.set_title(f'Histogram of "{column}"')
    ax.set_xlabel(f'Values of "{column}"')
    ax.set_ylabel("Density")
    ax.legend()
    ax.grid(True)

plt.tight_layout()
plt.show()
