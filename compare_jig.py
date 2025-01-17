import pandas as pd
import numpy as np
from sklearn.mixture import GaussianMixture
import matplotlib.pyplot as plt

# Load the uploaded file to examine its structure
file_path = "four_passes.csv"
data = pd.read_csv(file_path)



# Drop rows with NaNs in all four columns to simplify pairwise calculations
data_clean = data.dropna(subset=["a1", "a2", "a3", "a4"])

# Calculate pairwise differences while avoiding rows with 0 values
pairwise_differences = {}
columns = ["a1", "a2", "a3", "a4"]
for i, col1 in enumerate(columns):
    for col2 in columns[i + 1 :]:
        diff_name = f"{col1}-{col2}"
        valid_rows = (data_clean[col1] != 0) & (data_clean[col2] != 0)
        pairwise_differences[diff_name] = (
            data_clean.loc[valid_rows, col1] - data_clean.loc[valid_rows, col2]
        )

# Fit Gaussian Mixture Model (GMM) to each set of pairwise differences and create histograms
gmm_results = []
plt.figure(figsize=(20, 15))

for idx, (key, diffs) in enumerate(pairwise_differences.items(), start=1):
    # Reshape data for GMM
    diffs_reshaped = diffs.dropna().values.reshape(-1, 1)

    # Fit GMM with 2 components
    gmm = GaussianMixture(n_components=2, random_state=42)
    gmm.fit(diffs_reshaped)

    # Store GMM parameters
    for i in range(gmm.n_components):
        gmm_results.append(
            {
                "Difference": key,
                "Component": i + 1,
                "Mean": gmm.means_[i, 0],
                "StdDev": np.sqrt(gmm.covariances_[i, 0, 0]),
                "Weight": gmm.weights_[i],
            }
        )

    # Create histogram and plot GMM fit
    plt.subplot(3, 2, idx)
    plt.hist(diffs_reshaped, bins=50, density=True, alpha=0.6, label="Data Histogram")

    x = np.linspace(diffs_reshaped.min(), diffs_reshaped.max(), 500).reshape(-1, 1)
    y_gmm = np.exp(gmm.score_samples(x))
    plt.plot(x, y_gmm, label="Combined GMM Fit")

    # Add individual components
    for i in range(gmm.n_components):
        y_component = gmm.weights_[i] * (
            1
            / (np.sqrt(2 * np.pi * gmm.covariances_[i, 0, 0]))
            * np.exp(-0.5 * ((x - gmm.means_[i]) ** 2) / gmm.covariances_[i, 0, 0])
        )
        plt.plot(x, y_component, linestyle="--", label=f"Component {i + 1}")

    plt.title(f"Histogram and GMM Fit for {key}")
    plt.legend()

# Save GMM parameters to a CSV file
gmm_results_df = pd.DataFrame(gmm_results)
output_csv_path = "gmm_parameters.csv"
gmm_results_df.to_csv(output_csv_path, index=False)

plt.tight_layout()
plt.show()

output_csv_path
