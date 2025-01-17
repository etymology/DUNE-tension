import pandas as pd
import numpy as np
from sklearn.mixture import GaussianMixture
import matplotlib.pyplot as plt

# Load the uploaded CSV file to inspect its contents
file_path = 'jig_data.csv'
data = pd.read_csv(file_path)



# Calculate pairwise differences, excluding rows with NaN or 0 in any column
columns = data.columns
pairwise_differences = {}

for i, col1 in enumerate(columns):
    for j, col2 in enumerate(columns):
        if i < j:
            # Create a name for the pairwise difference
            diff_name = f"{col1}_minus_{col2}"
            # Compute differences while avoiding rows with NaN or 0
            valid_rows = (data[col1].notna() & data[col2].notna() & (data[col1] != 0) & (data[col2] != 0))
            pairwise_differences[diff_name] = data.loc[valid_rows, col1] - data.loc[valid_rows, col2]

# Convert to a DataFrame
differences_df = pd.DataFrame(pairwise_differences)

# Fit and plot histograms with Gaussian Mixture Models for each difference
gaussian_params = []
for col in differences_df.columns:
    values = differences_df[col].dropna()
    if not values.empty:
        # Fit a Gaussian Mixture Model (2 components)
        gmm = GaussianMixture(n_components=2, random_state=42)
        gmm.fit(values.values.reshape(-1, 1))
        
        # Extract GMM parameters
        means = gmm.means_.flatten()
        std_devs = np.sqrt(gmm.covariances_).flatten()
        weights = gmm.weights_
        gaussian_params.append({
            "Difference": col,
            "Mean_1": means[0], "Std_Dev_1": std_devs[0], "Weight_1": weights[0],
            "Mean_2": means[1], "Std_Dev_2": std_devs[1], "Weight_2": weights[1],
        })
        
        # Plot histogram with GMM fit
        x = np.linspace(values.min(), values.max(), 500).reshape(-1, 1)
        gmm_pdf = np.exp(gmm.score_samples(x))
        component_pdfs = [weights[i] * np.exp(-0.5 * ((x - means[i]) / std_devs[i])**2) / 
                          (std_devs[i] * np.sqrt(2 * np.pi)) for i in range(2)]
        
        plt.figure()
        plt.hist(values, bins=30, density=True, alpha=0.6, label="Histogram")
        plt.plot(x, gmm_pdf, label="GMM Fit", lw=2)
        plt.plot(x, component_pdfs[0], '--', label="Component 1")
        plt.plot(x, component_pdfs[1], '--', label="Component 2")
        plt.title(f"Histogram and GMM Fit: {col}")
        plt.xlabel("Difference")
        plt.ylabel("Density")
        plt.legend()

# Save Gaussian parameters to CSV
params_df = pd.DataFrame(gaussian_params)
params_file_path = 'gaussian_params.csv'
params_df.to_csv(params_file_path, index=False)

# Show histograms to user
plt.tight_layout()

plt.show()

# Provide the file path to the user
params_file_path
