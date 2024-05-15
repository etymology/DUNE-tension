import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.interpolate import griddata
from sklearn.decomposition import PCA

def main():
    # Define the data
    data_b_side_bottom = {
        "x (mm)": [1100, 1100, 1100, 6930, 6930, 6930, 4015, 4015, 4015],
        "y (mm)": [191.8, 2485.9, 1336.2, 195, 2489.6, 1339.6, 196, 2486.5, 1336.5],
        "z (inches)": [0.391, 0.125, 0.210, 0.608, 0.132, 0.341, 0.5, -0.02, 0.2]
    }

    # Convert the data to a DataFrame
    df = pd.DataFrame(data_b_side_bottom)

    # Convert z-values from inches to millimeters
    df['z (mm)'] = df['z (inches)'] * 25.4

    # Prepare data for PCA
    X = df[['x (mm)', 'y (mm)', 'z (mm)']].values

    # Apply PCA
    pca = PCA()
    X_pca = pca.fit_transform(X)

    # Rotate data based on PCA components
    df_pca = pd.DataFrame(X_pca, columns=['PC1', 'PC2', 'PC3'])

    # Create a grid for interpolation of the first two principal components
    xi = np.linspace(df_pca['PC1'].min(), df_pca['PC1'].max(), 100)
    yi = np.linspace(df_pca['PC2'].min(), df_pca['PC2'].max(), 100)
    xi, yi = np.meshgrid(xi, yi)

    # Interpolate z-values (now 'PC3') on this new grid
    zi = griddata((df_pca['PC1'], df_pca['PC2']), df_pca['PC3'], (xi, yi), method='cubic')

    # Plotting
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    z_scale = 300
    # Set the aspect ratio based on original data dimensions
    x_range = df['x (mm)'].max() - df['x (mm)'].min()
    y_range = df['y (mm)'].max() - df['y (mm)'].min()
    z_range = df_pca['PC3'].max() - df_pca['PC3'].min()
    max_range = np.array([x_range, y_range, z_range*z_scale]).max()

    ax.set_box_aspect([x_range / max_range, y_range / max_range, z_range*z_scale / max_range])

    surf = ax.plot_surface(xi, yi, zi, cmap='viridis', edgecolor='none')

    # Colorbar and labels
    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5).set_label('Height along new Z axis (mm)')
    ax.set_xlabel('PC1')
    ax.set_ylabel('PC2')
    ax.set_zlabel('PC3 (New Z)')
    ax.set_title('3D Surface Plot of B Side Bottom with PCA Rotation')

    # Show the plot
    plt.show()

if __name__ == "__main__":
    main()
