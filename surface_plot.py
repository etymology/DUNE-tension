import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.interpolate import griddata

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

    # Adjust z values to have zero mean (optional step)
    # df['z (mm)'] -= df['z (mm)'].mean()

    # Create a grid for interpolation
    xi = np.linspace(df['x (mm)'].min(), df['x (mm)'].max(), 100)
    yi = np.linspace(df['y (mm)'].min(), df['y (mm)'].max(), 100)
    xi, yi = np.meshgrid(xi, yi)

    # Interpolate z-values
    zi = griddata((df['x (mm)'], df['y (mm)']), df['z (mm)'], (xi, yi), method='cubic')

    # Plotting
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(xi, yi, zi, cmap='viridis', edgecolor='none')

    # Setting the aspect ratio
    x_range = np.abs(df['x (mm)'].max() - df['x (mm)'].min())
    y_range = np.abs(df['y (mm)'].max() - df['y (mm)'].min())
    z_range = np.abs(df['z (mm)'].max() - df['z (mm)'].min())
    max_range = np.array([x_range, y_range, z_range * 100]).max()

    ax.set_box_aspect([x_range / max_range, y_range / max_range, (z_range * 100) / max_range])

    # Colorbar and labels
    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5).set_label('Height (mm)')
    ax.set_xlabel('X (mm)')
    ax.set_ylabel('Y (mm)')
    ax.set_zlabel('Z (mm)')
    ax.set_title('3D Surface Plot of B Side Bottom with Z in Millimeters')

    # Show the plot
    plt.show()

if __name__ == "__main__":
    main()
