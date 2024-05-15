# Re-importing necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.interpolate import griddata

# Re-defining the data for B Side Bottom and B Side Top
data_b_side_bottom = {
    "x (mm)": [1100, 1100, 1100, 
               6930, 6930, 6930, 
               4015, 4015, 4015],
    "y (mm)": [191.8, 2485.9, 1336.2, 
               195, 2489.6, 1339.6, 
               196, 2486.5, 1336.5],
    "z (inches)": [0.391, 0.125, 0.210, 
                   0.608, 0.132, 0.341, 
                   0.5, -0.02, 0.2]
}
data_b_side_top = {
    "x (mm)": [1100, 1100, 1100, 6930, 6930, 6930, 4015, 4015, 4015],
    "y (mm)": [192.1, 2486.1, 1337, 194.7, 2489.5, 1340, 191.3, 2486.7, 1337],
    "z (inches)": [0.405, 0.215, 0.295, 0.63, 0.205, 0.385, 0.505, 0.062, 0.22]
}



df_b_bottom = pd.DataFrame(data_b_side_bottom)
df_b_top = pd.DataFrame(data_b_side_top)

# Define the grid for interpolation
xi = np.linspace(min(df_b_bottom['x (mm)'].min(), df_b_top['x (mm)'].min()), max(df_b_bottom['x (mm)'].max(), df_b_top['x (mm)'].max()), 100)
yi = np.linspace(min(df_b_bottom['y (mm)'].min(), df_b_top['y (mm)'].min()), max(df_b_bottom['y (mm)'].max(), df_b_top['y (mm)'].max()), 100)
xi, yi = np.meshgrid(xi, yi)

# Interpolate z-values on the grid for both new datasets
zi_b_bottom = griddata((df_b_bottom['x (mm)'], df_b_bottom['y (mm)']), df_b_bottom['z (inches)'], (xi, yi), method='cubic')
zi_b_top = griddata((df_b_top['x (mm)'], df_b_top['y (mm)']), df_b_top['z (inches)'], (xi, yi), method='cubic')

# Plotting both surfaces together
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Surface for B Side Bottom
surf_b_bottom = ax.plot_surface(xi, yi, zi_b_bottom, cmap='coolwarm', alpha=0.6, edgecolor='none')
# Surface for B Side Top
surf_b_top = ax.plot_surface(xi, yi, zi_b_top, cmap='winter', alpha=0.6, edgecolor='none')

# Adding colorbars and labels
cbar_b_bottom = fig.colorbar(surf_b_bottom, ax=ax, shrink=0.5, aspect=10, pad=0.1)
cbar_b_bottom.set_label('Height (inches) - B Side Bottom')
cbar_b_top = fig.colorbar(surf_b_top, ax=ax, shrink=0.5, aspect=10, pad=0.1)
cbar_b_top.set_label('Height (inches) - B Side Top')

ax.set_xlabel('X (mm)')
ax.set_ylabel('Y (mm)')
ax.set_zlabel('Z (inches)')
ax.set_title('Combined 3D Surface Plot of B Sides')

plt.show()


# Calculate the difference between the z-values of B Side Top and B Side Bottom
zi_difference = zi_b_top - zi_b_bottom

# Plotting the difference
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Surface for the difference
surf_difference = ax.plot_surface(xi, yi, zi_difference, cmap='coolwarm', alpha=0.8, edgecolor='none')

# Adding a colorbar and labels
cbar = fig.colorbar(surf_difference, ax=ax, shrink=0.5, aspect=10)
cbar.set_label('Height Difference (inches)')

ax.set_xlabel('X (mm)')
ax.set_ylabel('Y (mm)')
ax.set_zlabel('Z Difference (inches)')
ax.set_title('3D Surface Plot of Height Differences Between B Sides')

plt.show()


# Adjusting plot to maintain scale in x and y directions
fig = plt.figure(figsize=(12, 10))
ax = fig.add_subplot(111, projection='3d')

# Surface for B Side Bottom
surf_b_bottom = ax.plot_surface(xi, yi, zi_b_bottom, cmap='coolwarm', alpha=0.6, edgecolor='none')
# Surface for B Side Top
surf_b_top = ax.plot_surface(xi, yi, zi_b_top, cmap='winter', alpha=0.6, edgecolor='none')

# Setting the aspect ratio based on the ranges of x and y to reflect actual dimensions
x_range = np.abs(df_b_bottom['x (mm)'].max() - df_b_bottom['x (mm)'].min())
y_range = np.abs(df_b_bottom['y (mm)'].max() - df_b_bottom['y (mm)'].min())
z_range = np.abs(max(df_b_bottom['z (inches)'].max(), df_b_top['z (inches)'].max()) - 
                 min(df_b_bottom['z (inches)'].min(), df_b_top['z (inches)'].min()))

# The largest range is used to set aspect ratio for all axes equally
max_range = np.array([x_range, y_range, z_range]).max()
ax.set_box_aspect([x_range / max_range, y_range / max_range, z_range / max_range])  # Normalizing the aspect ratio

# Adding colorbars and labels
cbar_b_bottom = fig.colorbar(surf_b_bottom, ax=ax, shrink=0.5, aspect=10, pad=0.1)
cbar_b_bottom.set_label('Height (inches) - B Side Bottom')
cbar_b_top = fig.colorbar(surf_b_top, ax=ax, shrink=0.5, aspect=10, pad=0.1)
cbar_b_top.set_label('Height (inches) - B Side Top')

ax.set_xlabel('X (mm)')
ax.set_ylabel('Y (mm)')
ax.set_zlabel('Z (inches)')
ax.set_title('Combined 3D Surface Plot of B Sides to Scale')

plt.show()

# Adjusting z scale by increasing it 1000 times relative to x and y
fig = plt.figure(figsize=(12, 10))
ax = fig.add_subplot(111, projection='3d')

# Surface for B Side Bottom
surf_b_bottom = ax.plot_surface(xi, yi, zi_b_bottom, cmap='coolwarm', alpha=0.6, edgecolor='none')
# Surface for B Side Top
surf_b_top = ax.plot_surface(xi, yi, zi_b_top, cmap='winter', alpha=0.6, edgecolor='none')

# Setting the adjusted aspect ratio to emphasize the z dimension
ax.set_box_aspect([x_range / max_range, y_range / max_range, (z_range * 5000) / max_range])  # Scaling z-axis by 1000

# Adding colorbars and labels
cbar_b_bottom = fig.colorbar(surf_b_bottom, ax=ax, shrink=0.5, aspect=10, pad=0.1)
cbar_b_bottom.set_label('Height (inches) - B Side Bottom')
cbar_b_top = fig.colorbar(surf_b_top, ax=ax, shrink=0.5, aspect=10, pad=0.1)
cbar_b_top.set_label('Height (inches) - B Side Top')

ax.set_xlabel('X (mm)')
ax.set_ylabel('Y (mm)')
ax.set_zlabel('Z (inches)')
ax.set_title('Combined 3D Surface Plot of B Sides with Enhanced Z Scale')

plt.show()

# Scaling the z-dimension by 5000 times relative to x and y for the specified plot orientation
fig = plt.figure(figsize=(12, 10))
ax = fig.add_subplot(111, projection='3d')

# Surface for B Side Bottom
surf_b_bottom = ax.plot_surface(xi, yi, zi_b_bottom, cmap='coolwarm', alpha=0.6, edgecolor='none')
# Surface for B Side Top
surf_b_top = ax.plot_surface(xi, yi, zi_b_top, cmap='winter', alpha=0.6, edgecolor='none')

# Setting the adjusted aspect ratio to emphasize z dimension
ax.set_box_aspect([x_range / max_range, (y_range * 5) / max_range, (z_range * 5000) / max_range])

# Rotate the plot to make x and z horizontal and y vertical
ax.view_init(elev=90, azim=-90)  # Y vertical

# Adding colorbars and labels
cbar_b_bottom = fig.colorbar(surf_b_bottom, ax=ax, shrink=0.5, aspect=10, pad=0.1)
cbar_b_bottom.set_label('Height (inches) - B Side Bottom')
cbar_b_top = fig.colorbar(surf_b_top, ax=ax, shrink=0.5, aspect=10, pad=0.1)
cbar_b_top.set_label('Height (inches) - B Side Top')

ax.set_xlabel('X (mm)')
ax.set_zlabel('Z (inches)')  # Z is now horizontally scaled by 5000 times
ax.set_ylabel('Y (mm)')  # Y remains vertical
ax.set_title('3D Surface Plot with Y as Vertical Axis and Enhanced Z Scale')

plt.show()
