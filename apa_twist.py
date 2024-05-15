# Re-import necessary libraries and re-define the data and functions
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata

# Data with middle and top supports
data_b_side_top = {
    "x (mm)": [1100, 1100, 1100, 
               6930, 6930, 6930, 
               4015, 4015, 4015],
    "y (mm)": [192.1, 2486.1, 1337, 
               194.7, 2489.5, 1340, 
               191.3, 2486.7, 1337],
    "z (inches)": [0.405, 0.215, 0.295, 
                   0.63, 0.205, 0.385, 
                   0.505, 0.062, 0.22] 
}



data_a_side_top = {
    "x (mm)": [1100, 1100, 1100, 
               6930, 6930, 6930, 
               4015, 4015, 4015],
    "y (mm)": [193, 2482.1, 1337, 
               196, 2486, 1340.6, 
               191.3, 2483.4, 1337.6],
    "z (inches)": [-0.415, -0.615, -0.51, 
                   -0.18, -0.605, -0.395, 
                   -0.3, -0.71, -0.5]
}

#Data with middle and bottom supports

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

df1 = pd.DataFrame(data_b_side_top)
df2 = pd.DataFrame(data_a_side_top)

# Define the grid for interpolation
xi = np.linspace(min(df1['x (mm)'].min(), df2['x (mm)'].min()), max(df1['x (mm)'].max(), df2['x (mm)'].max()), 100)
yi = np.linspace(min(df1['y (mm)'].min(), df2['y (mm)'].min()), max(df1['y (mm)'].max(), df2['y (mm)'].max()), 100)
xi, yi = np.meshgrid(xi, yi)

# Interpolate z-values on the grid for both datasets
zi1 = griddata((df1['x (mm)'], df1['y (mm)']), df1['z (inches)'], (xi, yi), method='cubic')
zi2 = griddata((df2['x (mm)'], df2['y (mm)']), df2['z (inches)'], (xi, yi), method='cubic')

# Plotting
fig = plt.figure(figsize=(14, 7))

# Surface 1 (inverted z-values)
ax1 = fig.add_subplot(121, projection='3d')
surf1 = ax1.plot_surface(xi, yi, zi1, cmap='viridis', edgecolor='none')
fig.colorbar(surf1, ax=ax1, shrink=0.5, aspect=5).set_label('Inverted Height (inches)')
ax1.set_title('Surface 1 (Inverted Z)')
ax1.set_xlabel('X (mm)')
ax1.set_ylabel('Y (mm)')
ax1.set_zlabel('Z (inches)')

# Surface 2
ax2 = fig.add_subplot(122, projection='3d')
surf2 = ax2.plot_surface(xi, yi, zi2, cmap='plasma', edgecolor='none')
fig.colorbar(surf2, ax=ax2, shrink=0.5, aspect=5).set_label('Height (inches)')
ax2.set_title('Surface 2')
ax2.set_xlabel('X (mm)')
ax2.set_ylabel('Y (mm)')
ax2.set_zlabel('Z (inches)')

plt.show()


# Combined plotting in the same 3D plot
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Surface 1 (inverted z-values)
surf1 = ax.plot_surface(xi, yi, zi1, cmap='viridis', alpha=0.6, edgecolor='none', label='Surface 1 (Inverted)')
# Surface 2
surf2 = ax.plot_surface(xi, yi, zi2, cmap='plasma', alpha=0.6, edgecolor='none', label='Surface 2')

# Adding colorbars and labels
cbar1 = fig.colorbar(surf1, ax=ax, shrink=0.5, aspect=10, pad=0.1)
cbar1.set_label('Inverted Height (inches) - Surface 1')
cbar2 = fig.colorbar(surf2, ax=ax, shrink=0.5, aspect=10, pad=0.1)
cbar2.set_label('Height (inches) - Surface 2')

ax.set_xlabel('X (mm)')
ax.set_ylabel('Y (mm)')
ax.set_zlabel('Z (inches)')
ax.set_title('Combined 3D Surface Plot')

plt.show()
