import numpy as np
from stl import mesh
import os
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull

# Define STL file path
MESH_FILE = "/home/jofre/Members/Eduard/Paper2/Simulations/NACA_0012_AOA12_Re50000_1716x1662x128/Geometrical_data/naca0012.stl"

# Import STL file
stl_mesh = mesh.Mesh.from_file(MESH_FILE)

# Extract 2D section at a specific Z coordinate
z_slice = 0.5 
tolerance = 0.01

# Find intersection points
section_points = []
for triangle in stl_mesh.vectors:
    for i in range(3):
        v1 = triangle[i]
        v2 = triangle[(i+1) % 3]
        # Check if edge crosses z_slice
        if (v1[2] - z_slice) * (v2[2] - z_slice) <= 0:
            if abs(v2[2] - v1[2]) > 1e-10:
                t = (z_slice - v1[2]) / (v2[2] - v1[2])
                x = v1[0] + t * (v2[0] - v1[0])
                y = v1[1] + t * (v2[1] - v1[1])
                section_points.append([x, y])

section_points = np.array(section_points)

# Remove duplicates
section_points = np.unique(np.round(section_points, 6), axis=0)

# Sort by angle from centroid (follows boundary)
centroid = np.mean(section_points, axis=0)
angles = np.arctan2(section_points[:, 1] - centroid[1], section_points[:, 0] - centroid[0])
section_points = section_points[np.argsort(angles)]

# Plot the 2D section as a line
plt.figure(figsize=(10, 6))
plt.plot(section_points[:, 0], section_points[:, 1], 'k-', linewidth=0.5)
plt.xlabel('X axis')
plt.ylabel('Y axis')
plt.title(f'2D Section of STL Mesh (Z = {z_slice})')
plt.axis('equal')
plt.grid(True)
plt.show()