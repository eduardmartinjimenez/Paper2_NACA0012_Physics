import os
import sys
import h5py
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from glob import glob

module_path = "/home/jofre/Members/Eduard/Paper2/Python_scripts/Data_loader"
if module_path not in sys.path:
    sys.path.append(module_path)

from data_loader_functions import CompressedSnapshotLoader

# ============================================================================
# Configuration
# ============================================================================
# Mean shear stress data file
MEAN_DATA_DIR = "/home/jofre/Members/Eduard/Paper2/Simulations/NACA_0012_AOA12_Re50000_1716x1662x128/Mean_data/Mean_Shear_Stress/"
MEAN_DATA_NAME = "mean_wall_shear_stress.h5"

# Geometrical data file (to get interface points and chord positions)
GEO_PATH = "/home/jofre/Members/Eduard/Paper2/Simulations/NACA_0012_AOA12_Re50000_1716x1662x128/Geometrical_data/"
GEO_NAME = "3d_NACA0012_Re50000_AoA12_Geometrical_Data.h5"
GEO_FILE = os.path.join(GEO_PATH, GEO_NAME)

# Reference parameters
rho_ref = 1.0   # Reference density [kg/m3]
u_infty = 1.0   # Free-stream velocity [m/s]
c = 1.0         # Airfoil chord length [m]
Re_c = 50000    # Reynolds number [-]
q_inf = 0.5 * rho_ref * u_infty**2  # Dynamic pressure [Pa]

AOA = 12  # Angle of attack [degrees]

# ============================================================================
# Utilities
# ============================================================================

def assert_exists(path: str, kind: str = "File") -> None:
    if not os.path.exists(path):
        raise FileNotFoundError(f"{kind} does not exist: {path}")
    print(f"  [OK] {kind}: {path}")

# ============================================================================
# Load geometrical data
# ============================================================================
print("=" * 70)
print("LOADING GEOMETRICAL DATA")
print("=" * 70)

assert_exists(GEO_FILE, "Geometrical data")

with h5py.File(GEO_FILE, "r") as f:
    interface_points = f["interface_points"][:]    # (N_surf, 3)
    interface_indices_i = f["interface_indices_i"][:]
    interface_indices_j = f["interface_indices_j"][:]

N_surf = len(interface_points)
print(f"  Number of 2D interface points: {N_surf}")

# Extract x, y coordinates for chord-wise analysis
x_interface = interface_points[:, 0]  # (N_surf,)
y_interface = interface_points[:, 1]  # (N_surf,)

# Compute x/c for each point
x_over_c = x_interface / c

# ============================================================================
# load menan wall shear stress data
# ============================================================================
print("\n" + "=" * 70)
print("LOADING MEAN WALL SHEAR STRESS DATA")
print("=" * 70)

mean_data_file = os.path.join(MEAN_DATA_DIR, MEAN_DATA_NAME)
assert_exists(mean_data_file, "Mean wall shear stress data")

with h5py.File(mean_data_file, "r") as f:
    tau_w_mean = f["tau_w_mean"][:]  # (N_surf,)

print(f"  Loaded mean wall shear stress data shape: {tau_w_mean.shape}")

# ============================================================================
# Compute Cf
# ============================================================================
print(f"  tau_w range: [{np.min(tau_w_mean):.6e}, {np.max(tau_w_mean):.6e}]")  # ADD THIS

# Compute skin friction coefficient
# Cf = tau_w / q_inf
Cf_values = tau_w_mean / q_inf  # ADD THIS

# ============================================================================
# Organize by chord position and separate upper/lower surfaces
# ============================================================================
print("\n" + "=" * 70)
print("ORGANIZING DATA BY CHORD POSITION")
print("=" * 70)

# Separate upper and lower surfaces
# Use y-coordinate: upper surface has larger y, lower has smaller y
y_mean = np.mean(y_interface)
upper_mask = y_interface > y_mean
lower_mask = ~upper_mask

# Upper surface
x_c_upper = x_over_c[upper_mask]
Cf_upper = Cf_values[upper_mask]  
y_upper = y_interface[upper_mask]

# Sort by x/c
sort_idx_upper = np.argsort(x_c_upper)
x_c_upper = x_c_upper[sort_idx_upper]
Cf_upper = Cf_upper[sort_idx_upper] 
y_upper = y_upper[sort_idx_upper]

# Lower surface
x_c_lower = x_over_c[lower_mask]
Cf_lower = Cf_values[lower_mask] 
y_lower = y_interface[lower_mask]

# Sort by x/c
sort_idx_lower = np.argsort(x_c_lower)
x_c_lower = x_c_lower[sort_idx_lower]
Cf_lower = Cf_lower[sort_idx_lower]  
y_lower = y_lower[sort_idx_lower]

print(f"  Upper surface points: {len(x_c_upper)}")
print(f"  Lower surface points: {len(x_c_lower)}")

# ============================================================================
# Create plots
# ============================================================================
print("\n" + "=" * 70)
print("GENERATING PLOTS")
print("=" * 70)

# Plot 2: Cf distribution along chord
fig, ax = plt.subplots(figsize=(12, 7))

ax.plot(x_c_upper, Cf_upper, 'b-o', label='Upper surface', markersize=4, linewidth=1.5)
ax.plot(x_c_lower, Cf_lower, 'r-s', label='Lower surface', markersize=4, linewidth=1.5)
ax.axhline(y=0, color='k', linestyle='--', linewidth=0.8, alpha=0.5)

ax.set_xlabel('x/c', fontsize=14)
ax.set_ylabel('$C_f$', fontsize=14)
ax.set_title(f'Skin Friction Coefficient Distribution - NACA 0012\n' + 
             f'AoA = {AOA}°, Re = {Re_c:,}',
             fontsize=15)
ax.legend(loc='best', fontsize=12)
ax.grid(True, alpha=0.3, linestyle='--')

plt.tight_layout()

plt.show()
