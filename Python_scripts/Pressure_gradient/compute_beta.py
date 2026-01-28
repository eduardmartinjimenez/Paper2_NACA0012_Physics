import os
import sys
import h5py
import numpy as np

module_path = "/home/jofre/Members/Eduard/Paper2/Python_scripts/Data_loader"

if module_path not in sys.path:
    sys.path.append(module_path)

from data_loader_functions import CompressedSnapshotLoader

### SAVE RESULTS
# Save variables: beta
SAVE_DIR = "/home/jofre/Members/Eduard/Paper2/Simulations/NACA_0012_AOA12_Re50000_1716x1662x128/Mean_data/"
SAVE_NAME = "3d_NACA0012_Re50000_AoA12_Beta_19680000.h5"
SAVE_PATH = os.path.join(SAVE_DIR, SAVE_NAME)

# Create directory if it doesn't exist
os.makedirs(SAVE_DIR, exist_ok=True)

# Load Geometrical data
### GEOMETRICAL STUFF
GEO_PATH = "/home/jofre/Members/Eduard/Paper2/Simulations/NACA_0012_AOA12_Re50000_1716x1662x128/Geometrical_data"
GEO_NAME = "3d_NACA0012_Re50000_AoA12_Geometrical_Data.h5"
GEO_FILE = os.path.join(GEO_PATH, GEO_NAME)

# Check if the Geometrical data file exists
if os.path.exists(GEO_FILE):
    print(f"Data file exists! {GEO_FILE}")
else:
    print(f"Data file does not exist: {GEO_FILE}")

# Load geometrical data
with h5py.File(GEO_FILE, "r") as f:
    interface_points = f["interface_points"][:]
    proj_points = f["proj_points"][:]
    proj_normals = f["proj_normals"][:]
    proj_distances = f["proj_distances"][:]
    interface_indices_i = f["interface_indices_i"][:]
    interface_indices_j = f["interface_indices_j"][:]

# Load Cp data
CP_PATH = "/home/jofre/Members/Eduard/Paper2/Simulations/NACA_0012_AOA12_Re50000_1716x1662x128/Mean_data/"
CP_NAME = "3d_NACA0012_Re50000_AoA12_Cp_19680000.h5"
CP_FILE = os.path.join(CP_PATH, CP_NAME)

if os.path.exists(CP_FILE):
    print(f"Cp data file exists! {CP_FILE}")
else:
    print(f"Cp data file does not exist: {CP_FILE}")

with h5py.File(CP_FILE, "r") as f:
    Cp_values = f["Cp_values"][:]
print("Loaded Cp data.")

# Load Cf data
CF_PATH = "/home/jofre/Members/Eduard/Paper2/Simulations/NACA_0012_AOA12_Re50000_1716x1662x128/Mean_data/"
CF_NAME = "3d_NACA0012_Re50000_AoA12_Cf_19680000.h5"
CF_FILE = os.path.join(CF_PATH, CF_NAME)

if os.path.exists(CF_FILE):
    print(f"Cf data file exists! {CF_FILE}")
else:
    print(f"Cf data file does not exist: {CF_FILE}")

with h5py.File(CF_FILE, "r") as f:
    Cf_values = f["Cf_values"][:]
    tau_w = f["tau_wall"][:]
print("Loaded Cf and wall shear stress data.")

# Load Mesh data
MESH_PATH = "/home/jofre/Members/Eduard/Paper2/Simulations/NACA_0012_AOA12_Re50000_1716x1662x128/Geometrical_data/"
MESH_NAME = "3d_NACA0012_Re50000_AoA12-CROP-MESH.h5"
MESH_FILE = os.path.join(MESH_PATH, MESH_NAME)

if os.path.exists(MESH_FILE):
    print(f"Mesh file exists! {MESH_FILE}")
else:
    print(f"Mesh file does not exist: {MESH_FILE}")

# Initialize loader
loader = CompressedSnapshotLoader(MESH_FILE)

# Load snapshot fields
SNAPSHOT_PATH = "/home/jofre/Members/Eduard/Paper2/Simulations/NACA_0012_AOA12_Re50000_1716x1662x128/temporal_last_snapshot/"
SNAPSHOT_NAME = "3d_NACA0012_Re50000_AoA12_avg_19680000-COMP-DATA.h5"
SNAPSHOT_FILE = os.path.join(SNAPSHOT_PATH, SNAPSHOT_NAME)

if os.path.exists(SNAPSHOT_FILE):
    print(f"Data file exists! {SNAPSHOT_FILE}")
else:
    print(f"Data file does not exist: {SNAPSHOT_FILE}")

fields_avg = loader.load_snapshot_avg(SNAPSHOT_FILE)

# Reference parameters
rho_ref = 1.0  # Reference density [kg/m3]
u_infty = 1.0  # Free-stream velocity [m/s]
c = 1.0  # Airfoil chord length [m]
Re_c = 50000  # Reynolds number [-]
mu_ref = rho_ref * u_infty * c / Re_c  # Dynamic viscosity [Pa s]
nu_ref = mu_ref / rho_ref  # Kinetic viscosity [m2/s]

print(f"Reference parameters:")
print(f"  rho_ref = {rho_ref} [kg/m3]")
print(f"  u_infty = {u_infty} [m/s]")
print(f"  c = {c} [m]")
print(f"  Re_c = {Re_c}")
print(f"  mu_ref = {mu_ref} [Pa s]")

# COMPUTE PRESSURE GRADIENT
# Extract x and y coordinates of projection points
x_proj = proj_points[:, 0]
y_proj = proj_points[:, 1]
x_over_c = x_proj / c

# Compute centroid of all projected points
centroid_x = np.mean(x_proj)
centroid_y = np.mean(y_proj)

# Compute polar angles relative to the centroid
angles = np.arctan2(y_proj - centroid_y, x_proj - centroid_x)

# Sort indices by angle to form a closed loop around the airfoil
sorted_indices = np.argsort(angles)
sorted_x = x_proj[sorted_indices]
sorted_y = y_proj[sorted_indices]
sorted_Cp = Cp_values[sorted_indices]
sorted_tau_w = tau_w[sorted_indices]

# Stack into array of shape (N, 2)
sorted_coords = np.stack((sorted_x, sorted_y), axis=1)

# Compute distances to previous and next point in the sorted loop
prev_coords = np.roll(sorted_coords, shift=1, axis=0)
next_coords = np.roll(sorted_coords, shift=-1, axis=0)

# Compute distances along the surface
dist_prev = np.linalg.norm(sorted_coords - prev_coords, axis=1)
dist_next = np.linalg.norm(sorted_coords - next_coords, axis=1)
delta_s = 0.5 * (dist_prev + dist_next)

# Compute dCp/ds using central differences (normalized by surface distance)
dCp_ds_sorted = 0.5 * (np.roll(sorted_Cp, shift=-1) - np.roll(sorted_Cp, shift=1)) / delta_s

# Convert dCp/ds to dP/dx
# Cp = (P - P_bulk) / q_inf, so dCp/ds = (1/q_inf) * dP/ds
# dP/ds = Cp * q_inf * dCp/ds (in terms of local derivative)
# dP/dx = dP/ds * (ds/dx) where ds/dx relates arc length to x-distance

# For x-direction gradient:
dx_sorted = 0.5 * (np.roll(sorted_x, shift=-1) - np.roll(sorted_x, shift=1))
dx_sorted = np.abs(dx_sorted) + 1e-8  # Avoid division by zero

# Pressure gradient dP/dx (dimensional)
q_inf = 0.5 * rho_ref * u_infty**2
dP_dx = dCp_ds_sorted * q_inf * delta_s / dx_sorted

print(f"Pressure gradient computed.")
print(f"dP/dx min: {np.min(dP_dx):.6e}, dP/dx max: {np.max(dP_dx):.6e}")

# ESTIMATE DISPLACEMENT THICKNESS
# Using the relationship: δ* ≈ (Cf / 2)^0.5 * x for boundary layer approximations
# Or use momentum integral: H = δ*/θ where θ is momentum thickness

# Simple approximation using Cf:
# For boundary layers: τ_w = (1/2) * ρ * U_∞ * dU_e/dy at wall
# And displacement thickness relates to Cf through momentum equation

# Using the Thwaites-type relationship:
# Compute a characteristic boundary layer thickness from Cf and tau_w
# δ = tau_w / (ρ * u_infty * Cf) is related to local thickness

# More practical: use correlations between Cf and shape factor H
# For separated and attached flows: δ* can be estimated from H and momentum thickness θ

# Using approximation: δ* ≈ 0.1 * sqrt(x) for attached flow (Blasius-like)
# Better: δ* relates through H (shape factor) where δ* = H * θ
# and θ = (Cf/2) * x (approximate)

# Cumulative approach: integrate momentum thickness and apply shape factor
sorted_x_over_c = sorted_x / c

# Compute cumulative momentum thickness
# For attached flow: θ ≈ sqrt(Cf/2) * x
Cf_sorted = 2.0 * sorted_tau_w / (rho_ref * u_infty**2)
theta_sorted = np.sqrt(np.abs(Cf_sorted) / 2.0) * sorted_x

# Estimate shape factor H (varies from ~1.4 (separation) to ~1.25 (Blasius) to ~1.05 (recovery))
# Use a simple model based on Cp (pressure gradient indicator)
H_sorted = 1.4 - 0.3 * np.tanh(2.0 * sorted_Cp)  # Decreases with favorable pressure gradient

# Compute displacement thickness
delta_star_sorted = H_sorted * theta_sorted

print(f"Displacement thickness computed.")
print(f"δ* min: {np.min(delta_star_sorted):.6e}, δ* max: {np.max(delta_star_sorted):.6e}")

# COMPUTE PRESSURE-GRADIENT PARAMETER β
# β = (δ*/τ_w) * dP/dx = (δ* * c / (μ * U_∞²)) * dP/dx

# Avoid division by zero
sorted_tau_w_safe = np.where(np.abs(sorted_tau_w) < 1e-10, 1e-10, sorted_tau_w)

# Compute beta
beta_sorted = (delta_star_sorted / sorted_tau_w_safe) * dP_dx

# Alternative form: β = (δ* * c) / (μ * U_∞²) * dP/dx (dimensionally consistent)
beta_sorted_alt = (delta_star_sorted * c) / (mu_ref * u_infty**2) * dP_dx

print(f"\n" + "="*60)
print("PRESSURE-GRADIENT PARAMETER β STATISTICS")
print("="*60)
print(f"β min:  {np.min(beta_sorted):.6e}")
print(f"β max:  {np.max(beta_sorted):.6e}")
print(f"β mean: {np.mean(beta_sorted):.6e}")
print(f"β std:  {np.std(beta_sorted):.6e}")
print(f"\nThwaites' separation criterion: β ≈ -0.126")
print(f"Points with β < -0.126 (potential separation): {np.sum(beta_sorted < -0.126)}")
print("="*60 + "\n")

# Map back to original order
beta_values = np.zeros_like(beta_sorted)
beta_values[sorted_indices] = beta_sorted

# SAVE RESULTS TO HDF5 FILE
with h5py.File(SAVE_PATH, "w") as f:
    f.create_dataset("beta_values", data=beta_values)
    f.create_dataset("delta_star", data=delta_star_sorted)
    f.create_dataset("theta", data=theta_sorted)
    f.create_dataset("H_shape_factor", data=H_sorted)
    f.create_dataset("dP_dx", data=dP_dx)
    f.create_dataset("dCp_ds", data=dCp_ds_sorted)
    f.create_dataset("interface_indices_i", data=interface_indices_i)
    f.create_dataset("interface_indices_j", data=interface_indices_j)
    f.create_dataset("x_over_c", data=sorted_x_over_c)

print(f"Beta values saved to {SAVE_PATH}")

# LOAD AND CHECK SAVED VALUES
with h5py.File(SAVE_PATH, "r") as f:
    beta_loaded = f["beta_values"][:]
    delta_star_loaded = f["delta_star"][:]

print("Loaded beta values from file. Number of points:", beta_loaded.shape[0])
print(f"Verification - β min: {np.min(beta_loaded):.6e}, β max: {np.max(beta_loaded):.6e}")
print(f"Verification - δ* min: {np.min(delta_star_loaded):.6e}, δ* max: {np.max(delta_star_loaded):.6e}")
