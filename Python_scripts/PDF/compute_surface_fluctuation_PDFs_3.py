import os
import sys
import h5py
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from glob import glob
from scipy import stats
from scipy.interpolate import interp1d

module_path = "/home/jofre/Members/Eduard/Paper2/Python_scripts/Data_loader"
if module_path not in sys.path:
    sys.path.append(module_path)

# ============================================================================
# Configuration
# ============================================================================

# Base directory containing all batch folders
BASE_DIR = "/home/jofre/Members/Eduard/Paper2/Simulations/NACA_0012_AOA5_Re50000_1716x1662x128/Snapshots/"

# Pattern to match batch directories
BATCH_PATTERN = "batch_*"

# Geometrical data file
GEO_PATH = "/home/jofre/Members/Eduard/Paper2/Simulations/NACA_0012_AOA5_Re50000_1716x1662x128/Geometrical_data/"
GEO_NAME = "3d_NACA0012_Re50000_AoA5_Geometrical_Data.h5"
GEO_FILE = os.path.join(GEO_PATH, GEO_NAME)

# Output directory
OUTPUT_DIR = "/home/jofre/Members/Eduard/Paper2/Simulations/NACA_0012_AOA5_Re50000_1716x1662x128/Mean_data/PDF_analysis/"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Reference parameters
rho_ref = 1.0
u_infty = 1.0
c = 1.0
Re_c = 50000
q_inf = 0.5 * rho_ref * u_infty**2

# ============================================================================
# ANALYSIS PARAMETERS
# ============================================================================

# Chord locations for PDF analysis (x/c values)
X_C_LOCATIONS = [0.5, 0.7, 0.9]

# Number of bins for histogram (PDF)
N_BINS = 250

# ============================================================================
# Load geometrical data
# ============================================================================
print("=" * 70)
print("PDF ANALYSIS OF SURFACE FLUCTUATIONS")
print("=" * 70)
print(f"\nAnalysis configuration:")
print(f"  Chord locations (x/c): {X_C_LOCATIONS}")
print(f"  Number of bins: {N_BINS}")
print(f"  Suction side (upper surface): closest point selected")

print("\n" + "=" * 70)
print("LOADING GEOMETRICAL DATA")
print("=" * 70)

with h5py.File(GEO_FILE, "r") as f:
    interface_points = f["interface_points"][:]
    interface_indices_i = f["interface_indices_i"][:]
    interface_indices_j = f["interface_indices_j"][:]

N_surf = len(interface_points)
print(f"  Number of 2D interface points: {N_surf}")

# Extract coordinates
x_interface = interface_points[:, 0]
y_interface = interface_points[:, 1]
x_over_c = x_interface / c

# Separate upper and lower surfaces
y_mean = np.mean(y_interface)
upper_mask = y_interface > y_mean
lower_mask = ~upper_mask

# ============================================================================
# Find points at specified chord locations
# ============================================================================
print("\n" + "=" * 70)
print("IDENTIFYING CLOSEST POINT ON SUCTION SIDE")
print("=" * 70)

# Dictionary to store indices for each x/c location
point_indices = {}

for x_c_target in X_C_LOCATIONS:
    # Find points on upper surface (suction side)
    upper_indices = np.where(upper_mask)[0]
    
    if len(upper_indices) == 0:
        print(f"  x/c = {x_c_target:.2f}: No points on upper surface!")
        continue
    
    # Find the closest point to target x/c
    distances = np.abs(x_over_c[upper_indices] - x_c_target)
    closest_idx_in_upper = np.argmin(distances)
    closest_global_idx = upper_indices[closest_idx_in_upper]
    
    actual_x_c = x_over_c[closest_global_idx]
    actual_y = y_interface[closest_global_idx]
    
    point_indices[x_c_target] = {
        'indices': np.array([closest_global_idx]),  # Global index of closest point
        'x_c_actual': actual_x_c,                   # Actual x/c (might not be exactly x/c_target)
        'y': actual_y,                              # y-coordinate of this point
    }
    
    print(f"  x/c = {x_c_target:.2f}: index {closest_global_idx} at actual x/c = {actual_x_c:.4f}")

if len(point_indices) == 0:
    raise RuntimeError("No points found at any specified chord locations!")

# ============================================================================
# Find all surface data files
# ============================================================================
print("\n" + "=" * 70)
print("SEARCHING FOR SURFACE DATA FILES")
print("=" * 70)

batch_dirs = sorted(glob(os.path.join(BASE_DIR, BATCH_PATTERN)))
print(f"  Found {len(batch_dirs)} batch directories")

all_surface_files = []
for batch_dir in batch_dirs:
    surface_dir = os.path.join(batch_dir, "Surface_data")
    if not os.path.exists(surface_dir):
        continue
    
    surface_files = sorted(glob(os.path.join(surface_dir, "surface_*.h5")))
    all_surface_files.extend(surface_files)

N_total_snapshots = len(all_surface_files)
print(f"  Total surface data files: {N_total_snapshots}")

if N_total_snapshots == 0:
    raise RuntimeError("No surface data files found!")

# ============================================================================
# Load all snapshots and compute mean (for fluctuations)
# ============================================================================
print("\n" + "=" * 70)
print("COMPUTING TIME-AVERAGED 2D FIELDS")
print("=" * 70)

p_w_2d_sum = None
tau_w_2d_sum = None
p_w_2_2d_sum = None
tau_w_2_2d_sum = None
n_snapshots = 0

print(f"Loading {N_total_snapshots} snapshots to compute mean...")

for idx, surface_file in enumerate(all_surface_files):
    if (idx + 1) % 20 == 0 or idx == 0:
        print(f"  Loading snapshot {idx+1}/{N_total_snapshots}...", flush=True)
    
    try:
        with h5py.File(surface_file, "r") as f:

            p_w = f["p_w"][:] + f.attrs["p_bulk"]   # (Nz_phys, N_surf)
            tau_w = f["tau_w"][:]    # (Nz_phys, N_surf)
            
            p_w_2 = p_w * p_w
            tau_w_2 = tau_w * tau_w
            
            # Spanwise average for each snapshot
            p_w_2d = np.mean(p_w, axis=0)   # (N_surf,)
            tau_w_2d = np.mean(tau_w, axis=0)       # (N_surf,)
            p_w_2_2d = np.mean(p_w_2, axis=0)   # (N_surf,)
            tau_w_2_2d = np.mean(tau_w_2, axis=0)       # (N_surf,)
            

            if p_w_2d_sum is None:
                p_w_2d_sum = p_w_2d.copy()
                tau_w_2d_sum = tau_w_2d.copy()
                p_w_2_2d_sum = p_w_2_2d.copy()
                tau_w_2_2d_sum = tau_w_2_2d.copy()
                Nz_phys = p_w.shape[0]  # Store for later
            else:
                p_w_2d_sum += p_w_2d
                tau_w_2d_sum += tau_w_2d
                p_w_2_2d_sum += p_w_2_2d
                tau_w_2_2d_sum += tau_w_2_2d
            
            n_snapshots += 1
            
    except Exception as e:
        print(f"  [WARNING] Error loading {surface_file}: {e}")
        continue

if n_snapshots == 0:
    raise RuntimeError("No valid snapshots loaded; check surface files and datasets.")

# Compute 2D time-averaged means
p_w_mean = p_w_2d_sum / n_snapshots      # (N_surf,)
tau_w_mean = tau_w_2d_sum / n_snapshots  # (N_surf,)
p_w_2_mean = p_w_2_2d_sum / n_snapshots      # (N_surf,)
tau_w_2_mean = tau_w_2_2d_sum / n_snapshots  # (N_surf,)

print(f"  Successfully loaded {n_snapshots} snapshots")
print(f"  2D mean shape: (N_surf={len(p_w_mean)})")
print(f"  Spanwise planes in each snapshot: Nz={Nz_phys}")

# ============================================================================
# Collect surface data at each chord location
# ============================================================================
print("\n" + "=" * 70)
print("COLLECTING SURFACE DATA")
print("=" * 70)

surface_data = {}

for x_c_target in point_indices.keys():
    surface_data[x_c_target] = {
        'p_w': [],
        'tau_w': []
    }

print(f"Processing {n_snapshots} snapshots for surface data...")

for idx, surface_file in enumerate(all_surface_files[:n_snapshots]):
    if (idx + 1) % 20 == 0 or idx == 0:
        print(f"  Processing snapshot {idx+1}/{n_snapshots}...", flush=True)
    
    try:
        with h5py.File(surface_file, "r") as f:
            p_w = f["p_w"][:] + f.attrs["p_bulk"]   # (Nz_phys, N_surf)
            tau_w = f["tau_w"][:]  # (Nz_phys, N_surf)
        
        # Extract at each chord location
        for x_c_target, point_info in point_indices.items():
            idx_point = point_info['indices'][0]  # Single point
            
            # Extract surface values at all z-locations for this point
            p_at_xc = p_w[:, idx_point]    # (Nz_phys,)
            tau_at_xc = tau_w[:, idx_point]  # (Nz_phys,)
            
            # Store all z-locations as independent samples
            surface_data[x_c_target]['p_w'].extend(p_at_xc)
            surface_data[x_c_target]['tau_w'].extend(tau_at_xc)
        
    except Exception as e:
        print(f"  [WARNING] Error processing {surface_file}: {e}")
        continue


# ============================================================================
# Compute Compute histogram (PDF)
# ============================================================================
print("\n" + "=" * 70)
print("COMPUTING PDFs")
print("=" * 70)

pdf_data = {}

for x_c_target in point_indices.keys():
    p_samples = np.asarray(surface_data[x_c_target]['p_w'])
    tau_samples = np.asarray(surface_data[x_c_target]['tau_w'])
    
    # Compute histogram (PDF) using numpy
    p_hist, p_bin_edges = np.histogram(p_samples, bins=N_BINS, density=True)
    tau_hist, tau_bin_edges = np.histogram(tau_samples, bins=N_BINS, density=True)
    
    # Compute bin centers for plotting
    p_bin_centers = 0.5 * (p_bin_edges[:-1] + p_bin_edges[1:])
    tau_bin_centers = 0.5 * (tau_bin_edges[:-1] + tau_bin_edges[1:])
    
    pdf_data[x_c_target] = {
        'p_bins': p_bin_centers,
        'p_pdf': p_hist,
        'tau_bins': tau_bin_centers,
        'tau_pdf': tau_hist,
    }
    
    # Print summary statistics for this location
    print(f"\n  x/c = {x_c_target:.2f}:")
    print(f"    Pressure PDF:")
    print(f"      Number of samples: {len(p_samples)}")
    print(f"      Data range:     [{np.min(p_samples):.4f}, {np.max(p_samples):.4f}]")
    print(f"      Max PDF value:  {np.max(p_hist):.4f}")
    print(f"    Shear stress PDF:")
    print(f"      Number of samples: {len(tau_samples)}")
    print(f"      Data range:     [{np.min(tau_samples):.4f}, {np.max(tau_samples):.4f}]")
    print(f"      Max PDF value:  {np.max(tau_hist):.4f}")

# ============================================================================
# Plot PDFs for each chord location
# ============================================================================
print("\n" + "=" * 70)
print("PLOTTING PDFs")
print("=" * 70)

# Create a single figure for all PDFs
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

for x_c_target in point_indices.keys():
    pdf_info = pdf_data[x_c_target]
    
    # Plot pressure PDF
    ax1.plot(pdf_info['p_bins'], pdf_info['p_pdf'], label=f"x/c = {x_c_target:.2f}")
    
    # Plot shear stress PDF
    ax2.plot(pdf_info['tau_bins'], pdf_info['tau_pdf'], label=f"x/c = {x_c_target:.2f}")

ax1.set_xlabel("Pressure $p$")
ax1.set_ylabel("PDF")
ax1.grid(True)
ax1.legend()

ax2.set_xlabel("Shear stress $\\tau_w$")
ax2.set_ylabel("PDF")
ax2.grid(True)
ax2.legend()

plt.tight_layout()
# plt.savefig("surface_PDFs.png", dpi=300)
plt.show()

# ============================================================================
# Compute and Plot Normalized Fluctuation PDFs
# ============================================================================
print("\n" + "=" * 70)
print("COMPUTING NORMALIZED FLUCTUATION PDFs")
print("=" * 70)

normalized_pdf_data = {}

for x_c_target in point_indices.keys():
    idx_point = point_indices[x_c_target]['indices'][0]
    
    # Get mean and variance at this point from 2D time-averaged fields
    p_mean_at_xc = p_w_mean[idx_point]
    tau_mean_at_xc = tau_w_mean[idx_point]
    p_2_mean_at_xc = p_w_2_mean[idx_point]
    tau_2_mean_at_xc = tau_w_2_mean[idx_point]
    
    # Compute standard deviation: sqrt(<x²> - <x>²)
    p_std = np.sqrt(p_2_mean_at_xc - p_mean_at_xc**2)
    tau_std = np.sqrt(tau_2_mean_at_xc - tau_mean_at_xc**2)
    
    # Get raw samples
    p_samples = np.asarray(surface_data[x_c_target]['p_w'])
    tau_samples = np.asarray(surface_data[x_c_target]['tau_w'])
    
    # Compute normalized fluctuations: (x - <x>) / std(x)
    p_fluct_norm = (p_samples - p_mean_at_xc) / p_std
    tau_fluct_norm = (tau_samples - tau_mean_at_xc) / tau_std
    
    # Compute histogram (PDF) of normalized fluctuations
    p_hist, p_bin_edges = np.histogram(p_fluct_norm, bins=N_BINS, density=True)
    tau_hist, tau_bin_edges = np.histogram(tau_fluct_norm, bins=N_BINS, density=True)
    
    # Compute bin centers
    p_bin_centers = 0.5 * (p_bin_edges[:-1] + p_bin_edges[1:])
    tau_bin_centers = 0.5 * (tau_bin_edges[:-1] + tau_bin_edges[1:])
    
    normalized_pdf_data[x_c_target] = {
        'p_bins': p_bin_centers,
        'p_pdf': p_hist,
        'tau_bins': tau_bin_centers,
        'tau_pdf': tau_hist,
    }
    
    # Verify normalized fluctuations have mean 0 and unit variance
    p_norm_mean = np.mean(p_fluct_norm)
    p_norm_var = np.var(p_fluct_norm)
    tau_norm_mean = np.mean(tau_fluct_norm)
    tau_norm_var = np.var(tau_fluct_norm)
    
    print(f"\n  x/c = {x_c_target:.2f}:")
    print(f"    <p> = {p_mean_at_xc:.6f}, std(p) = {p_std:.6f}")
    print(f"    <tau_w> = {tau_mean_at_xc:.6f}, std(tau_w) = {tau_std:.6f}")
    print(f"    Normalized p' range: [{np.min(p_fluct_norm):.2f}, {np.max(p_fluct_norm):.2f}]")
    print(f"    Normalized tau_w' range: [{np.min(tau_fluct_norm):.2f}, {np.max(tau_fluct_norm):.2f}]")
    print(f"    --- Verification ---")
    print(f"    Normalized p':    mean = {p_norm_mean:.6f}, var = {p_norm_var:.6f}")
    print(f"    Normalized tau_w': mean = {tau_norm_mean:.6f}, var = {tau_norm_var:.6f}")

# Plot normalized fluctuation PDFs
print("\n" + "=" * 70)
print("PLOTTING NORMALIZED FLUCTUATION PDFs")
print("=" * 70)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
for x_c_target in point_indices.keys():
    pdf_info = normalized_pdf_data[x_c_target]
    
    # Plot normalized pressure fluctuation PDF
    ax1.plot(pdf_info['p_bins'], pdf_info['p_pdf'], label=f"x/c = {x_c_target:.2f}")
    
    # Plot normalized shear stress fluctuation PDF
    ax2.plot(pdf_info['tau_bins'], pdf_info['tau_pdf'], label=f"x/c = {x_c_target:.2f}")
ax1.set_xlabel("Normalized Pressure Fluctuation $p'$")
ax1.set_ylabel("PDF")
ax1.grid(True)
ax1.legend()
ax2.set_xlabel("Normalized Shear Stress Fluctuation $\\tau_w'$")
ax2.set_ylabel("PDF")
ax2.grid(True)
ax2.legend()
plt.tight_layout()
# plt.savefig("normalized_surface_fluctuation_PDFs.png", dpi=300)
plt.show()
