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
BASE_DIR = "/home/jofre/Members/Eduard/Paper2/Simulations/NACA_0012_AOA12_Re50000_1716x1662x128/Snapshots/"

# Pattern to match batch directories
BATCH_PATTERN = "batch_*"

# Geometrical data file
GEO_PATH = "/home/jofre/Members/Eduard/Paper2/Simulations/NACA_0012_AOA12_Re50000_1716x1662x128/Geometrical_data/"
GEO_NAME = "3d_NACA0012_Re50000_AoA12_Geometrical_Data.h5"
GEO_FILE = os.path.join(GEO_PATH, GEO_NAME)

# Output directory
OUTPUT_DIR = "/home/jofre/Members/Eduard/Paper2/Simulations/NACA_0012_AOA12_Re50000_1716x1662x128/Mean_data/PDF_analysis/"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Reference parameters
rho_ref = 1.0
u_infty = 1.0
c = 1.0
Re_c = 50000
q_inf = 0.5 * rho_ref * u_infty**2
AOA = 12

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
n_snapshots = 0

print(f"Loading {N_total_snapshots} snapshots to compute mean...")

for idx, surface_file in enumerate(all_surface_files):
    if (idx + 1) % 20 == 0 or idx == 0:
        print(f"  Loading snapshot {idx+1}/{N_total_snapshots}...", flush=True)
    
    try:
        with h5py.File(surface_file, "r") as f:

            p_w = f["p_w"][:] + f.attrs["p_bulk"]   # (Nz_phys, N_surf)
            tau_w = f["tau_w"][:]    # (Nz_phys, N_surf)
            
            
            # Spanwise average for each snapshot
            p_w_2d = np.mean(p_w, axis=0)   # (N_surf,)
            tau_w_2d = np.mean(tau_w, axis=0)       # (N_surf,)
            
            if p_w_2d_sum is None:
                p_w_2d_sum = p_w_2d.copy()
                tau_w_2d_sum = tau_w_2d.copy()
                Nz_phys = p_w.shape[0]  # Store for later
            else:
                p_w_2d_sum += p_w_2d
                tau_w_2d_sum += tau_w_2d
            
            n_snapshots += 1
            
    except Exception as e:
        print(f"  [WARNING] Error loading {surface_file}: {e}")
        continue

if n_snapshots == 0:
    raise RuntimeError("No valid snapshots loaded; check surface files and datasets.")

# Compute 2D time-averaged means
p_w_mean = p_w_2d_sum / n_snapshots      # (N_surf,)
tau_w_mean = tau_w_2d_sum / n_snapshots  # (N_surf,)

print(f"  Successfully loaded {n_snapshots} snapshots")
print(f"  2D mean shape: (N_surf={len(p_w_mean)})")
print(f"  Spanwise planes in each snapshot: Nz={Nz_phys}")

# ============================================================================
# Collect fluctuations at each chord location
# ============================================================================
print("\n" + "=" * 70)
print("COLLECTING FLUCTUATION DATA")
print("=" * 70)

fluctuations_data = {}

for x_c_target in point_indices.keys():
    fluctuations_data[x_c_target] = {
        'p_w_prime': [],
        'tau_w_prime': []
    }

print(f"Processing {n_snapshots} snapshots for fluctuations...")

for idx, surface_file in enumerate(all_surface_files[:n_snapshots]):
    if (idx + 1) % 20 == 0 or idx == 0:
        print(f"  Processing snapshot {idx+1}/{n_snapshots}...", flush=True)
    
    try:
        with h5py.File(surface_file, "r") as f:
            p_w = f["p_w"][:] + f.attrs["p_bulk"]   # (Nz_phys, N_surf)
            tau_w = f["tau_w"][:]  # (Nz_phys, N_surf)
        
        # Compute fluctuations: 3D field minus 2D mean (broadcasting)
        # p_w_mean has shape (N_surf,)
        # p_w has shape (Nz_phys, N_surf)
        # Broadcasting: (Nz_phys, N_surf) - (N_surf,) → (Nz_phys, N_surf)
        p_w_prime = p_w - p_w_mean[np.newaxis, :]  # (Nz_phys, N_surf)
        tau_w_prime = tau_w - tau_w_mean[np.newaxis, :]  # (Nz_phys, N_surf)
        
        # Extract at each chord location (closest point only)
        for x_c_target, point_info in point_indices.items():
            idx_point = point_info['indices'][0]  # Single point
            
            # Extract fluctuations at all z-locations for this point
            p_prime_at_xc = p_w_prime[:, idx_point]    # (Nz_phys,)
            tau_prime_at_xc = tau_w_prime[:, idx_point]  # (Nz_phys,)
            
            # Store all z-locations as independent samples
            fluctuations_data[x_c_target]['p_w_prime'].extend(p_prime_at_xc)
            fluctuations_data[x_c_target]['tau_w_prime'].extend(tau_prime_at_xc)
        
    except Exception as e:
        print(f"  [WARNING] Error processing {surface_file}: {e}")
        continue


# ============================================================================
# Compute RMS manually
# ============================================================================
print("\n" + "=" * 70)
print("COMPUTING RMS VALUES MANUALLY")
print("=" * 70)

rms_data = {}

for x_c_target in point_indices.keys():
    p_prime = np.asarray(fluctuations_data[x_c_target]['p_w_prime'])
    tau_prime = np.asarray(fluctuations_data[x_c_target]['tau_w_prime'])
    
    n_samples = len(p_prime)
    
    # --- PRESSURE RMS ---
    # Step 1: Verify mean of fluctuations is ~0
    p_prime_mean = np.sum(p_prime) / n_samples
    tau_prime_mean = np.sum(tau_prime) / n_samples

    # Step 2: Square each fluctuation value
    p_prime_squared = p_prime ** 2  # Element-wise squaring
    tau_prime_squared = tau_prime ** 2

    
    # Step 3: Compute mean of squared fluctuations: <p'^2>
    p_prime_squared_mean = np.sum(p_prime_squared) / n_samples
    tau_prime_squared_mean = np.sum(tau_prime_squared) / n_samples
    
    # Step 4: Take square root to get RMS
    p_rms = np.sqrt(p_prime_squared_mean)
    tau_rms = np.sqrt(tau_prime_squared_mean)
        
    # Store results
    rms_data[x_c_target] = {
        'p_rms': p_rms,
        'p_variance': p_prime_squared_mean,  # <p'^2>
        'p_mean_check': p_prime_mean,        # Should be ~0
        'tau_rms': tau_rms,
        'tau_variance': tau_prime_squared_mean,  # <tau'^2>
        'tau_mean_check': tau_prime_mean,        # Should be ~0
        'n_samples': n_samples
    }
    
    print(f"\n  x/c = {x_c_target:.2f} ({n_samples} samples):")
    print(f"    Pressure fluctuations:")
    print(f"      Mean (should be ~0):  {p_prime_mean:.6e}")
    print(f"      Variance <p'^2>:      {p_prime_squared_mean:.6e}")
    print(f"      RMS = √<p'^2>:        {p_rms:.6e}")
    print(f"    Shear stress fluctuations:")
    print(f"      Mean (should be ~0):  {tau_prime_mean:.6e}")
    print(f"      Variance <τ'^2>:      {tau_prime_squared_mean:.6e}")
    print(f"      RMS = √<τ'^2>:        {tau_rms:.6e}")

# ============================================================================
# Normalize fluctuations by RMS
# ============================================================================
print("\n" + "=" * 70)
print("NORMALIZING FLUCTUATIONS BY RMS")
print("=" * 70)

normalized_data = {}

for x_c_target in point_indices.keys():
    p_prime = np.asarray(fluctuations_data[x_c_target]['p_w_prime'])
    tau_prime = np.asarray(fluctuations_data[x_c_target]['tau_w_prime'])
    
    p_rms = rms_data[x_c_target]['p_rms']
    tau_rms = rms_data[x_c_target]['tau_rms']
    
    # Normalize: divide each value by RMS
    p_prime_norm = p_prime / p_rms        # (Nz_phys,)
    tau_prime_norm = tau_prime / tau_rms
    
    normalized_data[x_c_target] = {
        'p_prime_norm': p_prime_norm,
        'tau_prime_norm': tau_prime_norm
    }
    
    # Verify: RMS of normalized data should be 1.0
    p_norm_rms_check = np.sqrt(np.sum(p_prime_norm**2) / len(p_prime_norm))
    tau_norm_rms_check = np.sqrt(np.sum(tau_prime_norm**2) / len(tau_prime_norm))
    
    print(f"  x/c = {x_c_target:.2f}:")
    print(f"    Normalized pressure RMS (should be 1.0):     {p_norm_rms_check:.6f}")
    print(f"    Normalized shear stress RMS (should be 1.0): {tau_norm_rms_check:.6f}")

# ============================================================================
# Compute histogram (PDF) manually
# ============================================================================
print("\n" + "=" * 70)
print("COMPUTING PDFs MANUALLY")
print("=" * 70)

pdf_data = {}

for x_c_target in point_indices.keys():
    p_norm = normalized_data[x_c_target]['p_prime_norm']
    tau_norm = normalized_data[x_c_target]['tau_prime_norm']
    
    # --- MANUAL HISTOGRAM FOR PRESSURE ---
    
    # Step 1: Define bin edges
    # Find range of data
    p_min = np.min(p_norm)
    p_max = np.max(p_norm)
    
    # Create N_BINS equally-spaced bin edges
    # We need N_BINS+1 edges to define N_BINS bins
    p_bin_edges = np.linspace(p_min, p_max, N_BINS + 1)
    
    # Step 2: Initialize bin counts
    p_counts = np.zeros(N_BINS)
    
    # Step 3: Count samples in each bin
    for value in p_norm:
        # Find which bin this value belongs to
        # bin_index ranges from 0 to N_BINS-1
        bin_index = int((value - p_min) / (p_max - p_min) * N_BINS)
        
        # Handle edge case: if value == p_max, bin_index = N_BINS
        if bin_index >= N_BINS:
            bin_index = N_BINS - 1
        
        p_counts[bin_index] += 1
    
    # Step 4: Normalize to get PDF
    # PDF should integrate to 1.0
    # For histogram: ∫ PDF dx = Σ (PDF_i * Δx_i) = 1
    # Therefore: PDF_i = count_i / (total_count * Δx)
    
    total_samples = len(p_norm)
    bin_width = (p_max - p_min) / N_BINS
    
    p_pdf = p_counts / (total_samples * bin_width)
    
    # Step 5: Compute bin centers for plotting
    p_bin_centers = np.zeros(N_BINS)
    for i in range(N_BINS):
        p_bin_centers[i] = 0.5 * (p_bin_edges[i] + p_bin_edges[i+1])
    
    # --- MANUAL HISTOGRAM FOR SHEAR STRESS ---
    tau_min = np.min(tau_norm)
    tau_max = np.max(tau_norm)
    tau_bin_edges = np.linspace(tau_min, tau_max, N_BINS + 1)
    tau_counts = np.zeros(N_BINS)
    
    for value in tau_norm:
        bin_index = int((value - tau_min) / (tau_max - tau_min) * N_BINS)
        if bin_index >= N_BINS:
            bin_index = N_BINS - 1
        tau_counts[bin_index] += 1
    
    tau_bin_width = (tau_max - tau_min) / N_BINS
    tau_pdf = tau_counts / (total_samples * tau_bin_width)
    
    tau_bin_centers = np.zeros(N_BINS)
    for i in range(N_BINS):
        tau_bin_centers[i] = 0.5 * (tau_bin_edges[i] + tau_bin_edges[i+1])
    
    # --- VERIFY PDF NORMALIZATION ---
    # Check: ∫ PDF dx = Σ (PDF_i * Δx_i) should equal 1.0
    p_integral = np.sum(p_pdf * bin_width)
    tau_integral = np.sum(tau_pdf * tau_bin_width)
    
    # Store results
    pdf_data[x_c_target] = {
        'p_bins': p_bin_centers,
        'p_pdf': p_pdf,
        'p_bin_edges': p_bin_edges,
        'p_bin_width': bin_width,
        'tau_bins': tau_bin_centers,
        'tau_pdf': tau_pdf,
        'tau_bin_edges': tau_bin_edges,
        'tau_bin_width': tau_bin_width,
    }
    
    print(f"\n  x/c = {x_c_target:.2f}:")
    print(f"    Pressure PDF:")
    print(f"      Data range:     [{p_min:.4f}, {p_max:.4f}]")
    print(f"      Bin width:      {bin_width:.6f}")
    print(f"      Integral check: {p_integral:.6f} (should be 1.0)")
    print(f"      Max PDF value:  {np.max(p_pdf):.4f}")
    print(f"    Shear stress PDF:")
    print(f"      Data range:     [{tau_min:.4f}, {tau_max:.4f}]")
    print(f"      Bin width:      {tau_bin_width:.6f}")
    print(f"      Integral check: {tau_integral:.6f} (should be 1.0)")
    print(f"      Max PDF value:  {np.max(tau_pdf):.4f}")

# ============================================================================
# Save results
# ============================================================================
print("\n" + "=" * 70)
print("SAVING RESULTS")
print("=" * 70)

output_file = os.path.join(OUTPUT_DIR, "surface_fluctuation_PDFs.h5")

with h5py.File(output_file, "w") as f:
    # Save metadata
    f.attrs["n_snapshots"] = n_snapshots
    f.attrs["n_bins"] = N_BINS
    f.attrs["surface_selection"] = "suction side (upper surface)"
    f.attrs["point_selection"] = "closest point to each x/c location"
    f.attrs["normalization"] = "by RMS values"
    f.attrs["AOA"] = AOA
    f.attrs["Re"] = Re_c
    f.attrs["q_inf"] = q_inf
    
    # Save data for each x/c location
    for x_c_target, data in pdf_data.items():
        grp = f.create_group(f"xc_{x_c_target:.2f}")
        
        # PDFs (normalized)
        grp.create_dataset("p_bins", data=data['p_bins'])
        grp.create_dataset("p_pdf", data=data['p_pdf'])
        grp.create_dataset("tau_bins", data=data['tau_bins'])
        grp.create_dataset("tau_pdf", data=data['tau_pdf'])

print(f"  Results saved to: {output_file}")

# ============================================================================
# Plotting
# ============================================================================
print("\n" + "=" * 70)
print("GENERATING PLOTS")
print("=" * 70)

# Plot 1: All pressure PDFs on one plot
fig, ax = plt.subplots(figsize=(12, 8))

colors = plt.cm.viridis(np.linspace(0, 1, len(pdf_data)))

for idx, (x_c_target, data) in enumerate(pdf_data.items()):
    ax.plot(data['p_bins'], data['p_pdf'], '-o', 
            color=colors[idx], linewidth=2, markersize=4,
            label=f'x/c = {x_c_target:.2f}')

ax.set_xlabel("Normalized pressure fluctuation, $p'_w/p'_{w,rms}$", fontsize=14)
ax.set_ylabel('PDF', fontsize=14)
ax.set_title(f'Pressure Fluctuation PDFs - NACA 0012, AoA={AOA}°, Re={Re_c:,}\n' +
             f'{n_snapshots} snapshots', fontsize=15)
ax.legend(loc='best', fontsize=10, ncol=2)
ax.grid(True, alpha=0.3, linestyle='--')
ax.set_yscale('log')

plt.tight_layout()
plot1_path = os.path.join(OUTPUT_DIR, "pressure_fluctuation_PDFs.png")
plt.savefig(plot1_path, dpi=300, bbox_inches='tight')
print(f"  Plot saved: {plot1_path}")
plt.show()

# Plot 2: All wall shear stress PDFs on one plot
fig, ax = plt.subplots(figsize=(12, 8))

for idx, (x_c_target, data) in enumerate(pdf_data.items()):
    ax.plot(data['tau_bins'], data['tau_pdf'], '-o', 
            color=colors[idx], linewidth=2, markersize=4,
            label=f'x/c = {x_c_target:.2f}')

ax.set_xlabel("Normalized wall shear stress fluctuation, $\\tau'_w/\\tau'_{w,rms}$", fontsize=14)
ax.set_ylabel('PDF', fontsize=14)
ax.set_title(f'Wall Shear Stress Fluctuation PDFs - NACA 0012, AoA={AOA}°, Re={Re_c:,}\n' +
             f'{n_snapshots} snapshots', fontsize=15)
ax.legend(loc='best', fontsize=10, ncol=2)
ax.grid(True, alpha=0.3, linestyle='--')
ax.set_yscale('log')

plt.tight_layout()
plot2_path = os.path.join(OUTPUT_DIR, "wall_shear_stress_fluctuation_PDFs.png")
plt.savefig(plot2_path, dpi=300, bbox_inches='tight')
print(f"  Plot saved: {plot2_path}")
plt.show()

print("\n" + "=" * 70)
print("ALL DONE!")
print(f"Results saved to: {OUTPUT_DIR}")
print("=" * 70)