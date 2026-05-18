import os
import sys
import h5py
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d

# ============================================================================
# Set LaTeX style for plots
# ============================================================================
plt.rc('text', usetex=True)
plt.rc('font', size=16, family='serif')
plt.rc('text.latex', preamble=r'\usepackage{amsmath} \usepackage{amssymb}')

# ============================================================================
# Configuration
# ============================================================================

# Path to the saved surface data HDF5 file
SURFACE_DATA_AOA5_FILE = "/home/jofre/Members/Eduard/Paper2/Simulations/NACA_0012_AOA5_Re50000_1716x1662x128/Mean_data/PDF_analysis/surface_data_slices_AoA5_Re50000.h5"
SURFACE_DATA_AOA12_FILE = "/home/jofre/Members/Eduard/Paper2/Simulations/NACA_0012_AOA12_Re50000_1716x1662x128/Mean_data/PDF_analysis/surface_data_slices_AoA12_Re50000.h5"


# Output directory
OUTPUT_DIR = "/home/jofre/Members/Eduard/Paper2/Figures/"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Number of bins for histogram (PDF)
N_BINS = 100

# Smoothing strength for PDF curves
SMOOTH_SIGMA = 2

# ============================================================================
# List available slices in the HDF5 files
# ============================================================================
print("\n" + "=" * 70)
print("AVAILABLE SLICE LOCATIONS IN HDF5 FILES")
print("=" * 70)

available_slices = {}

for aoa, file_path in [("AOA5", SURFACE_DATA_AOA5_FILE), ("AOA12", SURFACE_DATA_AOA12_FILE)]:
    print(f"\n{aoa}:")
    with h5py.File(file_path, "r") as f:
        slices = list(f.keys())
        available_slices[aoa] = slices
        print(f"Found {len(slices)} slices:")
        for slice_name in sorted(slices):
            grp = f[slice_name]
            x_c = grp.attrs.get("x_c", "N/A")
            num_p = grp.attrs.get("num_samples_p", 0)
            num_tau = grp.attrs.get("num_samples_tau", 0)
            print(f"  {slice_name}: x/c = {x_c:.4f}, p samples = {num_p}, tau samples = {num_tau}")

# ============================================================================
# SELECT SLICES TO PROCESS
# ============================================================================
print("\n" + "=" * 70)
print("SELECT SLICES TO PROCESS")
print("=" * 70)

# Get unique slice names (without AOA prefix)
unique_slices = sorted(list(set(
    slice_name for aoa in available_slices.keys() 
    for slice_name in available_slices[aoa]
)))

print(f"\nAvailable slices:")
for i, slice_name in enumerate(unique_slices):
    print(f"  {i}: {slice_name}")

print(f"\nEnter slice indices separated by commas (or 'all' for all slices)")
print(f"Example: 0,2,4")
user_input = input("> ").strip()

if user_input.lower() == "all":
    selected_slice_names = unique_slices
else:
    try:
        indices = [int(i.strip()) for i in user_input.split(",")]
        selected_slice_names = [unique_slices[i] for i in indices if 0 <= i < len(unique_slices)]
        if len(selected_slice_names) == 0:
            raise ValueError("No valid indices selected")
    except (ValueError, IndexError) as e:
        print(f"[ERROR] Invalid input: {e}")
        selected_slice_names = unique_slices

print(f"\nSelected {len(selected_slice_names)} slices for processing:")
for slice_name in selected_slice_names:
    print(f"  - {slice_name}")

# ============================================================================
# Load surface data from HDF5 for both AOA5 and AOA12
# ============================================================================
print("\n" + "=" * 70)
print("LOADING SURFACE DATA FROM HDF5")
print("=" * 70)

surface_data = {}
slice_metadata = {}

for aoa, file_path in [("AOA5", SURFACE_DATA_AOA5_FILE), ("AOA12", SURFACE_DATA_AOA12_FILE)]:
    print(f"\n{aoa}:")
    with h5py.File(file_path, "r") as f:
        for slice_name in selected_slice_names:
            if slice_name in f.keys():
                grp = f[slice_name]
                
                pressure_samples = grp["p_w"][:]
                
                key = f"{aoa}_{slice_name}"
                surface_data[key] = {
                    'p_w': pressure_samples,
                }
                
                slice_metadata[key] = {
                    'aoa': aoa,
                    'slice': slice_name,
                    'x_c': grp.attrs.get("x_c", None),
                }
                
                print(f"  {slice_name}: p_w {pressure_samples.shape}")

# =========================================================================
# Compute histogram (PDF) for raw p_w values
# =========================================================================
print("\n" + "=" * 70)
print("COMPUTING RAW P_W PDFs")
print("=" * 70)

pdf_data = {}

for key in sorted(surface_data.keys()):
    pressure_samples = surface_data[key]['p_w']

    pressure_hist, pressure_bin_edges = np.histogram(pressure_samples, bins=N_BINS, density=True)
    pressure_bin_centers = 0.5 * (pressure_bin_edges[:-1] + pressure_bin_edges[1:])

    pdf_data[key] = {
        'pressure_bins': pressure_bin_centers,
        'pressure_pdf': pressure_hist,
    }

    aoa = slice_metadata[key]['aoa']
    slice_name = slice_metadata[key]['slice']
    print(f"\n  {aoa}_{slice_name}:")
    print(f"    Raw p_w PDF:")
    print(f"      Number of samples: {len(pressure_samples)}")
    print(f"      Data range:     [{np.min(pressure_samples):.4f}, {np.max(pressure_samples):.4f}]")
    print(f"      Max PDF value:  {np.max(pressure_hist):.4f}")

# ============================================================================
# CREATE PLOTS FOR RAW P_W (One per AOA showing different locations)
# ============================================================================
# print("\n" + "=" * 70)
# print("CREATING RAW P_W PLOTS")
# print("=" * 70)

# # Create one plot per AOA for raw p_w
# for aoa in ["AOA5", "AOA12"]:
#     fig, ax = plt.subplots(figsize=(12, 7))
    
#     # Plot all selected slices for this AOA
#     for slice_name in selected_slice_names:
#         key = f"{aoa}_{slice_name}"
#         if key in pdf_data:
#             pressure_bins = pdf_data[key]['pressure_bins']
#             pressure_pdf = pdf_data[key]['pressure_pdf']
#             x_c = slice_metadata[key]['x_c']
            
#             ax.plot(pressure_bins, pressure_pdf, linewidth=2, label=f"x/c = {x_c:.4f}", alpha=0.8)
    
#     ax.set_xlabel(r"Wall Shear Stress $\tau_w$ (Pa)", fontsize=12)
#     ax.set_ylabel(r"Probability Density", fontsize=12)
#     ax.set_title(f"PDF of Wall Shear Stress - {aoa} (Re = 50000)", fontsize=13, fontweight='bold')
#     ax.legend(fontsize=10, loc='best')
#     ax.grid(True, alpha=0.3, which='both')
#     ax.set_yscale('log')
    
#     plt.show()

# =========================================================================
# Compute normalized fluctuation PDFs (using standard deviation normalization)
# =========================================================================
print("\n" + "=" * 70)
print("COMPUTING NORMALIZED FLUCTUATION PDFs")
print("=" * 70)

normalized_pdf_data = {}
norm_stats = {}

for key in sorted(surface_data.keys()):
    pressure_samples = surface_data[key]['p_w']
    
    # Compute mean and standard deviation
    pressure_mean = np.mean(pressure_samples)
    pressure_std = np.std(pressure_samples)
    
    if pressure_std == 0.0:
        print(f"  [WARNING] Zero std for {key}; skipping normalized PDFs")
        continue
    
    # Normalize fluctuations by standard deviation
    pressure_fluct_norm = (pressure_samples - pressure_mean) / pressure_std
    
    # Compute histogram
    pressure_hist, pressure_bin_edges = np.histogram(pressure_fluct_norm, bins=N_BINS, density=True)
    pressure_bin_centers = 0.5 * (pressure_bin_edges[:-1] + pressure_bin_edges[1:])
    
    normalized_pdf_data[key] = {
        'pressure_bins': pressure_bin_centers,
        'pressure_pdf': pressure_hist,
    }
    
    # Store statistics
    pressure_norm_mean = np.mean(pressure_fluct_norm)
    pressure_norm_var = np.var(pressure_fluct_norm)
    
    norm_stats[key] = {
        'pressure_mean': pressure_mean,
        'pressure_std': pressure_std,
        'pressure_norm_mean': pressure_norm_mean,
        'pressure_norm_var': pressure_norm_var,
    }
    
    aoa = slice_metadata[key]['aoa']
    slice_name = slice_metadata[key]['slice']
    print(f"\n  {aoa}_{slice_name}:")
    print(f"    <p_w> = {pressure_mean:.6f}, std(p_w) = {pressure_std:.6f}")
    print(f"    Normalized p_w' range: [{np.min(pressure_fluct_norm):.4f}, {np.max(pressure_fluct_norm):.4f}]")
    print(f"    --- Verification ---")
    print(f"    Normalized p_w': mean = {pressure_norm_mean:.6f}, var = {pressure_norm_var:.6f}")

# ============================================================================
# CREATE PLOTS FOR NORMALIZED FLUCTUATIONS (One per AOA showing different locations)
# ============================================================================
# print("\n" + "=" * 70)
# print("CREATING NORMALIZED FLUCTUATION PLOTS")
# print("=" * 70)

# # Create one plot per AOA for normalized fluctuations
# for aoa in ["AOA5", "AOA12"]:
#     fig, ax = plt.subplots(figsize=(12, 7))
    
#     # Plot all selected slices for this AOA
#     for slice_name in selected_slice_names:
#         key = f"{aoa}_{slice_name}"
#         if key in normalized_pdf_data:
#             pressure_bins = normalized_pdf_data[key]['pressure_bins']
#             pressure_pdf = normalized_pdf_data[key]['pressure_pdf']
#             x_c = slice_metadata[key]['x_c']

#             ax.plot(pressure_bins, pressure_pdf, linewidth=2, label=f"x/c = {x_c:.4f}", alpha=0.8)

#     ax.set_xlabel(r"Normalized Pressure Fluctuation $p_w' / \sigma(p_w)$", fontsize=12)
#     ax.set_ylabel(r"Probability Density", fontsize=12)
#     ax.set_title(f"PDF of Normalized Wall Pressure Fluctuations - {aoa} (Re = 50000)", fontsize=13, fontweight='bold')
#     ax.legend(fontsize=10, loc='best')
#     ax.grid(True, alpha=0.3, which='both')
#     ax.set_yscale('log')
    
#     plt.show()

# ============================================================================
# CREATE COMBINED PLOT (Both AOA5 and AOA12 on same plot)
# ============================================================================
# print("\n" + "=" * 70)
# print("CREATING COMBINED NORMALIZED FLUCTUATION PLOT")
# print("=" * 70)

# fig, ax = plt.subplots(figsize=(8, 6))

# # Define colors for each slice location
# colors = ["red", "blue", "green", "orange", "purple", "cyan"]

# for idx, slice_name in enumerate(selected_slice_names):
#     color = colors[idx]
#     x_c = slice_metadata[f"AOA5_{slice_name}"]['x_c']
    
#     # Plot AOA5 with solid lines
#     key_aoa5 = f"AOA5_{slice_name}"
#     if key_aoa5 in normalized_pdf_data:
#         pressure_bins = normalized_pdf_data[key_aoa5]['pressure_bins']
#         pressure_pdf = normalized_pdf_data[key_aoa5]['pressure_pdf']
#         ax.plot(pressure_bins, pressure_pdf, linewidth=2.5, label=f"AOA5 - x/c = {x_c:.4f}", 
#                 color=color, linestyle='-', alpha=0.8)
    
#     # Plot AOA12 with dashed lines
#     key_aoa12 = f"AOA12_{slice_name}"
#     if key_aoa12 in normalized_pdf_data:
#         pressure_bins = normalized_pdf_data[key_aoa12]['pressure_bins']
#         pressure_pdf = normalized_pdf_data[key_aoa12]['pressure_pdf']
#         ax.plot(pressure_bins, pressure_pdf, linewidth=2.5, label=f"AOA12 - x/c = {x_c:.4f}", 
#                 color=color, linestyle='--', alpha=0.8)

# ax.set_xlabel(r"$p'_w / p'_{w,\mathrm{rms}}$", fontsize=13)
# ax.set_ylabel(r"$\mathrm{PDF}(p'_w/p'_{w,\mathrm{rms}})$", fontsize=13)
# ax.set_yscale('log')
# ax.set_xlim(-11, 11)
# ax.set_ylim(1e-5, 1)

# plt.tight_layout()
# plt.show()

# ============================================================================
# CREATE COMBINED PLOT WITH SMOOTHING 
# ============================================================================
print("\n" + "=" * 70)
print("CREATING SMOOTHED VS RAW COMPARISON PLOT")
print("=" * 70)

fig, ax = plt.subplots(figsize=(8, 6))

for idx, slice_name in enumerate(selected_slice_names):
    color = ["red", "blue", "green", "orange", "purple", "cyan"]
    x_c = slice_metadata[f"AOA5_{slice_name}"]['x_c']

    for aoa, linestyle in [("AOA5", "-"), ("AOA12", "--")]:
        key = f"{aoa}_{slice_name}"
        if key not in normalized_pdf_data:
            continue

        pressure_bins = normalized_pdf_data[key]['pressure_bins']
        pressure_pdf = normalized_pdf_data[key]['pressure_pdf']
        pressure_pdf_smooth = gaussian_filter1d(pressure_pdf, sigma=SMOOTH_SIGMA)

        # Smoothed curve (highlighted)
        # Only add label for AOA5 (for legend)
        label = f"x/c = {x_c:.1f}" if aoa == "AOA5" else None
        ax.plot(
            pressure_bins,
            pressure_pdf_smooth,
            linewidth=2,
            color=color[idx % len(color)],
            linestyle=linestyle,
            alpha=1,
            label=label,
        )

ax.set_xlabel(r"$p'_w / p'_{w,\mathrm{rms}}$", fontsize=16)
ax.set_ylabel(r"$\mathrm{PDF}(p'_w/p'_{w,\mathrm{rms}})$", fontsize=16)
ax.set_yscale('log')
ax.set_xlim(-5, 5)
ax.set_ylim(1e-5, 1)
ax.legend(loc='upper left', fontsize=16, frameon=False)

plt.tight_layout()
png_path = os.path.join(OUTPUT_DIR, "pressure_w_fluctuation_pdf.png")
eps_path = os.path.join(OUTPUT_DIR, "pressure_w_fluctuation_pdf.eps")
plt.savefig(png_path, dpi=300, bbox_inches='tight')
plt.savefig(eps_path, bbox_inches='tight')
plt.show()

print("\n" + "=" * 70)
print("PLOTTING COMPLETE")
print("=" * 70)
