import os
import sys
import h5py
import numpy as np
import matplotlib.pyplot as plt

# ============================================================================
# Configuration
# ============================================================================

# Path to the saved surface data HDF5 file
SURFACE_DATA_FILE = "/home/jofre/Members/Eduard/Paper2/Simulations/NACA_0012_AOA12_Re50000_1716x1662x128/Mean_data/PDF_analysis/surface_data_slices.h5"

# Output directory
OUTPUT_DIR = "/home/jofre/Members/Eduard/Paper2/Simulations/NACA_0012_AOA12_Re50000_1716x1662x128/Mean_data/PDF_analysis/"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Number of bins for histogram (PDF)S
N_BINS = 250

# ============================================================================
# List available slices in the HDF5 file
# ============================================================================
print("\n" + "=" * 70)
print("AVAILABLE SLICE LOCATIONS IN HDF5 FILE")
print("=" * 70)

with h5py.File(SURFACE_DATA_FILE, "r") as f:
    available_slices = list(f.keys())
    print(f"Found {len(available_slices)} slices:")
    for slice_name in sorted(available_slices):
        grp = f[slice_name]
        x_c = grp.attrs.get("x_c", "N/A")
        num_p = grp.attrs.get("num_samples_p", 0)
        num_tau = grp.attrs.get("num_samples_tau", 0)
        print(f"  {slice_name}: x/c = {x_c:.4f}, p samples = {num_p}, tau samples = {num_tau}")

# ============================================================================
# USER INPUT: Select which slices to plot
# ============================================================================
print("\n" + "=" * 70)
print("SELECT SLICES TO PLOT")
print("=" * 70)
print("Enter the slice names separated by commas (or 'all' for all slices)")
print(f"Example: slice_1,slice_3,slice_5")
user_input = input("> ").strip()

if user_input.lower() == "all":
    selected_slices = sorted(available_slices)
else:
    selected_slices = [s.strip() for s in user_input.split(",")]
    # Validate selection
    for s in selected_slices:
        if s not in available_slices:
            print(f"[WARNING] Slice '{s}' not found in file. Available: {available_slices}")
            selected_slices.remove(s)

if len(selected_slices) == 0:
    raise RuntimeError("No valid slices selected.")

print(f"\nSelected {len(selected_slices)} slices for plotting:")
for s in selected_slices:
    print(f"  - {s}")

# ============================================================================
# Load surface data from HDF5
# ============================================================================
print("\n" + "=" * 70)
print("LOADING SURFACE DATA FROM HDF5")
print("=" * 70)

surface_data = {}
slice_metadata = {}

with h5py.File(SURFACE_DATA_FILE, "r") as f:
    for slice_name in selected_slices:
        grp = f[slice_name]
        
        p_samples = grp["p_w"][:]
        tau_samples = grp["tau_w"][:]
        
        surface_data[slice_name] = {
            'p_w': p_samples,
            'tau_w': tau_samples,
        }
        
        slice_metadata[slice_name] = {
            'x_c': grp.attrs.get("x_c", None),
            'interface_x': grp.attrs.get("interface_x", None),
            'interface_y': grp.attrs.get("interface_y", None),
            'interface_index': grp.attrs.get("interface_index", None),
            'y_grid_index': grp.attrs.get("y_grid_index", None),
        }
        
        print(f"  {slice_name}: p_w {p_samples.shape}, tau_w {tau_samples.shape}")

# =========================================================================
# Compute histogram (PDF)
# =========================================================================
print("\n" + "=" * 70)
print("COMPUTING PDFs")
print("=" * 70)

pdf_data = {}

for slice_name in selected_slices:
    p_samples = surface_data[slice_name]['p_w']
    tau_samples = surface_data[slice_name]['tau_w']

    p_hist, p_bin_edges = np.histogram(p_samples, bins=N_BINS, density=True)
    tau_hist, tau_bin_edges = np.histogram(tau_samples, bins=N_BINS, density=True)

    p_bin_centers = 0.5 * (p_bin_edges[:-1] + p_bin_edges[1:])
    tau_bin_centers = 0.5 * (tau_bin_edges[:-1] + tau_bin_edges[1:])

    pdf_data[slice_name] = {
        'p_bins': p_bin_centers,
        'p_pdf': p_hist,
        'tau_bins': tau_bin_centers,
        'tau_pdf': tau_hist,
    }

    print(f"\n  {slice_name}:")
    print(f"    Pressure PDF:")
    print(f"      Number of samples: {len(p_samples)}")
    print(f"      Data range:     [{np.min(p_samples):.4f}, {np.max(p_samples):.4f}]")
    print(f"      Max PDF value:  {np.max(p_hist):.4f}")
    print(f"    Shear stress PDF:")
    print(f"      Number of samples: {len(tau_samples)}")
    print(f"      Data range:     [{np.min(tau_samples):.4f}, {np.max(tau_samples):.4f}]")
    print(f"      Max PDF value:  {np.max(tau_hist):.4f}")

# =========================================================================
# Plot PDFs for each slice location
# =========================================================================
print("\n" + "=" * 70)
print("PLOTTING PDFs")
print("=" * 70)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

for slice_name in selected_slices:
    pdf_info = pdf_data[slice_name]
    x_c_slice = slice_metadata[slice_name]['x_c']
    label = f"{slice_name} (x/c = {x_c_slice:.3f})"

    ax1.plot(pdf_info['p_bins'], pdf_info['p_pdf'], label=label, linewidth=2)
    ax2.plot(pdf_info['tau_bins'], pdf_info['tau_pdf'], label=label, linewidth=2)

ax1.set_xlabel("Pressure $p$", fontsize=12)
ax1.set_ylabel("PDF", fontsize=12)
ax1.set_title("Pressure PDF", fontsize=13, fontweight='bold')
ax1.set_yscale('log')
ax1.grid(True, alpha=0.3, which='both')
ax1.legend(fontsize=10)

ax2.set_xlabel("Shear stress $\\tau_w$", fontsize=12)
ax2.set_ylabel("PDF", fontsize=12)
ax2.set_title("Wall Shear Stress PDF", fontsize=13, fontweight='bold')
ax2.set_yscale('log')
ax2.grid(True, alpha=0.3, which='both')
ax2.legend(fontsize=10)

plt.tight_layout()
pdf_figure_path = os.path.join(OUTPUT_DIR, "PDFs_selected_slices.png")
plt.savefig(pdf_figure_path, dpi=150, bbox_inches='tight')
print(f"PDF figure saved to: {pdf_figure_path}")
plt.show()

# =========================================================================
# Compute and Plot Normalized Fluctuation PDFs
# =========================================================================
print("\n" + "=" * 70)
print("COMPUTING NORMALIZED FLUCTUATION PDFs")
print("=" * 70)

normalized_pdf_data = {}

for slice_name in selected_slices:
    p_samples = surface_data[slice_name]['p_w']
    tau_samples = surface_data[slice_name]['tau_w']

    p_mean = np.mean(p_samples)
    tau_mean = np.mean(tau_samples)
    p_std = np.std(p_samples)
    tau_std = np.std(tau_samples)

    if p_std == 0.0 or tau_std == 0.0:
        print(f"  [WARNING] Zero std for {slice_name}; skipping normalized PDFs")
        continue

    p_fluct_norm = (p_samples - p_mean) / p_std
    tau_fluct_norm = (tau_samples - tau_mean) / tau_std

    p_hist, p_bin_edges = np.histogram(p_fluct_norm, bins=N_BINS, density=True)
    tau_hist, tau_bin_edges = np.histogram(tau_fluct_norm, bins=N_BINS, density=True)

    p_bin_centers = 0.5 * (p_bin_edges[:-1] + p_bin_edges[1:])
    tau_bin_centers = 0.5 * (tau_bin_edges[:-1] + tau_bin_edges[1:])

    normalized_pdf_data[slice_name] = {
        'p_bins': p_bin_centers,
        'p_pdf': p_hist,
        'tau_bins': tau_bin_centers,
        'tau_pdf': tau_hist,
    }

    p_norm_mean = np.mean(p_fluct_norm)
    p_norm_var = np.var(p_fluct_norm)
    tau_norm_mean = np.mean(tau_fluct_norm)
    tau_norm_var = np.var(tau_fluct_norm)

    print(f"\n  {slice_name}:")
    print(f"    <p> = {p_mean:.6f}, std(p) = {p_std:.6f}")
    print(f"    <tau_w> = {tau_mean:.6f}, std(tau_w) = {tau_std:.6f}")
    print(f"    Normalized p' range: [{np.min(p_fluct_norm):.2f}, {np.max(p_fluct_norm):.2f}]")
    print(f"    Normalized tau_w' range: [{np.min(tau_fluct_norm):.2f}, {np.max(tau_fluct_norm):.2f}]")
    print(f"    --- Verification ---")
    print(f"    Normalized p':    mean = {p_norm_mean:.6f}, var = {p_norm_var:.6f}")
    print(f"    Normalized tau_w': mean = {tau_norm_mean:.6f}, var = {tau_norm_var:.6f}")

print("\n" + "=" * 70)
print("PLOTTING NORMALIZED FLUCTUATION PDFs")
print("=" * 70)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

for slice_name in selected_slices:
    if slice_name not in normalized_pdf_data:
        continue
    
    pdf_info = normalized_pdf_data[slice_name]
    x_c_slice = slice_metadata[slice_name]['x_c']
    label = f"{slice_name} (x/c = {x_c_slice:.3f})"

    ax1.plot(pdf_info['p_bins'], pdf_info['p_pdf'], label=label, linewidth=2)
    ax2.plot(pdf_info['tau_bins'], pdf_info['tau_pdf'], label=label, linewidth=2)

ax1.set_xlabel("Normalized Pressure Fluctuation $p'$", fontsize=12)
ax1.set_ylabel("PDF", fontsize=12)
ax1.set_title("Normalized Pressure Fluctuation PDF", fontsize=13, fontweight='bold')
ax1.set_yscale('log')
ax1.grid(True, alpha=0.3, which='both')
ax1.legend(fontsize=10)

ax2.set_xlabel("Normalized Shear Stress Fluctuation $\\tau_w'$", fontsize=12)
ax2.set_ylabel("PDF", fontsize=12)
ax2.set_title("Normalized Wall Shear Stress Fluctuation PDF", fontsize=13, fontweight='bold')
ax2.set_yscale('log')
ax2.grid(True, alpha=0.3, which='both')
ax2.legend(fontsize=10)

plt.tight_layout()
norm_pdf_figure_path = os.path.join(OUTPUT_DIR, "normalized_PDFs_selected_slices.png")
plt.savefig(norm_pdf_figure_path, dpi=150, bbox_inches='tight')
print(f"Normalized PDF figure saved to: {norm_pdf_figure_path}")
plt.show()

print("\n" + "=" * 70)
print("DONE")
print("=" * 70)


