"""
Comprehensive Probe and Airfoil Visualization
==============================================

This script:
1. Discovers all energy spectra HDF5 files in the Energy_spectra directory
2. Extracts probe locations and slice information from each file
3. Loads the airfoil surface geometry
4. Creates detailed visualizations showing probes and airfoil for each slice
5. Generates a combined overview showing all slices together

"""

import os
import sys
import h5py
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import re
from matplotlib.colors import LogNorm
from matplotlib.ticker import LogLocator, LogFormatterSciNotation

# ============================================================================
# Configuration
# ============================================================================

# # Paths
# ENERGY_SPECTRA_DIR = "/home/jofre/Members/Eduard/Paper2/Simulations/NACA_0012_AOA12_Re50000_1716x1662x128/Mean_data/Energy_spectra/"
# GEO_PATH = "/home/jofre/Members/Eduard/Paper2/Simulations/NACA_0012_AOA12_Re50000_1716x1662x128/Geometrical_data/"
# GEO_NAME = "3d_NACA0012_Re50000_AoA12_Geometrical_Data.h5"
# GEO_FILE = os.path.join(GEO_PATH, GEO_NAME)

# # TKE data path
# TKE_DATA_PATH = "/home/jofre/Members/Eduard/Paper2/Simulations/NACA_0012_AOA12_Re50000_1716x1662x128/Mean_data/"
# TKE_DATA_NAME = "tke_turbulent_kinetic_energy.h5"
# TKE_DATA_FILE = os.path.join(TKE_DATA_PATH, TKE_DATA_NAME)

# # Output directory
# OUTPUT_DIR = "/home/jofre/Members/Eduard/Paper2/Simulations/NACA_0012_AOA12_Re50000_1716x1662x128/Mean_data/Energy_spectra/Probe_visualizations"
# os.makedirs(OUTPUT_DIR, exist_ok=True)

# # Physical parameters
# AOA_deg = 12.0

# Paths
ENERGY_SPECTRA_DIR = "/home/jofre/Members/Eduard/Paper2/Simulations/NACA_0012_AOA5_Re50000_1716x1662x128/Mean_data/Energy_spectra"
GEO_PATH = "/home/jofre/Members/Eduard/Paper2/Simulations/NACA_0012_AOA5_Re50000_1716x1662x128/Geometrical_data/"
GEO_NAME = "3d_NACA0012_Re50000_AoA5_Geometrical_Data.h5"
GEO_FILE = os.path.join(GEO_PATH, GEO_NAME)

# TKE data path
TKE_DATA_PATH = "/home/jofre/Members/Eduard/Paper2/Simulations/NACA_0012_AOA5_Re50000_1716x1662x128/Mean_data/"
TKE_DATA_NAME = "tke_turbulent_kinetic_energy.h5"
TKE_DATA_FILE = os.path.join(TKE_DATA_PATH, TKE_DATA_NAME)

# Output directory
OUTPUT_DIR = "/home/jofre/Members/Eduard/Paper2/Simulations/NACA_0012_AOA5_Re50000_1716x1662x128/Mean_data/Energy_spectra/Probe_visualizations"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Physical parameters
AOA_deg = 5.0

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def discover_energy_spectra_files(directory: str) -> dict:
    """
    Discover all energy spectra HDF5 files in the directory.

    Returns:
        dict: {slice_id: file_path}
    """
    files_dict = {}

    # for file_path in Path(directory).glob("energy_spectra_data_*.h5"):
    for file_path in Path(directory).glob("energy_spectra_data_*.h5"):

        filename = file_path.name
        match = re.search(r'energy_spectra_data_(slice_\d+)', filename)
        if match:
            slice_id = match.group(1)
            files_dict[slice_id] = str(file_path)

    return files_dict


def load_probe_data_from_h5(h5_file: str) -> dict:
    """
    Load probe information and spectra from HDF5 file.

    Returns:
        dict: {
            'slice_id': str,
            'slice_x': float,
            'probes': [ {'y_actual': float, 'y_target': float, 'index': int}, ... ]
            'frequencies': ndarray,
            'f_star': ndarray,
            'E_uu': ndarray,
            'E_vv': ndarray,
        }
    """
    data = {}

    with h5py.File(h5_file, 'r') as f:
        # Global attributes
        data['slice_id'] = f.attrs['slice_id']
        data['slice_x'] = f.attrs['slice_x']
        data['AOA_deg'] = f.attrs.get('AOA_deg', AOA_deg)
        data['dt_save'] = f.attrs.get('dt_save', None)
        data['fs'] = f.attrs.get('fs', None)

        # Probe information and spectra
        probes = []
        for key in sorted(f.keys()):
            if key.startswith('probe_'):
                grp = f[key]
                probe_info = {
                    'probe_name': key,
                    'y_actual': grp.attrs['y_actual'],
                    'y_target': grp.attrs['y_target'],
                    'j_index': grp.attrs['j_index'],
                    'y_distance_error': grp.attrs.get('y_distance_error', 0.0),
                    'E_uu': grp['E_uu'][...],
                    'E_vv': grp['E_vv'][...],
                }
                probes.append(probe_info)

        data['probes'] = probes

        # Load first probe's frequency arrays (same for all probes)
        if 'probe_00' in f:
            grp = f['probe_00']
            data['frequencies'] = grp['frequencies'][...]
            data['f_star'] = grp['f_star'][...]

            # Load spectra from first probe as example
            data['E_uu_example'] = grp['E_uu'][...]
            data['E_vv_example'] = grp['E_vv'][...]

    return data


def load_airfoil_surface(geo_file: str) -> np.ndarray:
    """
    Load airfoil surface geometry.

    Returns:
        ndarray: Interface points (all surface points)
    """
    with h5py.File(geo_file, 'r') as f:
        interface_points = f["interface_points"][...].astype(np.float64)

    return interface_points


def load_tke_data(tke_file: str) -> dict:
    """
    Load TKE field and metadata from HDF5 file.

    Returns:
        dict: {
            'tke': ndarray,
            'x_coords': ndarray,
            'y_coords': ndarray,
            'u_infty': float,
            'AOA': float,
        }
    """
    data = {}

    with h5py.File(tke_file, 'r') as f:
        # Load TKE field and coordinates
        data['tke'] = f["tke"][...]
        data['x_coords'] = f["x"][:]
        data['y_coords'] = f["y"][:]

        # Load metadata
        data['u_infty'] = f.attrs['u_infty']
        data['AOA'] = f.attrs['AOA']

    return data


def plot_spectra_for_slice(data: dict, output_dir: str):
    """
    Plot Euu and Evv spectra for all probes in a slice.

    Creates a figure with subplots for each probe showing both Euu and Evv.
    """
    slice_id = data['slice_id']
    probes = data['probes']
    frequencies = data['f_star']  # Use normalized frequency

    if len(probes) == 0:
        print(f"  Warning: No probes found for {slice_id}")
        return

    # Determine grid layout for subplots
    n_probes = len(probes)
    n_cols = 3
    n_rows = (n_probes + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 5*n_rows))
    axes = axes.flatten()  # Flatten to 1D for easier indexing

    # Plot each probe
    for probe_idx, probe in enumerate(probes):
        ax = axes[probe_idx]

        E_uu = probe['E_uu']
        E_vv = probe['E_vv']
        y_actual = probe['y_actual']
        probe_name = probe['probe_name']

        # Filter out zero/negative values for log scale
        E_uu_pos = np.where(E_uu > 0, E_uu, np.nan)
        E_vv_pos = np.where(E_vv > 0, E_vv, np.nan)

        # Plot both spectra
        ax.loglog(frequencies, E_uu_pos, 'b-', linewidth=2, label=r'$E_{uu}$', alpha=0.8)
        ax.loglog(frequencies, E_vv_pos, 'r-', linewidth=2, label=r'$E_{vv}$', alpha=0.8)

        # Formatting
        ax.set_xlabel(r'$f^* = fD/U_\infty$', fontsize=11, fontweight='bold')
        ax.set_ylabel(r'Energy Spectra', fontsize=11, fontweight='bold')
        ax.set_title(f'{probe_name} at y={y_actual:.4f}', fontsize=11, fontweight='bold')
        ax.grid(True, alpha=0.3, which='both')
        ax.legend(fontsize=10, loc='upper right')

    # Hide unused subplots
    for idx in range(n_probes, len(axes)):
        axes[idx].axis('off')

    fig.suptitle(f'{slice_id}: Energy Spectra $E_{{uu}}$ and $E_{{vv}}$ vs Normalized Frequency',
                 fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.show()


def plot_spectra_comparison_all_slices(all_slice_data: dict, output_dir: str):
    """
    Create comparison plots of spectra across all slices.

    Shows Euu and Evv from a representative probe in each slice overlapped.
    """
    if not all_slice_data:
        return

    slice_ids = sorted(all_slice_data.keys())

    # Use first probe from each slice
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    colors = plt.cm.tab10(np.linspace(0, 1, len(slice_ids)))

    for (slice_id, data), color in zip(zip(slice_ids, [all_slice_data[s] for s in slice_ids]), colors):
        if len(data['probes']) == 0:
            continue

        probe = data['probes'][0]  # First probe
        frequencies = data['f_star']
        E_uu = probe['E_uu']
        E_vv = probe['E_vv']
        slice_x = data['slice_x']

        # Filter out zero/negative values
        E_uu_pos = np.where(E_uu > 0, E_uu, np.nan)
        E_vv_pos = np.where(E_vv > 0, E_vv, np.nan)

        # Plot Euu
        axes[0].loglog(frequencies, E_uu_pos, linewidth=2, color=color,
                       label=f'{slice_id} (x={slice_x:.4f})', alpha=0.8)

        # Plot Evv
        axes[1].loglog(frequencies, E_vv_pos, linewidth=2, color=color,
                       label=f'{slice_id} (x={slice_x:.4f})', alpha=0.8)

    # Formatting for Euu
    axes[0].set_xlabel(r'$f^* = fD/U_\infty$', fontsize=12, fontweight='bold')
    axes[0].set_ylabel(r'$E_{uu}$', fontsize=12, fontweight='bold')
    axes[0].set_title(r'Streamwise Velocity Spectra $E_{uu}$ (First Probe per Slice)',
                      fontsize=12, fontweight='bold')
    axes[0].grid(True, alpha=0.3, which='both')
    axes[0].legend(fontsize=10, loc='best')

    # Formatting for Evv
    axes[1].set_xlabel(r'$f^* = fD/U_\infty$', fontsize=12, fontweight='bold')
    axes[1].set_ylabel(r'$E_{vv}$', fontsize=12, fontweight='bold')
    axes[1].set_title(r'Vertical Velocity Spectra $E_{vv}$ (First Probe per Slice)',
                      fontsize=12, fontweight='bold')
    axes[1].grid(True, alpha=0.3, which='both')
    axes[1].legend(fontsize=10, loc='best')

    plt.tight_layout()
    plt.show()


def plot_spectra_rodriguez_style(data: dict, output_dir: str):
    """
    Create stacked energy spectra plot similar to Rodriguez Fig. 11.

    Plots spectra from multiple probes with vertical offset for clarity,
    showing evolution from separated shear layer to wake.
    """
    slice_id = data['slice_id']
    probes = data['probes']
    frequencies = data['f_star']

    if len(probes) == 0:
        print(f"  Warning: No probes found for {slice_id}")
        return

    # Create figure with two subplots (Euu and Evv)
    fig, axes = plt.subplots(1, 2, figsize=(14, 8))

    n_probes = len(probes)

    # Vertical offset factor for stacking spectra
    offset_factor = 50  # Adjust this to control spacing

    # Plot each probe's spectra with vertical offset
    for probe_idx, probe in enumerate(probes):
        y_actual = probe['y_actual']
        probe_name = probe['probe_name']
        E_uu = probe['E_uu']
        E_vv = probe['E_vv']

        # Filter out zero/negative values
        E_uu_pos = np.where(E_uu > 0, E_uu, np.nan)
        E_vv_pos = np.where(E_vv > 0, E_vv, np.nan)

        # Vertical offset (higher index = top of plot)
        offset = offset_factor ** (probe_idx / (n_probes - 1)) if n_probes > 1 else 1

        # Plot Euu with offset
        axes[0].loglog(frequencies, E_uu_pos * offset, linewidth=1.5, alpha=0.8,
                       label=f'{probe_name} (y={y_actual:.4f})')

        # Plot Evv with offset
        axes[1].loglog(frequencies, E_vv_pos * offset, linewidth=1.5, alpha=0.8,
                       label=f'{probe_name} (y={y_actual:.4f})')

    # Add -5/3 reference slope (inertial subrange)
    freq_ref = frequencies[(frequencies > 0.1) & (frequencies < 10)]
    slope_ref = freq_ref ** (-5/3)

    # Normalize to fit in plot
    for ax_idx, ax in enumerate(axes):
        # Scale reference slope for visibility
        scale_factor = 1e3
        ax.loglog(freq_ref, slope_ref * scale_factor, 'k--', linewidth=2,
                  alpha=0.5, label=r'$-5/3$ slope')

    # Formatting for Euu
    axes[0].set_xlabel(r'$f^*$', fontsize=13, fontweight='bold')
    axes[0].set_ylabel(r'$E_{uu}$ (offset)', fontsize=13, fontweight='bold')
    axes[0].set_title(f'{slice_id}: Streamwise Velocity Spectra (Stacked)',
                      fontsize=13, fontweight='bold')
    axes[0].grid(True, alpha=0.3, which='both')
    axes[0].legend(fontsize=9, loc='best')

    # Formatting for Evv
    axes[1].set_xlabel(r'$f^*$', fontsize=13, fontweight='bold')
    axes[1].set_ylabel(r'$E_{vv}$ (offset)', fontsize=13, fontweight='bold')
    axes[1].set_title(f'{slice_id}: Vertical Velocity Spectra (Stacked)',
                      fontsize=13, fontweight='bold')
    axes[1].grid(True, alpha=0.3, which='both')
    axes[1].legend(fontsize=9, loc='best')

    plt.tight_layout()
    plt.show()


def plot_stacked_energy_spectra_all_slices(all_slice_data: dict, output_dir: str):
    """
    Create stacked energy spectra visualization with all probes from all slices in a single plot.

    Each probe has a unique color. Probes are labeled only by their coordinates (x, y).
    """
    if not all_slice_data:
        return

    # Create figure with two subplots (Euu and Evv)
    fig, axes = plt.subplots(1, 2, figsize=(16, 9))

    # Calculate total number of probes for coloring
    slice_ids = sorted(all_slice_data.keys())
    total_probes = sum(len(all_slice_data[sid]['probes']) for sid in slice_ids)
    probe_colors = plt.cm.tab20(np.linspace(0, 1, min(total_probes, 20)))
    if total_probes > 20:
        probe_colors = plt.cm.hsv(np.linspace(0, 1, total_probes))

    # Vertical offset factor for stacking spectra
    offset_factor = 100  # Adjust this to control spacing

    probe_counter = 0

    # Plot each probe from each slice
    for slice_idx, (slice_id, data) in enumerate(sorted(all_slice_data.items())):
        probes = data['probes']
        frequencies = data['f_star']
        slice_x = data['slice_x']

        if len(probes) == 0:
            continue

        for probe_idx, probe in enumerate(probes):
            y_actual = probe['y_actual']
            E_uu = probe['E_uu']
            E_vv = probe['E_vv']

            # Filter out zero/negative values
            E_uu_pos = np.where(E_uu > 0, E_uu, np.nan)
            E_vv_pos = np.where(E_vv > 0, E_vv, np.nan)

            # Vertical offset based on position in all probes
            offset = offset_factor ** (probe_counter / (total_probes - 1)) if total_probes > 1 else 1

            # Create label with only coordinates
            label = f'(x={slice_x:.4f}, y={y_actual:.4f})'
            probe_color = probe_colors[probe_counter] if probe_counter < len(probe_colors) else probe_colors[probe_counter % len(probe_colors)]

            # Plot Euu with offset
            axes[0].loglog(frequencies, E_uu_pos * offset, linewidth=1.5, alpha=0.8,
                           label=label, color=probe_color)

            # Plot Evv with offset
            axes[1].loglog(frequencies, E_vv_pos * offset, linewidth=1.5, alpha=0.8,
                           label=label, color=probe_color)

            probe_counter += 1

    # Add -5/3 reference slope (inertial subrange)
    frequencies = all_slice_data[slice_ids[0]]['f_star']
    freq_ref = frequencies[(frequencies > 0.1) & (frequencies < 10)]
    slope_ref = freq_ref ** (-5/3)

    # Normalize to fit in plot
    for ax_idx, ax in enumerate(axes):
        # Scale reference slope for visibility
        scale_factor = 1e3
        ax.loglog(freq_ref, slope_ref * scale_factor, 'k--', linewidth=2,
                  alpha=0.5, label=r'$-5/3$ slope')

    # Formatting for Euu
    axes[0].set_xlabel(r'$f^*$', fontsize=13, fontweight='bold')
    axes[0].set_ylabel(r'$E_{uu}$ (offset)', fontsize=13, fontweight='bold')
    axes[0].set_title(r'Streamwise Velocity Spectra $E_{uu}$ (All Slices & Probes - Stacked)',
                      fontsize=13, fontweight='bold')
    axes[0].grid(True, alpha=0.3, which='both')
    axes[0].legend(fontsize=8, loc='best', ncol=2)
    axes[0].set_xlim(0.05, 3e2)
    axes[0].set_ylim(1e-10, 1)

    # Formatting for Evv
    axes[1].set_xlabel(r'$f^*$', fontsize=13, fontweight='bold')
    axes[1].set_ylabel(r'$E_{vv}$ (offset)', fontsize=13, fontweight='bold')
    axes[1].set_title(r'Vertical Velocity Spectra $E_{vv}$ (All Slices & Probes - Stacked)',
                      fontsize=13, fontweight='bold')
    axes[1].grid(True, alpha=0.3, which='both')
    axes[1].legend(fontsize=8, loc='best', ncol=2)
    axes[1].set_xlim(0.05, 3e2)
    axes[1].set_ylim(1e-10, 1)

    plt.tight_layout()
    plt.show()



def plot_overlaid_energy_spectra_all_slices(all_slice_data: dict, output_dir: str):
    """
    Create overlaid energy spectra visualization with all probes from all slices in a single plot.

    Each probe has a unique color. Probes are labeled only by their coordinates (x, y).
    No vertical offset is applied - all spectra are on the same scale.
    """
    if not all_slice_data:
        return

    # Create figure with two subplots (Euu and Evv)
    fig, axes = plt.subplots(1, 2, figsize=(16, 9))

    # Calculate total number of probes for coloring
    slice_ids = sorted(all_slice_data.keys())
    total_probes = sum(len(all_slice_data[sid]['probes']) for sid in slice_ids)
    probe_colors = plt.cm.tab20(np.linspace(0, 1, min(total_probes, 20)))
    if total_probes > 20:
        probe_colors = plt.cm.hsv(np.linspace(0, 1, total_probes))

    probe_counter = 0

    # Plot each probe from each slice
    for slice_idx, (slice_id, data) in enumerate(sorted(all_slice_data.items())):
        probes = data['probes']
        frequencies = data['f_star']
        slice_x = data['slice_x']

        if len(probes) == 0:
            continue

        for probe_idx, probe in enumerate(probes):
            y_actual = probe['y_actual']
            E_uu = probe['E_uu']
            E_vv = probe['E_vv']

            # Filter out zero/negative values
            E_uu_pos = np.where(E_uu > 0, E_uu, np.nan)
            E_vv_pos = np.where(E_vv > 0, E_vv, np.nan)

            # Create label with only coordinates
            label = f'(x={slice_x:.4f}, y={y_actual:.4f})'
            probe_color = probe_colors[probe_counter] if probe_counter < len(probe_colors) else probe_colors[probe_counter % len(probe_colors)]

            # Plot Euu without offset
            axes[0].loglog(frequencies, E_uu_pos, linewidth=1.5, alpha=0.7,
                           label=label, color=probe_color)

            # Plot Evv without offset
            axes[1].loglog(frequencies, E_vv_pos, linewidth=1.5, alpha=0.7,
                           label=label, color=probe_color)

            probe_counter += 1

    # Add -5/3 reference slope (inertial subrange)
    frequencies = all_slice_data[slice_ids[0]]['f_star']
    freq_ref = frequencies[(frequencies > 0.1) & (frequencies < 10)]
    slope_ref = freq_ref ** (-5/3)

    # Normalize to fit in plot
    for ax_idx, ax in enumerate(axes):
        # Scale reference slope for visibility
        scale_factor = 1e-1
        ax.loglog(freq_ref, slope_ref * scale_factor, 'k--', linewidth=2,
                  alpha=0.5, label=r'$-5/3$ slope')

    # Formatting for Euu
    axes[0].set_xlabel(r'$f^*$', fontsize=13, fontweight='bold')
    axes[0].set_ylabel(r'$E_{uu}$', fontsize=13, fontweight='bold')
    axes[0].set_title(r'Streamwise Velocity Spectra $E_{uu}$ (All Slices & Probes)',
                      fontsize=13, fontweight='bold')
    axes[0].grid(True, alpha=0.3, which='both')
    axes[0].legend(fontsize=8, loc='best', ncol=2)
    axes[0].set_xlim(0.05, 3e2)
    axes[0].set_ylim(1e-10, 1)

    # Formatting for Evv
    axes[1].set_xlabel(r'$f^*$', fontsize=13, fontweight='bold')
    axes[1].set_ylabel(r'$E_{vv}$', fontsize=13, fontweight='bold')
    axes[1].set_title(r'Vertical Velocity Spectra $E_{vv}$ (All Slices & Probes)',
                      fontsize=13, fontweight='bold')
    axes[1].grid(True, alpha=0.3, which='both')
    axes[1].legend(fontsize=8, loc='best', ncol=2)
    axes[1].set_xlim(0.05, 3e2)
    axes[1].set_ylim(1e-10, 1)

    plt.tight_layout()
    plt.show()



print("="*80)
print("COMPREHENSIVE PROBE AND AIRFOIL VISUALIZATION")
print("="*80)

# Load airfoil surface
print(f"\nLoading airfoil surface from: {GEO_FILE}")
airfoil_surface = load_airfoil_surface(GEO_FILE)
print(f"  Airfoil surface points: {len(airfoil_surface)}")

# Load TKE data
print(f"\nLoading TKE data from: {TKE_DATA_FILE}")
try:
    tke_data = load_tke_data(TKE_DATA_FILE)
    print(f"  TKE field shape: {tke_data['tke'].shape}")
    print(f"  Coordinates: x={tke_data['x_coords'].shape}, y={tke_data['y_coords'].shape}")
except Exception as e:
    print(f"  Warning: Could not load TKE data: {e}")
    tke_data = None

# Discover energy spectra files
print(f"\nSearching for energy spectra files in: {ENERGY_SPECTRA_DIR}")
spectra_files = discover_energy_spectra_files(ENERGY_SPECTRA_DIR)
print(f"  Found {len(spectra_files)} energy spectra files:")
for slice_id in sorted(spectra_files.keys()):
    print(f"    - {slice_id}: {os.path.basename(spectra_files[slice_id])}")

if not spectra_files:
    print("No energy spectra files found. Exiting.")
    sys.exit(1)

# Load probe data from all files
print("\nLoading probe data from all files:")
all_slice_data = {}
for slice_id, h5_file in sorted(spectra_files.items()):
    print(f"  Processing {slice_id}...")
    try:
        data = load_probe_data_from_h5(h5_file)
        all_slice_data[slice_id] = data
        print(f"    ✓ Loaded {len(data['probes'])} probes at x={data['slice_x']:.6f}")
    except Exception as e:
        print(f"    ✗ Error loading {slice_id}: {str(e)}")

# ============================================================================
# CREATE COMBINED OVERVIEW (ALL SLICES) - FIRST PLOT
# ============================================================================

print("\n[1] Creating combined overview plot with TKE, airfoil surface, and probe locations...")

fig, ax = plt.subplots(figsize=(14, 8))

# Plot TKE field as background if available
if tke_data is not None:
    # Normalize TKE by u_infty^2
    u_infty = tke_data['u_infty']
    tke_norm = tke_data['tke'] / (u_infty ** 2)

    # Filter out zero/negative values for log scale
    tke_norm_pos = np.where(tke_norm > 0, tke_norm, np.nan)

    # Determine color scale range
    tke_min = np.nanmin(tke_norm_pos)
    tke_max = np.nanmax(tke_norm_pos)

    # Create log-spaced contour levels
    levels = np.logspace(np.log10(tke_min), np.log10(tke_max), 20)

    # Plot TKE contourf
    cf = ax.contourf(tke_data['x_coords'], tke_data['y_coords'], tke_norm_pos,
                     levels=levels, cmap="viridis",
                     norm=LogNorm(vmin=tke_min, vmax=tke_max), zorder=0, alpha=0.8)

    # Add colorbar for TKE
    cbar = plt.colorbar(cf, ax=ax, label=r"TKE / $u_\infty^2$", fraction=0.02)
    cbar.ax.yaxis.set_major_locator(LogLocator(base=10, numticks=10))
    cbar.ax.yaxis.set_major_formatter(LogFormatterSciNotation(base=10))

# Plot airfoil surface
ax.scatter(airfoil_surface[:, 0], airfoil_surface[:, 1],
           color='k', s=5, label='Airfoil surface', alpha=0.8, zorder=2)

# Plot all slices with their probes
slice_colors_main = plt.cm.tab10(np.linspace(0, 1, len(all_slice_data)))

for (slice_id, data), slice_color in zip(sorted(all_slice_data.items()), slice_colors_main):
    slice_x = data['slice_x']
    probes = data['probes']

    # Plot slice line
    ax.axvline(x=slice_x, color=slice_color, linewidth=2, linestyle='--',
               alpha=0.7, zorder=1)

    # Plot probes on this slice
    probe_colors = plt.cm.viridis(np.linspace(0, 1, len(probes)))
    for probe_idx, (probe, probe_color) in enumerate(zip(probes, probe_colors)):
        y_actual = probe['y_actual']
        ax.plot(slice_x, y_actual, 'o', markersize=8,
                color=probe_color, markeredgecolor=slice_color,
                markeredgewidth=2, zorder=5, alpha=0.9)
        # Add probe label with coordinates
        label_text = f"({slice_x:.2f},{y_actual:.2f})"
        ax.text(slice_x + 0.01, y_actual, label_text, fontsize=8,
                verticalalignment='center', zorder=6)

# Add legend
legend_elements = [
    plt.Line2D([0], [0], color='k', lw=3, label='Airfoil surface'),
]

# Add slice entries
for (slice_id, data), slice_color in zip(sorted(all_slice_data.items()), slice_colors_main):
    n_probes = len(data['probes'])
    legend_elements.append(
        plt.Line2D([0], [0], color=slice_color, lw=2, linestyle='--',
                   label=f'{slice_id} ({n_probes} probes, x={data["slice_x"]:.4f})')
    )

ax.legend(handles=legend_elements, loc='upper right', fontsize=11, ncol=1)

ax.set_xlabel('x (chord)', fontsize=13, fontweight='bold')
ax.set_ylabel('y (chord)', fontsize=13, fontweight='bold')
ax.set_title(f'All Probes and Airfoil Surface\nAOA = {AOA_deg}°, Total probes per slice shown',
             fontsize=14, fontweight='bold')
ax.set_xlim(-0.15, 1.15)
ax.set_ylim(-0.1, 0.5)
ax.grid(True, alpha=0.3)
ax.set_aspect('equal')
ax.margins(0.1)

plt.tight_layout()

# ============================================================================
# CREATE STACKED ENERGY SPECTRA VISUALIZATION - SECOND PLOT (ALL PROBES Euu & Evv)
# ============================================================================

print("\n[2] Creating overlaid energy spectra visualization (all probes from all slices)...")
try:
    plot_overlaid_energy_spectra_all_slices(all_slice_data, OUTPUT_DIR)
except Exception as e:
    print(f"    Warning: Could not create overlaid energy spectra visualization for all slices: {str(e)}")

# print("\n[3] Creating stacked energy spectra visualization (all probes from all slices)...")
# try:
#     plot_stacked_energy_spectra_all_slices(all_slice_data, OUTPUT_DIR)
# except Exception as e:
#     print(f"    Warning: Could not create stacked energy spectra visualization for all slices: {str(e)}")

# ============================================================================
# PRINT SUMMARY
# ============================================================================

print("\n" + "="*80)
print("SUMMARY OF PROBE VISUALIZATION")
print("="*80)
print(f"\nAOA: {AOA_deg}°")
print(f"\nEnergy Spectra Files Found: {len(all_slice_data)}")

for slice_id in sorted(all_slice_data.keys()):
    data = all_slice_data[slice_id]
    slice_x = data['slice_x']
    probes = data['probes']

    print(f"\n{slice_id}:")
    print(f"  Slice x-location: {slice_x:.6f}")
    print(f"  Number of probes: {len(probes)}")
    print(f"  Probe y-locations:")
    for i, probe in enumerate(probes):
        y_actual = probe['y_actual']
        y_target = probe['y_target']
        error = probe['y_distance_error']
        print(f"    P{i}: y_actual={y_actual:.6e}, y_target={y_target:.6e}, error={error:.6e}")

print(f"\nOutput Directory: {OUTPUT_DIR}")
print(f"\nVisualization plots generated (in order):")
print(f"  [1] Overview plot: Probe locations with TKE field and airfoil surface")
print(f"  [2] Overlaid Energy Spectra Visualization: All probes from all slices on same scale (separate Euu and Evv)")
print(f"      (No vertical offset - all probes directly compared)")
print(f"  [3] Stacked Energy Spectra Visualization: All probes from all slices (separate Euu and Evv)")
print(f"      (With vertical offset for clarity - probes differentiated by slice color and labeled with coordinates x,y)")

print("="*80)
