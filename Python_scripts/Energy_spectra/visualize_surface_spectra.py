"""
Surface Variable Spectra Visualization (τ' and p'_w)

Visualizes wall shear stress and pressure energy spectra across two angles of attack
and three chordwise locations. Creates a 2x2 figure comparing AoA=5° and AoA=12°.
"""

import os
import sys
import h5py
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import re

# LaTeX style
plt.rc('text', usetex=True)
plt.rc('font', size=16, family='serif')
plt.rc('text.latex', preamble=r'\usepackage{amsmath} \usepackage{amssymb}')

# ============================================================================
# Configuration
# ============================================================================

CASES = {
    5.0: {
        "label": r"$\alpha=5^\circ$",
        "directory": "/home/jofre/Members/Eduard/Paper2/Simulations/NACA_0012_AOA5_Re50000_1716x1662x128/Mean_data/Energy_spectra/",
    },
    12.0: {
        "label": r"$\alpha=12^\circ$",
        "directory": "/home/jofre/Members/Eduard/Paper2/Simulations/NACA_0012_AOA12_Re50000_1716x1662x128/Mean_data/Energy_spectra/",
    },
}

REQUIRED_SLICES = ["slice_5", "slice_7", "slice_9"]

# Plot settings
PLOT_PREMULTIPLIED_NORMALIZED = False
ADD_MINUS_5_3 = False
FSTAR_XLIM = (0.05, 300)
ETAUTAU_YLIM = None
EPP_YLIM = None

# Color palette for x/c locations
COLORS = {
    0.5: "red",
    0.7: "blue",
    0.9: "green",
}

FIGURE_DIR = "/home/jofre/Members/Eduard/Paper2/Figures"
os.makedirs(FIGURE_DIR, exist_ok=True)

# ============================================================================
# Utility Functions
# ============================================================================

def discover_surface_spectra_files(directory: str) -> dict:
    """
    Search for surface spectra files matching energy_spectra_surface_slice_*.h5.
    Returns dict with slice_id as key and filepath as value.
    """
    files_dict = {}
    for file_path in Path(directory).glob("energy_spectra_surface_*.h5"):
        filename = file_path.name
        match = re.search(r'energy_spectra_surface_(slice_\d+)\.h5', filename)
        if match:
            slice_id = match.group(1)
            files_dict[slice_id] = str(file_path)
    return files_dict


def load_surface_spectra_from_h5(h5_file: str) -> dict:
    """
    Load surface spectra from HDF5 file.
    Returns dict with slice metadata and spectral data.
    """
    data = {}
    with h5py.File(h5_file, 'r') as f:
        # Global attributes
        data['slice_id'] = f.attrs['slice_id']
        data['slice_x'] = f.attrs['slice_x']
        data['AOA_deg'] = f.attrs['AOA_deg']
        data['dt_save'] = f.attrs['dt_save']
        data['fs'] = f.attrs['fs']
        data['n_samples'] = f.attrs['n_samples']
        data['n_z'] = f.attrs['n_z']

        # Wall shear data
        grp_tau = f['tau_shear']
        data['f_star'] = grp_tau['f_star'][...]
        data['E_tautau'] = grp_tau['E_tautau'][...]
        data['var_tau'] = grp_tau.attrs['var_time_mean']
        data['rel_error_tau_percent'] = grp_tau.attrs['rel_error_percent']

        # Pressure data
        grp_p = f['pressure']
        f_star_p = grp_p['f_star'][...]
        data['E_pp'] = grp_p['E_pp'][...]
        data['var_p'] = grp_p.attrs['var_time_mean']
        data['rel_error_p_percent'] = grp_p.attrs['rel_error_percent']

        # Verify f_star arrays match
        if not np.allclose(data['f_star'], f_star_p):
            raise ValueError("f_star arrays in tau_shear and pressure groups do not match")

    return data


def load_case_data(case_config: dict) -> dict:
    """
    Discover and load surface spectra for an AoA case.
    Returns dict with slices sorted by actual slice_x.
    """
    directory = case_config['directory']

    print(f"\nSearching for surface spectra files in: {directory}")
    spectra_files = discover_surface_spectra_files(directory)
    print(f"  Found {len(spectra_files)} energy spectra files")

    if not spectra_files:
        print(f"  WARNING: No surface spectra files found in {directory}")
        return {}

    # Load all slices
    all_slice_data = {}
    for slice_id in REQUIRED_SLICES:
        if slice_id not in spectra_files:
            print(f"  WARNING: Required slice {slice_id} not found")
            continue

        try:
            h5_file = spectra_files[slice_id]
            data = load_surface_spectra_from_h5(h5_file)
            all_slice_data[slice_id] = data
            print(f"  ✓ {slice_id}: x/c={data['slice_x']:.4f}, "
                  f"AoA={data['AOA_deg']:.1f}°")
        except Exception as e:
            print(f"  ✗ Error loading {slice_id}: {str(e)}")

    # Sort by actual x-position
    sorted_slices = sorted(all_slice_data.items(),
                          key=lambda item: item[1]['slice_x'])

    return {slice_id: data for slice_id, data in sorted_slices}


def plot_surface_spectra_2x2(case_data_dict: dict):
    """
    Create 2x2 figure: tau (top), pressure (bottom); AoA 5° (left), 12° (right).
    """
    print("\nCreating 2x2 surface spectra figure...")

    fig, axes = plt.subplots(2, 2, figsize=(14, 11))

    # Panel layout: [tau_5, tau_12], [p_5, p_12]
    cases = [5.0, 12.0]
    panel_positions = [
        (0, 0, 'tau', 5.0),
        (0, 1, 'tau', 12.0),
        (1, 0, 'p', 5.0),
        (1, 1, 'p', 12.0),
    ]

    # Common x-limits
    fstar_min, fstar_max = FSTAR_XLIM
    freq_start = 1

    # Determine y-limits for each row
    all_tau_data = []
    all_p_data = []

    for var_type, aoa in [('tau', 5.0), ('tau', 12.0), ('p', 5.0), ('p', 12.0)]:
        case_slices = case_data_dict[aoa]
        for slice_id, slice_data in case_slices.items():
            if var_type == 'tau':
                E = slice_data['E_tautau'][freq_start:]
            else:
                E = slice_data['E_pp'][freq_start:]
            E_pos = E[E > 0]
            if len(E_pos) > 0:
                if var_type == 'tau':
                    all_tau_data.extend(E_pos)
                else:
                    all_p_data.extend(E_pos)

    # Auto y-limits if not set
    if ETAUTAU_YLIM is None and len(all_tau_data) > 0:
        tau_ylim = (np.min(all_tau_data) * 0.5, np.max(all_tau_data) * 2)
    else:
        tau_ylim = ETAUTAU_YLIM

    if EPP_YLIM is None and len(all_p_data) > 0:
        p_ylim = (np.min(all_p_data) * 0.5, np.max(all_p_data) * 2)
    else:
        p_ylim = EPP_YLIM

    # Plot each panel
    for row, col, var_type, aoa in panel_positions:
        ax = axes[row, col]
        case_slices = case_data_dict[aoa]

        if not case_slices:
            print(f"  No data for AoA={aoa}°")
            continue

        # Plot each slice
        for slice_id, slice_data in case_slices.items():
            f_star = slice_data['f_star'][freq_start:]
            slice_x = slice_data['slice_x']

            if var_type == 'tau':
                E = slice_data['E_tautau'][freq_start:]
                var = slice_data['var_tau']
                rel_error = slice_data['rel_error_tau_percent']
            else:
                E = slice_data['E_pp'][freq_start:]
                var = slice_data['var_p']
                rel_error = slice_data['rel_error_p_percent']

            # Warn if variance recovery error is large
            if rel_error > 1e-4:
                print(f"  WARNING: Large variance error for {slice_id} "
                      f"(AoA={aoa}°): {rel_error:.6e}%")

            # Replace non-positive with NaN
            E_plot = np.where(E > 0, E, np.nan)

            # Apply premultiplied normalization if requested
            if PLOT_PREMULTIPLIED_NORMALIZED:
                E_plot = f_star * E_plot / var

            # Get color from x-position (round to nearest in our palette)
            color_key = min(COLORS.keys(), key=lambda k: abs(k - slice_x))
            color = COLORS[color_key]

            # Plot
            label = rf"$x/c={slice_x:.2f}$"
            ax.loglog(f_star, E_plot, linewidth=1.6, alpha=0.85, label=label, color=color)

        # Add -5/3 reference slope if requested
        if ADD_MINUS_5_3:
            f_star_all = case_slices[list(case_slices.keys())[0]]['f_star']
            freq_ref = f_star_all[(f_star_all > 3) & (f_star_all < 20)]
            slope_ref = freq_ref ** (-5/3)
            ax.loglog(freq_ref, slope_ref * 1e6, 'k--', linewidth=1.5, alpha=0.4)

        # Formatting
        ax.set_xlim(FSTAR_XLIM)

        if var_type == 'tau':
            ylabel = r"$E_{\tau\tau}$"
            if PLOT_PREMULTIPLIED_NORMALIZED:
                ylabel = r"$f^* E_{\tau\tau}/\sigma_{\tau}^{2}$"
            if tau_ylim is not None:
                ax.set_ylim(tau_ylim)
            title = rf"$\tau_w^\prime$, $\alpha={aoa:.0f}^\circ$"
        else:
            ylabel = r"$E_{pp}$"
            if PLOT_PREMULTIPLIED_NORMALIZED:
                ylabel = r"$f^* E_{pp}/\sigma_{p}^{2}$"
            if p_ylim is not None:
                ax.set_ylim(p_ylim)
            title = rf"$p_w^\prime$, $\alpha={aoa:.0f}^\circ$"
            xlabel = r"$f^*$"

        ax.set_ylabel(ylabel, fontsize=13, fontweight='bold')
        if row == 1:
            ax.set_xlabel(xlabel, fontsize=13, fontweight='bold')
        ax.set_title(title, fontsize=13, fontweight='bold')

        # Add legend only in right column
        if col == 1:
            ax.legend(loc='upper right', fontsize=16, frameon=False)

    plt.tight_layout()

    # Save
    output_base = "surface_spectra_2x2"
    png_file = os.path.join(FIGURE_DIR, f"{output_base}.png")
    eps_file = os.path.join(FIGURE_DIR, f"{output_base}.eps")

    plt.savefig(png_file, dpi=300, bbox_inches='tight')
    plt.savefig(eps_file, dpi=300, bbox_inches='tight')
    print(f"\nFigure saved:")
    print(f"  PNG: {png_file}")
    print(f"  EPS: {eps_file}")

    plt.show()


def print_diagnostic_summary(case_data_dict: dict):
    """
    Print diagnostic summary of loaded spectra.
    """
    print("\n" + "="*80)
    print("DIAGNOSTIC SUMMARY")
    print("="*80)

    for aoa in sorted(case_data_dict.keys()):
        print(f"\nAoA = {aoa}°")
        case_slices = case_data_dict[aoa]

        for slice_id, slice_data in case_slices.items():
            slice_x = slice_data['slice_x']
            rel_error_tau = slice_data['rel_error_tau_percent']
            rel_error_p = slice_data['rel_error_p_percent']

            E_tau = slice_data['E_tautau']
            E_p = slice_data['E_pp']

            E_tau_pos = E_tau[E_tau > 0]
            E_p_pos = E_p[E_p > 0]

            print(f"  {slice_id} (x/c={slice_x:.4f}):")
            print(f"    τ' variance error:  {rel_error_tau:.6e}%")
            print(f"    p' variance error:  {rel_error_p:.6e}%")

            if len(E_tau_pos) > 0:
                print(f"    E_ττ range: [{E_tau_pos.min():.3e}, {E_tau_pos.max():.3e}]")
            if len(E_p_pos) > 0:
                print(f"    E_pp range:  [{E_p_pos.min():.3e}, {E_p_pos.max():.3e}]")

    print("="*80)


# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    print("="*80)
    print("SURFACE VARIABLE SPECTRA VISUALIZATION")
    print("="*80)

    # Load data for both AoA cases
    case_data_dict = {}
    for aoa, case_config in sorted(CASES.items()):
        print(f"\n{'─'*80}")
        print(f"Loading AoA = {aoa}°")
        print(f"{'─'*80}")
        case_data = load_case_data(case_config)
        case_data_dict[aoa] = case_data

    # Print diagnostics
    print_diagnostic_summary(case_data_dict)

    # Create figure
    plot_surface_spectra_2x2(case_data_dict)

    print("\nDone. Surface spectra figure saved.")
