"""
Streamwise Velocity Spectra Visualization
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

ENERGY_SPECTRA_DIR = "/home/jofre/Members/Eduard/Paper2/Simulations/NACA_0012_AOA12_Re50000_1716x1662x128/Mean_data/Energy_spectra"
AOA_deg = 12.0

# Plot settings
LOG_SHIFT_PER_SPECTRUM = 1.3  # Vertical shift in log10 space per spectrum (uniform spacing)
EUU_XLIM = (0.05, 300)
EUU_YLIM = (1e-10, 1e7)

# ============================================================================
# Load probe data
# ============================================================================

def discover_energy_spectra_files(directory: str) -> dict:
    files_dict = {}
    for file_path in Path(directory).glob("energy_spectra_data_*.h5"):
        filename = file_path.name
        match = re.search(r'energy_spectra_data_(slice_\d+)', filename)
        if match:
            slice_id = match.group(1)
            files_dict[slice_id] = str(file_path)
    return files_dict


def load_probe_data_from_h5(h5_file: str) -> dict:
    data = {}
    with h5py.File(h5_file, 'r') as f:
        data['slice_id'] = f.attrs['slice_id']
        data['slice_x'] = f.attrs['slice_x']

        probes = []
        for key in sorted(f.keys()):
            if key.startswith('probe_'):
                grp = f[key]
                probe_info = {
                    'y_actual': grp.attrs['y_actual'],
                    'E_uu': grp['E_uu'][...],
                }
                probes.append(probe_info)

        data['probes'] = probes

        if 'probe_00' in f:
            grp = f['probe_00']
            data['f_star'] = grp['f_star'][...]

    return data


print("="*80)
print("ENERGY SPECTRA Euu VISUALIZATION")
print("="*80)

# Discover and load files
print(f"\nSearching for energy spectra files in: {ENERGY_SPECTRA_DIR}")
spectra_files = discover_energy_spectra_files(ENERGY_SPECTRA_DIR)
print(f"Found {len(spectra_files)} energy spectra files")

if not spectra_files:
    print("No energy spectra files found. Exiting.")
    sys.exit(1)

# Load all probe data
print("\nLoading probe data...")
all_slice_data = {}
for slice_id, h5_file in sorted(spectra_files.items()):
    try:
        data = load_probe_data_from_h5(h5_file)
        all_slice_data[slice_id] = data
        print(f"  ✓ {slice_id}: {len(data['probes'])} probes at x/c={data['slice_x']:.4f}")
    except Exception as e:
        print(f"  ✗ Error loading {slice_id}: {str(e)}")

# ============================================================================
# Plot Euu spectra
# ============================================================================

print("\nCreating Euu spectra plot...")

fig, ax = plt.subplots(figsize=(6, 8))

slice_ids = sorted(all_slice_data.keys())
n_slices = len(slice_ids)

# Global spectrum counter for uniform spacing
spectrum_counter = 0

# Plot each slice with vertical offset
for slice_idx, slice_id in enumerate(slice_ids):
    data = all_slice_data[slice_id]
    probes = data['probes']
    frequencies = data['f_star']
    slice_x = data['slice_x']

    if len(probes) == 0:
        continue

    for probe_idx, probe in enumerate(probes):
        y_actual = probe['y_actual']
        E_uu = probe['E_uu']

        # Filter out zero/negative values
        E_uu_pos = np.where(E_uu > 0, E_uu, np.nan)

        # Uniform vertical offset based on global spectrum index
        log_offset = 10 ** (spectrum_counter * LOG_SHIFT_PER_SPECTRUM)
        spectrum_counter += 1

        label = f'x/c={slice_x:.3f}, y={y_actual:.4f}'
        ax.loglog(frequencies, E_uu_pos * log_offset, linewidth=1.5, alpha=0.8,
                  label=label, color='k')

# Add -5/3 reference slope
frequencies = all_slice_data[slice_ids[0]]['f_star']
freq_ref = frequencies[(frequencies > 3) & (frequencies < 20)]
slope_ref = freq_ref ** (-5/3)
ax.loglog(freq_ref, slope_ref * 1e6, 'k--', linewidth=2, alpha=0.5, label=r'$-5/3$')

# Add text label for slope
mid_idx = len(freq_ref) // 2
ax.text(freq_ref[mid_idx], slope_ref[mid_idx] * 1e6 * 2, r'$-5/3$', fontsize=12,
        fontweight='bold', color='k', ha='center')

# Formatting
ax.set_xlabel(r'$f^*$', fontsize=13, fontweight='bold')
ax.set_ylabel(r'$E_{uu}$', fontsize=13, fontweight='bold')

ax.set_xlim(EUU_XLIM)
if EUU_YLIM is not None:
    ax.set_ylim(EUU_YLIM)

plt.tight_layout()
plt.savefig('/home/jofre/Members/Eduard/Paper2/Figures/Euu_spectra_AoA12.png', dpi=300, bbox_inches='tight')
plt.savefig('/home/jofre/Members/Eduard/Paper2/Figures/Euu_spectra_AoA12.eps', dpi=300, bbox_inches='tight')
plt.show()

print("\nDone!")
