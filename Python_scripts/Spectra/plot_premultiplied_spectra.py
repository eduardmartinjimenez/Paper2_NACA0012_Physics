import os
import h5py
import numpy as np
import matplotlib.pyplot as plt

# ============================================================================
# Configuration
# ============================================================================

# Path to the saved spectra data (contains multiple chord locations)
DATA_DIR = "/home/jofre/Members/Eduard/Paper2/Simulations/NACA_0012_AOA5_Re50000_1716x1662x128/Mean_data/Premultiplied_spectra"

# Output figure path
OUTPUT_FILE = os.path.join(DATA_DIR, "premultiplied_spectra_all_locations.png")

# ============================================================================
# Load data
# ============================================================================

if not os.path.isdir(DATA_DIR):
    raise FileNotFoundError(f"DATA_DIR not found: {DATA_DIR}")

data_files = sorted(
    f for f in os.listdir(DATA_DIR)
    if f.endswith(".h5") and "premultiplied_spectra_data" in f
)

if len(data_files) == 0:
    raise FileNotFoundError(f"No premultiplied spectra files found in: {DATA_DIR}")

print(f"Found {len(data_files)} premultiplied spectra files:")
for fname in data_files:
    print(f"  - {fname}")

datasets = []
global_min = np.inf
global_max = -np.inf

for fname in data_files:
    fpath = os.path.join(DATA_DIR, fname)
    with h5py.File(fpath, "r") as f:
        kz_plus = f["kz_plus"][...]
        y_plus = f["y_plus"][...]
        premult_psd = f["premultiplied_psd"][...]

        slice_x = float(f.attrs.get("slice_x", np.nan))
        u_tau = float(f.attrs.get("u_tau", np.nan))
        snapshot_count = int(f.attrs.get("snapshot_count", -1))
        L_z = float(f.attrs.get("spanwise_domain", np.nan))

    lambda_z_plus = (2.0 * np.pi) / kz_plus
    log10_lambda_z_plus = np.log10(lambda_z_plus)

    global_min = min(global_min, np.nanmin(premult_psd))
    global_max = max(global_max, np.nanmax(premult_psd))

    datasets.append({
        "file": fname,
        "kz_plus": kz_plus,
        "y_plus": y_plus,
        "premult_psd": premult_psd,
        "log10_lambda_z_plus": log10_lambda_z_plus,
        "slice_x": slice_x,
        "u_tau": u_tau,
        "snapshot_count": snapshot_count,
        "L_z": L_z,
    })

# ============================================================================
# Create visualization for all locations
# ============================================================================

# Sort by chord location and limit to 5 plots
datasets = sorted(datasets, key=lambda d: d["slice_x"])
if len(datasets) > 5:
    datasets = datasets[:5]

n_plots = len(datasets)
fig, axes = plt.subplots(1, n_plots, figsize=(3.2 * n_plots, 3.2), sharey=True)

if n_plots == 1:
    axes = [axes]

print(f"\nGlobal spectrum range: {global_min:.2e} to {global_max:.2e}")
n_levels = 30
levels = np.linspace(global_min, global_max, n_levels)

for ax, data in zip(axes, datasets):
    X, Y = np.meshgrid(data["log10_lambda_z_plus"], data["y_plus"])
    Z = data["premult_psd"]

    contour_fill = ax.contourf(X, Y, Z, levels=levels, cmap='RdYlBu_r', extend='both')

    ax.set_yscale('log')
    ax.set_ylabel(r'$y^+$', fontsize=11)
    ax.tick_params(which='both', labelsize=10)

    ax.set_title(
        f'$x_{{ss}}/c = {data["slice_x"]:.3f}$, $N_{{snapshots}} = {data["snapshot_count"]}$',
        fontsize=11,
        pad=8,
    )

    ax.grid(True, which='both', alpha=0.3, linestyle='--', linewidth=0.5)
    ax.grid(True, which='major', alpha=0.5, linestyle='-', linewidth=0.7)

    ax.set_ylim(data["y_plus"].min(), data["y_plus"].max())
    ax.set_box_aspect(1)

for ax in axes:
    ax.set_xlabel(r'$\log_{10}(\lambda_z^+)$', fontsize=11)

cbar = fig.colorbar(contour_fill, ax=axes, pad=0.02)
cbar.set_label(r'$k_z\,\Phi_{u_tu_t}/u_\tau^2$', fontsize=12, labelpad=15)
cbar.ax.tick_params(labelsize=10)

fig.suptitle('Inner-scaled spanwise premultiplied power spectral density', fontsize=13, y=0.995)
plt.tight_layout()
# plt.savefig(OUTPUT_FILE, dpi=300, bbox_inches='tight')
# print(f"\nFigure saved to: {OUTPUT_FILE}")
plt.show()
