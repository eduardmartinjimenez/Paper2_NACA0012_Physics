import os
import sys
import h5py
import numpy as np
from glob import glob
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# ============================================================================
# Configuration  (keep in sync with wall_shear_correlations_2.py)
# ============================================================================

BASE_SURFACE_DIR = "/home/jofre/Members/Eduard/Paper2/Simulations/NACA_0012_AOA12_Re50000_1716x1662x128/Snapshots/"
BATCH_PATTERN    = "batch_*"

GEO_PATH = "/home/jofre/Members/Eduard/Paper2/Simulations/NACA_0012_AOA12_Re50000_1716x1662x128/Geometrical_data/"
GEO_NAME = "3d_NACA0012_Re50000_AoA12_Geometrical_Data.h5"
GEO_FILE = os.path.join(GEO_PATH, GEO_NAME)

OUTPUT_DIR = "/home/jofre/Members/Eduard/Paper2/Simulations/NACA_0012_AOA12_Re50000_1716x1662x128/Mean_data/Alpha_study/"
os.makedirs(OUTPUT_DIR, exist_ok=True)

X_C_LOCATIONS = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

# Alpha values to evaluate
ALPHA_VALUES = np.arange(0.0, 4.1, 0.25)

# ============================================================================
# Load geometrical data and find suction-side points
# ============================================================================
print("=" * 70)
print("LOADING GEOMETRICAL DATA")
print("=" * 70)

with h5py.File(GEO_FILE, "r") as f:
    interface_points    = f["interface_points"][:]
    interface_indices_i = f["interface_indices_i"][:]
    interface_indices_j = f["interface_indices_j"][:]

x_interface = interface_points[:, 0]
y_interface = interface_points[:, 1]

upper_mask = y_interface > np.mean(y_interface)

# Closest suction-side point for each target x/c
point_indices = {}
for x_c_target in X_C_LOCATIONS:
    upper_idx = np.where(upper_mask)[0]
    closest   = upper_idx[np.argmin(np.abs(x_interface[upper_idx] - x_c_target))]
    point_indices[x_c_target] = {
        'global_idx': closest,
        'x_c_actual': x_interface[closest],
        'y':          y_interface[closest],
    }
    print(f"  x/c = {x_c_target:.2f}  ->  global idx {closest}, "
          f"actual x/c = {x_interface[closest]:.4f}")

# ============================================================================
# Discover surface data files
# ============================================================================
print("\n" + "=" * 70)
print("SEARCHING FOR SURFACE DATA FILES")
print("=" * 70)

all_surface_files = []
for batch_dir in sorted(glob(os.path.join(BASE_SURFACE_DIR, BATCH_PATTERN))):
    surface_dir = os.path.join(batch_dir, "Surface_data")
    if not os.path.exists(surface_dir):
        continue
    all_surface_files.extend(sorted(glob(os.path.join(surface_dir, "surface_*.h5"))))

N_total = len(all_surface_files)
print(f"  Found {N_total} surface snapshots")
if N_total == 0:
    raise RuntimeError("No surface data files found!")

# ============================================================================
# Pass 1: accumulate sum and sum-of-squares to compute mean and RMS
# ============================================================================
print("\n" + "=" * 70)
print("PASS 1 — COMPUTING MEAN AND RMS OF TAU_W")
print("=" * 70)

sum1  = {xc: 0.0 for xc in X_C_LOCATIONS}   # sum(tau_w)
sum2  = {xc: 0.0 for xc in X_C_LOCATIONS}   # sum(tau_w^2)
count = {xc: 0   for xc in X_C_LOCATIONS}   # total scalar samples

for idx, fpath in enumerate(all_surface_files):
    if (idx + 1) % 50 == 0 or idx == 0:
        print(f"  File {idx+1}/{N_total}...", flush=True)
    try:
        with h5py.File(fpath, "r") as f:
            tau_w = f["tau_w"][:]   # (Nz, N_surf)
        for xc, info in point_indices.items():
            col = tau_w[:, info['global_idx']]   # (Nz,)
            sum1[xc]  += float(np.sum(col))
            sum2[xc]  += float(np.sum(col ** 2))
            count[xc] += col.size
    except Exception as e:
        print(f"  [WARNING] {fpath}: {e}")

tau_mean = {xc: sum1[xc] / count[xc]                               for xc in X_C_LOCATIONS}
tau_rms  = {xc: np.sqrt(sum2[xc] / count[xc] - tau_mean[xc] ** 2) for xc in X_C_LOCATIONS}

print("\n  Results:")
for xc in X_C_LOCATIONS:
    print(f"    x/c = {xc:.2f}:  <tau_w> = {tau_mean[xc]:.4e},  tau_rms = {tau_rms[xc]:.4e}")

# ============================================================================
# Pass 2: collect normalised fluctuations  tau' / tau_rms
# ============================================================================
print("\n" + "=" * 70)
print("PASS 2 — COLLECTING NORMALISED FLUCTUATIONS")
print("=" * 70)

tau_norm = {xc: [] for xc in X_C_LOCATIONS}

for idx, fpath in enumerate(all_surface_files):
    if (idx + 1) % 50 == 0 or idx == 0:
        print(f"  File {idx+1}/{N_total}...", flush=True)
    try:
        with h5py.File(fpath, "r") as f:
            tau_w = f["tau_w"][:]
        for xc, info in point_indices.items():
            col = tau_w[:, info['global_idx']]
            tau_norm[xc].extend((col - tau_mean[xc]) / tau_rms[xc])
    except Exception as e:
        print(f"  [WARNING] {fpath}: {e}")

for xc in X_C_LOCATIONS:
    tau_norm[xc] = np.array(tau_norm[xc])

print(f"\n  Samples per location: {tau_norm[X_C_LOCATIONS[0]].size}")

# ============================================================================
# Print event-fraction table
# ============================================================================
print("\n" + "=" * 70)
print("EVENT FRACTION TABLE  (fraction of samples with |tau'| > alpha*tau_rms)")
print("=" * 70)

header = f"{'alpha':>6s}" + "".join(f"  x/c={xc:.1f}" for xc in X_C_LOCATIONS)
print(header)
print("-" * len(header))

for alpha in ALPHA_VALUES:
    row = f"{alpha:6.2f}"
    for xc in X_C_LOCATIONS:
        frac = np.mean(np.abs(tau_norm[xc]) > alpha)
        row += f"  {frac:7.4f}"
    print(row)

# ============================================================================
# Print positive fluctuation event-fraction table
# ============================================================================
print("\n" + "=" * 70)
print("POSITIVE EVENT FRACTION TABLE  (fraction of samples with tau' > alpha*tau_rms)")
print("=" * 70)

header = f"{'alpha':>6s}" + "".join(f"  x/c={xc:.1f}" for xc in X_C_LOCATIONS)
print(header)
print("-" * len(header))

for alpha in ALPHA_VALUES:
    row = f"{alpha:6.2f}"
    for xc in X_C_LOCATIONS:
        frac = np.mean(tau_norm[xc] > alpha)
        row += f"  {frac:7.4f}"
    print(row)

# ============================================================================
# Print negative fluctuation event-fraction table
# ============================================================================
print("\n" + "=" * 70)
print("NEGATIVE EVENT FRACTION TABLE  (fraction of samples with tau' < -alpha*tau_rms)")
print("=" * 70)

header = f"{'alpha':>6s}" + "".join(f"  x/c={xc:.1f}" for xc in X_C_LOCATIONS)
print(header)
print("-" * len(header))

for alpha in ALPHA_VALUES:
    row = f"{alpha:6.2f}"
    for xc in X_C_LOCATIONS:
        frac = np.mean(tau_norm[xc] < -alpha)
        row += f"  {frac:7.4f}"
    print(row)

# ============================================================================
# Plot 1: PDF of tau'_w / tau_rms at every x/c location
# ============================================================================
print("\n" + "=" * 70)
print("PLOTTING PDFs")
print("=" * 70)

n_xc  = len(X_C_LOCATIONS)
ncols = 4
nrows = int(np.ceil(n_xc / ncols))

fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3.5 * nrows),
                         sharex=True, sharey=False)
axes = axes.flatten()

bins = np.linspace(-6, 6, 120)
bin_centers = 0.5 * (bins[:-1] + bins[1:])
x_gauss = np.linspace(-6, 6, 300)
gauss   = np.exp(-0.5 * x_gauss ** 2) / np.sqrt(2 * np.pi)

for i, xc in enumerate(X_C_LOCATIONS):
    ax   = axes[i]
    data = tau_norm[xc]

    hist, _ = np.histogram(data, bins=bins, density=True)
    ax.semilogy(bin_centers, hist, color='royalblue', lw=1.5, label='PDF')
    ax.semilogy(x_gauss, gauss, 'k--', lw=1, label='Gaussian')

    # Mark a few representative alpha levels
    for alpha_mark, col in [(1.0, 'orange'), (1.5, 'red'), (2.0, 'darkred')]:
        frac = np.mean(np.abs(data) > alpha_mark)
        ax.axvline( alpha_mark, color=col, lw=1.2, ls='--', alpha=0.8)
        ax.axvline(-alpha_mark, color=col, lw=1.2, ls='--', alpha=0.8,
                   label=f'α={alpha_mark:.1f} ({frac*100:.1f}%)')

    ax.set_title(f'x/c = {xc:.2f}', fontsize=10)
    ax.set_xlabel(r"$\tau'_w \,/\, \tau_{rms}$", fontsize=8)
    ax.set_ylabel('PDF', fontsize=8)
    ax.legend(fontsize=6, loc='upper right')
    ax.set_xlim(-6, 6)
    ax.grid(True, alpha=0.3, which='both')

# Hide unused axes
for j in range(n_xc, len(axes)):
    axes[j].set_visible(False)

fig.suptitle(r'PDF of $\tau^\prime_w / \tau_{rms}$ at suction-side locations', fontsize=13)
plt.tight_layout()
# pdf_path = os.path.join(OUTPUT_DIR, 'pdf_tau_prime_xc_all.png')
# plt.savefig(pdf_path, dpi=150)
# plt.close()
# print(f"  Saved: {pdf_path}")

# ============================================================================
# Plot 2: Event fraction vs alpha (one line per x/c location)
# ============================================================================
print("\nPLOTTING EVENT FRACTION vs ALPHA")

fractions = np.zeros((len(ALPHA_VALUES), n_xc))
for j, xc in enumerate(X_C_LOCATIONS):
    for i, alpha in enumerate(ALPHA_VALUES):
        fractions[i, j] = np.mean(np.abs(tau_norm[xc]) > alpha)

fig, ax = plt.subplots(figsize=(8, 5))
cmap = plt.cm.viridis(np.linspace(0, 1, n_xc))

for j, xc in enumerate(X_C_LOCATIONS):
    ax.plot(ALPHA_VALUES, fractions[:, j] * 100,
            color=cmap[j], lw=1.8, marker='o', ms=3, label=f'x/c={xc:.1f}')

# Reference lines
for ref, ls in [(0.05, '--'), (0.15, '-.'), (0.30, ':')]:
    ax.axhline(ref * 100, color='gray', lw=1, ls=ls,
               label=f'{int(ref*100)}% events')

ax.set_xlabel(r'$\alpha$', fontsize=13)
ax.set_ylabel('PF+NF event fraction  [%]', fontsize=12)
ax.set_title(r'Event fraction  ($|\tau^\prime_w| > \alpha\,\tau_{rms}$)', fontsize=13)
ax.legend(fontsize=8, ncol=2)
ax.grid(True, alpha=0.3)
ax.set_xlim(ALPHA_VALUES[0], ALPHA_VALUES[-1])
ax.set_ylim(0, 105)

# frac_path = os.path.join(OUTPUT_DIR, 'event_fraction_vs_alpha.png')
# plt.tight_layout()
# plt.savefig(frac_path, dpi=150)
# plt.close()
# print(f"  Saved: {frac_path}")

# ============================================================================
# Plot 3: PF and NF fractions separately vs alpha
# ============================================================================
print("\nPLOTTING PF/NF FRACTIONS SEPARATELY")

frac_PF = np.zeros((len(ALPHA_VALUES), n_xc))
frac_NF = np.zeros((len(ALPHA_VALUES), n_xc))
for j, xc in enumerate(X_C_LOCATIONS):
    for i, alpha in enumerate(ALPHA_VALUES):
        frac_PF[i, j] = np.mean(tau_norm[xc] >  alpha)
        frac_NF[i, j] = np.mean(tau_norm[xc] < -alpha)

fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)

for j, xc in enumerate(X_C_LOCATIONS):
    axes[0].plot(ALPHA_VALUES, frac_PF[:, j] * 100,
                 color=cmap[j], lw=1.8, marker='o', ms=3, label=f'x/c={xc:.1f}')
    axes[1].plot(ALPHA_VALUES, frac_NF[:, j] * 100,
                 color=cmap[j], lw=1.8, marker='o', ms=3, label=f'x/c={xc:.1f}')

for ax, title in zip(axes, ['PF events  ($\\tau^\\prime > \\alpha\\,\\tau_{rms}$)',
                              'NF events  ($\\tau^\\prime < -\\alpha\\,\\tau_{rms}$)']):
    ax.set_xlabel(r'$\alpha$', fontsize=13)
    ax.set_ylabel('Fraction [%]', fontsize=12)
    ax.set_title(title, fontsize=12)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(ALPHA_VALUES[0], ALPHA_VALUES[-1])
    ax.set_ylim(0, 55)

fig.suptitle('PF and NF event fractions vs threshold', fontsize=13)
plt.tight_layout()
# pf_nf_path = os.path.join(OUTPUT_DIR, 'pf_nf_fraction_vs_alpha.png')
# plt.savefig(pf_nf_path, dpi=150)
# plt.close()
# print(f"  Saved: {pf_nf_path}")
plt.show()

print("\n" + "=" * 70)
print("ALPHA STUDY COMPLETE")
print("=" * 70)
print(f"All outputs saved to: {OUTPUT_DIR}")
