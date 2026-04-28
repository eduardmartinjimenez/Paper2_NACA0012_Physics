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

# Tail probabilities to evaluate (fraction of samples in each tail)
P_VALUES = [0.30, 0.20, 0.15, 0.10, 0.05, 0.02, 0.01]

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
# Precompute quantile thresholds and equivalent alpha values for every (p, x/c)
# ============================================================================
# tau_norm[xc] = (tau_w - tau_mean) / tau_rms, so quantiles on it are already
# in rms units: alpha_equiv = quantile value directly (no further division).

n_xc = len(X_C_LOCATIONS)
n_p  = len(P_VALUES)
cmap = plt.cm.viridis(np.linspace(0, 1, n_xc))

# thresholds in normalised (rms) units,  shape (n_p, n_xc)
theta_plus_norm  = np.zeros((n_p, n_xc))   # upper quantile at (1-p)
theta_minus_norm = np.zeros((n_p, n_xc))   # lower quantile at p

# measured fractions — should be ~p by construction; printed for verification
frac_PF    = np.zeros((n_p, n_xc))
frac_NF    = np.zeros((n_p, n_xc))
frac_total = np.zeros((n_p, n_xc))

for j, xc in enumerate(X_C_LOCATIONS):
    data = tau_norm[xc]
    for i, p in enumerate(P_VALUES):
        q_plus  = float(np.quantile(data, 1.0 - p))
        q_minus = float(np.quantile(data, p))
        theta_plus_norm[i, j]  = q_plus
        theta_minus_norm[i, j] = q_minus
        frac_PF[i, j]    = np.mean(data >= q_plus)
        frac_NF[i, j]    = np.mean(data <= q_minus)
        frac_total[i, j] = frac_PF[i, j] + frac_NF[i, j]

# alpha_equiv = threshold magnitude in rms units (positive)
alpha_plus_equiv  =  theta_plus_norm           # already >= 0
alpha_minus_equiv = -theta_minus_norm          # flip sign to get positive value

# ============================================================================
# Per-p detailed tables
# ============================================================================
print("\n" + "=" * 70)
print("PER-p THRESHOLD TABLES")
print("=" * 70)

xc_header = "".join(f"  x/c={xc:.1f}" for xc in X_C_LOCATIONS)

for i, p in enumerate(P_VALUES):
    print(f"\n--- p = {p:.2f} ---")
    print(f"  {'':20s}{xc_header}")

    rows = {
        'theta+ / tau_rms' : [f"{theta_plus_norm[i, j]:+7.4f}" for j in range(n_xc)],
        'theta- / tau_rms' : [f"{theta_minus_norm[i, j]:+7.4f}" for j in range(n_xc)],
        'alpha+_equiv'     : [f"{alpha_plus_equiv[i, j]:7.4f}"  for j in range(n_xc)],
        'alpha-_equiv'     : [f"{alpha_minus_equiv[i, j]:7.4f}" for j in range(n_xc)],
        'PF frac (actual)' : [f"{frac_PF[i, j]:7.4f}"          for j in range(n_xc)],
        'NF frac (actual)' : [f"{frac_NF[i, j]:7.4f}"          for j in range(n_xc)],
        'Total frac'       : [f"{frac_total[i, j]:7.4f}"        for j in range(n_xc)],
    }
    for label, vals in rows.items():
        print(f"  {label:20s}" + "  " + "  ".join(vals))

# ============================================================================
# Summary tables: alpha_equiv vs p
# ============================================================================
print("\n" + "=" * 70)
print("ALPHA+ EQUIV  (theta+ / tau_rms)  vs  p")
print("=" * 70)
hdr = f"{'p':>6s}{xc_header}"
print(hdr)
print("-" * len(hdr))
for i, p in enumerate(P_VALUES):
    print(f"{p:6.2f}" + "".join(f"   {alpha_plus_equiv[i, j]:7.4f}" for j in range(n_xc)))

print("\n" + "=" * 70)
print("ALPHA- EQUIV  (|theta-| / tau_rms)  vs  p")
print("=" * 70)
print(hdr)
print("-" * len(hdr))
for i, p in enumerate(P_VALUES):
    print(f"{p:6.2f}" + "".join(f"   {alpha_minus_equiv[i, j]:7.4f}" for j in range(n_xc)))

# ============================================================================
# Plot 1: PDFs with quantile threshold markers
# ============================================================================
print("\n" + "=" * 70)
print("PLOTTING PDFs WITH QUANTILE MARKERS")
print("=" * 70)

ncols = 4
nrows = int(np.ceil(n_xc / ncols))
fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3.5 * nrows),
                         sharex=True, sharey=False)
axes = axes.flatten()

bins        = np.linspace(-6, 6, 120)
bin_centers = 0.5 * (bins[:-1] + bins[1:])
x_gauss     = np.linspace(-6, 6, 300)
gauss       = np.exp(-0.5 * x_gauss ** 2) / np.sqrt(2 * np.pi)

p_marks       = [0.10, 0.05, 0.02]
p_mark_colors = ['orange', 'red', 'darkred']

for i, xc in enumerate(X_C_LOCATIONS):
    ax   = axes[i]
    data = tau_norm[xc]

    hist, _ = np.histogram(data, bins=bins, density=True)
    ax.semilogy(bin_centers, hist, color='royalblue', lw=1.5, label='PDF')
    ax.semilogy(x_gauss, gauss, 'k--', lw=1, label='Gaussian')

    for p_mark, col in zip(p_marks, p_mark_colors):
        q_plus  = float(np.quantile(data, 1.0 - p_mark))
        q_minus = float(np.quantile(data, p_mark))
        ax.axvline(q_plus,  color=col, lw=1.2, ls='--', alpha=0.8)
        ax.axvline(q_minus, color=col, lw=1.2, ls='--', alpha=0.8,
                   label=f'p={p_mark:.2f}  α+={q_plus:.2f} / α−={abs(q_minus):.2f}')

    ax.set_title(f'x/c = {xc:.2f}', fontsize=10)
    ax.set_xlabel(r"$\tau'_w \,/\, \tau_{rms}$", fontsize=8)
    ax.set_ylabel('PDF', fontsize=8)
    ax.legend(fontsize=6, loc='upper right')
    ax.set_xlim(-6, 6)
    ax.grid(True, alpha=0.3, which='both')

for j in range(n_xc, len(axes)):
    axes[j].set_visible(False)

fig.suptitle(r'PDF of $\tau^\prime_w / \tau_{rms}$ — quantile thresholds marked',
             fontsize=13)
plt.tight_layout()
pdf_path = os.path.join(OUTPUT_DIR, 'pdf_tau_prime_quantile_markers.png')
plt.savefig(pdf_path, dpi=150)
print(f"  Saved: {pdf_path}")

# # ============================================================================
# # Plot 2: PF, NF, and Total fractions vs p
# # ============================================================================
# print("\nPLOTTING PF / NF / TOTAL FRACTIONS vs p")

p_arr = np.array(P_VALUES) * 100   # convert to %

# fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=False)

# for j, xc in enumerate(X_C_LOCATIONS):
#     kw = dict(color=cmap[j], lw=1.8, marker='o', ms=4, label=f'x/c={xc:.1f}')
#     axes[0].plot(p_arr, frac_PF[:, j]    * 100, **kw)
#     axes[1].plot(p_arr, frac_NF[:, j]    * 100, **kw)
#     axes[2].plot(p_arr, frac_total[:, j] * 100, **kw)

# # Ideal reference lines
# p_ref = np.linspace(0, max(P_VALUES), 100) * 100
# axes[0].plot(p_ref, p_ref,     'k--', lw=1, label='ideal (= p)')
# axes[1].plot(p_ref, p_ref,     'k--', lw=1, label='ideal (= p)')
# axes[2].plot(p_ref, 2 * p_ref, 'k--', lw=1, label='ideal (= 2p)')

# titles = [r'PF fraction  ($\tau^\prime \geq \theta_+$)',
#           r'NF fraction  ($\tau^\prime \leq \theta_-$)',
#           r'Total  PF + NF']
# for ax, title in zip(axes, titles):
#     ax.set_xlabel('p  [%]', fontsize=12)
#     ax.set_ylabel('Measured fraction  [%]', fontsize=11)
#     ax.set_title(title, fontsize=12)
#     ax.legend(fontsize=7, ncol=2)
#     ax.grid(True, alpha=0.3)
#     ax.set_xlim(0, max(P_VALUES) * 100 * 1.05)

# fig.suptitle('PF / NF / Total fractions vs tail probability p', fontsize=13)
# plt.tight_layout()
# frac_path = os.path.join(OUTPUT_DIR, 'pf_nf_fraction_vs_p.png')
# plt.savefig(frac_path, dpi=150)
# print(f"  Saved: {frac_path}")

# ============================================================================
# Plot 3: alpha_plus_equiv and alpha_minus_equiv vs p
# ============================================================================
print("\nPLOTTING EQUIVALENT ALPHA VALUES vs p")

fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=False)

for j, xc in enumerate(X_C_LOCATIONS):
    kw = dict(color=cmap[j], lw=1.8, marker='o', ms=4, label=f'x/c={xc:.1f}')
    axes[0].plot(p_arr, alpha_plus_equiv[:, j],  **kw)
    axes[1].plot(p_arr, alpha_minus_equiv[:, j], **kw)

for ax, title in zip(axes,
                     [r'$\alpha_+$ equiv  ($\theta_+ / \tau_{rms}$)',
                      r'$\alpha_-$ equiv  ($|\theta_-| / \tau_{rms}$)']):
    ax.set_xlabel('p  [%]', fontsize=12)
    ax.set_ylabel(r'$\alpha_{\rm equiv}$', fontsize=12)
    ax.set_title(title, fontsize=12)
    ax.legend(fontsize=7, ncol=2)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, max(P_VALUES) * 100 * 1.05)
    ax.set_ylim(bottom=0)

fig.suptitle(r'Equivalent $\alpha$ (in $\tau_{rms}$ units) vs tail probability p',
             fontsize=13)
plt.tight_layout()
alpha_path = os.path.join(OUTPUT_DIR, 'alpha_equiv_vs_p.png')
plt.savefig(alpha_path, dpi=150)
print(f"  Saved: {alpha_path}")

plt.show()

print("\n" + "=" * 70)
print("P-VALUE GATING STUDY COMPLETE")
print("=" * 70)
print(f"All outputs saved to: {OUTPUT_DIR}")
