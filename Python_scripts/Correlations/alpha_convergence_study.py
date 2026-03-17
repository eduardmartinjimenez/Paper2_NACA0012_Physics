import os
import h5py
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# ============================================================================
# Configuration
# ============================================================================

CORR_DIR = (
    "/home/jofre/Members/Eduard/Paper2/Simulations/"
    "NACA_0012_AOA12_Re50000_1716x1662x128/Mean_data/Wall_shear_correlations/test_1"
)
OUTPUT_DIR = (
    "/home/jofre/Members/Eduard/Paper2/Simulations/"
    "NACA_0012_AOA12_Re50000_1716x1662x128/Mean_data/Wall_shear_correlations/Figures/"
)
os.makedirs(OUTPUT_DIR, exist_ok=True)

X_C_LOCATIONS = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
ALPHA_VALUES  = [0.5, 1.0, 1.25, 1.5, 2.0]

# x/c locations shown in 2D map panels (subset for readability)
XC_MAP_SUBSET = [0.3, 0.5, 0.7]

# Minimum number of samples considered adequate for convergence
N_MIN_ADEQUATE = 500

# Fluid / simulation reference parameters
Re_c    = 50000
rho_ref = 1.0
u_infty = 1.0
c_ref   = 1.0
nu_ref  = u_infty * c_ref / Re_c   # kinematic viscosity = 1/Re_c

FILE_PATTERN = "wall_shear_correlation_xc_{xc:.3f}_alpha_{alpha:.1f}_all_fft.h5"

# ============================================================================
# Load all data from pre-computed HDF5 correlation files
# ============================================================================
print("=" * 70)
print("LOADING CORRELATION FILES")
print("=" * 70)

# data[alpha][xc] holds a dict of metrics or None if file is missing
data = {}

for alpha in ALPHA_VALUES:
    data[alpha] = {}
    for xc in X_C_LOCATIONS:
        fname = FILE_PATTERN.format(xc=xc, alpha=alpha)
        fpath = os.path.join(CORR_DIR, fname)

        if not os.path.exists(fpath):
            print(f"  [MISSING] {fname}")
            data[alpha][xc] = None
            continue

        with h5py.File(fpath, "r") as f:
            N_PF  = int(f.attrs["N_PF"])
            N_NF  = int(f.attrs["N_NF"])
            N_all = int(f.attrs["N_all"])
            tau_rms_PF = float(f.attrs["tau_rms_PF"])
            tau_rms_NF = float(f.attrs["tau_rms_NF"])
            tau_w_rms  = float(f.attrs["tau_w_rms"])
            tau_w_mean = float(f.attrs["tau_w_mean"])
            x_c_actual = float(f.attrs["x_c_actual"])
            y_actual   = float(f.attrs["y_actual"])

            # 2D slice at Dz=0 (z-index 0)
            R_PF_2d  = f["R_PF"][0, :, :].copy()   # (Ny, Nx)
            R_NF_2d  = f["R_NF"][0, :, :].copy()
            R_all_2d = f["R_all"][0, :, :].copy()
            x_2d     = f["x"][0, :, :].copy()
            y_2d     = f["y"][0, :, :].copy()

        peak_R_PF  = float(np.nanmax(np.abs(R_PF_2d)))
        peak_R_NF  = float(np.nanmax(np.abs(R_NF_2d)))
        peak_R_all = float(np.nanmax(np.abs(R_all_2d)))

        data[alpha][xc] = {
            "N_PF":        N_PF,
            "N_NF":        N_NF,
            "N_all":       N_all,
            "f_PF":        N_PF / N_all,
            "f_NF":        N_NF / N_all,
            "tau_rms_PF":  tau_rms_PF,
            "tau_rms_NF":  tau_rms_NF,
            "tau_w_rms":   tau_w_rms,
            "tau_w_mean":  tau_w_mean,
            "peak_R_PF":   peak_R_PF,
            "peak_R_NF":   peak_R_NF,
            "peak_R_all":  peak_R_all,
            "R_PF_2d":     R_PF_2d,
            "R_NF_2d":     R_NF_2d,
            "R_all_2d":    R_all_2d,
            "x_2d":        x_2d,
            "y_2d":        y_2d,
            "x_c_actual":  x_c_actual,
            "y_actual":    y_actual,
        }

        print(
            f"  xc={xc:.1f}  alpha={alpha:.1f}"
            f"  N_PF={N_PF:6d}  N_NF={N_NF:6d}"
            f"  f_PF={N_PF/N_all:.3f}  f_NF={N_NF/N_all:.3f}"
            f"  peak|R_PF|={peak_R_PF:.4f}  peak|R_NF|={peak_R_NF:.4f}"
        )

# ============================================================================
# Helper arrays for plotting
# ============================================================================

n_xc    = len(X_C_LOCATIONS)
n_alpha = len(ALPHA_VALUES)
xc_arr  = np.array(X_C_LOCATIONS)

# Colours: one per alpha value
alpha_colors = {0.5: "#762a83", 1.0: "#2166ac", 1.25: "#92c5de", 1.5: "#f4a582", 2.0: "#d6604d"}
alpha_labels = {a: f"α = {a:.2f}" for a in ALPHA_VALUES}

# Bar-group offset for grouped bar charts (5 groups, centred)
bar_width  = 0.17
bar_offset = {
    0.5:  -2.0 * bar_width,
    1.0:  -1.0 * bar_width,
    1.25:  0.0 * bar_width,
    1.5:   1.0 * bar_width,
    2.0:   2.0 * bar_width,
}

# ============================================================================
# Figure 1: Event fractions and sample counts
# ============================================================================
print("\nFigure 1: Event fractions and sample counts")

fig, axes = plt.subplots(2, 2, figsize=(13, 8), sharex=True)
fig.suptitle("Event fractions and sample counts vs chord location", fontsize=13)

ax_fPF, ax_fNF   = axes[0]
ax_NPF, ax_NNF   = axes[1]

xc_pos = np.arange(n_xc)   # integer positions for bar groups

for alpha in ALPHA_VALUES:
    off   = bar_offset[alpha]
    color = alpha_colors[alpha]
    label = alpha_labels[alpha]

    f_PF_vals  = [data[alpha][xc]["f_PF"]  * 100 if data[alpha][xc] else np.nan for xc in X_C_LOCATIONS]
    f_NF_vals  = [data[alpha][xc]["f_NF"]  * 100 if data[alpha][xc] else np.nan for xc in X_C_LOCATIONS]
    N_PF_vals  = [data[alpha][xc]["N_PF"]        if data[alpha][xc] else np.nan for xc in X_C_LOCATIONS]
    N_NF_vals  = [data[alpha][xc]["N_NF"]        if data[alpha][xc] else np.nan for xc in X_C_LOCATIONS]

    kw_bar = dict(width=bar_width, color=color, alpha=0.85, label=label, edgecolor="white", linewidth=0.5)
    ax_fPF.bar(xc_pos + off, f_PF_vals, **kw_bar)
    ax_fNF.bar(xc_pos + off, f_NF_vals, **kw_bar)
    ax_NPF.bar(xc_pos + off, N_PF_vals, **kw_bar)
    ax_NNF.bar(xc_pos + off, N_NF_vals, **kw_bar)

# Minimum adequate samples line
for ax in (ax_NPF, ax_NNF):
    ax.axhline(N_MIN_ADEQUATE, color="black", lw=1.2, ls="--",
               label=f"N_min = {N_MIN_ADEQUATE}")

for ax, title, ylabel in [
    (ax_fPF, r"PF event fraction  ($\tau'_w > \alpha\,\tau_{rms}$)",      "f_PF [%]"),
    (ax_fNF, r"NF event fraction  ($\tau'_w < -\alpha\,\tau_{rms}$)",     "f_NF [%]"),
    (ax_NPF, "PF sample count",                                            "N_PF"),
    (ax_NNF, "NF sample count",                                            "N_NF"),
]:
    ax.set_title(title, fontsize=10)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.grid(True, axis="y", alpha=0.35)
    ax.set_xticks(xc_pos)
    ax.set_xticklabels([f"{xc:.1f}" for xc in X_C_LOCATIONS])
    ax.legend(fontsize=8)

for ax in (ax_NPF, ax_NNF):
    ax.set_xlabel("x/c", fontsize=11)

plt.tight_layout()
# out = os.path.join(OUTPUT_DIR, "alpha_study_event_fractions.png")
# plt.savefig(out, dpi=150, bbox_inches="tight")
# plt.close()
# print(f"  Saved: {out}")


# ============================================================================
# Figure 2: 2D correlation maps at Dz=0 — R_PF across alpha for selected x/c
# ============================================================================
print("Figure 2: 2D R_PF maps (Dz=0) for R_PF and R_NF")

for field_key, field_label in [("R_PF_2d", "R_PF"), ("R_NF_2d", "R_NF")]:

    n_rows = len(XC_MAP_SUBSET)
    n_cols = len(ALPHA_VALUES)

    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(4.5 * n_cols, 3.8 * n_rows),
        sharex=False, sharey=False,
    )
    fig.suptitle(
        rf"$\Delta z = 0$ correlation map  ${field_label}$  —  sensitivity to $\alpha$",
        fontsize=13,
    )

    clim = 1   # symmetric colour scale

    for row, xc in enumerate(XC_MAP_SUBSET):
        for col, alpha in enumerate(ALPHA_VALUES):
            ax = axes[row, col]
            d  = data[alpha][xc]

            if d is None:
                ax.set_visible(False)
                continue

            R    = d[field_key]
            x2d  = d["x_2d"]
            y2d  = d["y_2d"]
            xref = d["x_c_actual"]
            yref = d["y_actual"]

            # Down-sample for speed (every 2nd point in each direction)
            sl = (slice(None, None, 2), slice(None, None, 2))
            pcm = ax.pcolormesh(
                x2d[sl], y2d[sl], R[sl],
                cmap="RdBu_r", vmin=-clim, vmax=clim,
                shading="auto", rasterized=True,
            )

            # Reference point
            ax.scatter(xref, yref, c="lime", s=60, marker="*",
                       edgecolors="black", linewidths=0.5, zorder=5)

            ax.set_xlim(x2d.min(), x2d.max())
            ax.set_ylim(y2d.min(), y2d.max())
            ax.set_aspect("equal", adjustable="box")
            ax.set_xlabel("x/c", fontsize=9)
            ax.set_ylabel("y/c", fontsize=9)
            ax.set_title(
                f"x/c = {xc:.1f}  |  α = {alpha:.1f}"
                f"\nN_{'PF' if 'PF' in field_key else 'NF'}"
                f" = {d['N_PF' if 'PF' in field_key else 'N_NF']}",
                fontsize=9,
            )

            plt.colorbar(pcm, ax=ax, fraction=0.03, pad=0.03,
                         label=field_label)

    plt.tight_layout()
    # out = os.path.join(OUTPUT_DIR, f"alpha_study_2d_map_{field_label}.png")
    # plt.savefig(out, dpi=130, bbox_inches="tight")
    # plt.close()
    # print(f"  Saved: {out}")


# ============================================================================
# Figure 3: Wall-normal profile of R_PF / R_NF at x = x_ref (spanwise Dz=0)
# ============================================================================
print("Figure 3: Wall-normal profiles at x_ref, Dz=0")

fig, axes = plt.subplots(
    2, n_xc, figsize=(2.8 * n_xc, 6),
    sharey=True, sharex=False,
)
fig.suptitle(
    r"Wall-normal profiles of $R_{PF}$ and $R_{NF}$ at $x = x_{ref}$, $\Delta z = 0$",
    fontsize=12,
)

for j, xc in enumerate(X_C_LOCATIONS):
    ax_PF = axes[0, j]
    ax_NF = axes[1, j]

    for alpha in ALPHA_VALUES:
        d = data[alpha][xc]
        if d is None:
            continue

        x2d  = d["x_2d"]
        y2d  = d["y_2d"]
        xref = d["x_c_actual"]
        yref = d["y_actual"]

        # Find column index closest to x_ref
        x_1d = x2d[0, :]                            # (Nx,)
        ix   = int(np.argmin(np.abs(x_1d - xref)))

        R_PF_prof = d["R_PF_2d"][:, ix]             # (Ny,)
        R_NF_prof = d["R_NF_2d"][:, ix]
        y_prof    = y2d[:, ix]                       # (Ny,)

        # Convert to wall units: y+ = (y - y_wall) * u_tau / nu
        u_tau  = np.sqrt(np.abs(d["tau_w_mean"]) / rho_ref)
        y_plus = (y_prof - yref) * u_tau / nu_ref

        # Keep only y+ > 0 (above the wall)
        mask = y_plus > 0
        kw = dict(color=alpha_colors[alpha], lw=1.5, label=alpha_labels[alpha])
        ax_PF.plot(y_plus[mask], R_PF_prof[mask], **kw)
        ax_NF.plot(y_plus[mask], R_NF_prof[mask], **kw)

    for ax, title in [(ax_PF, r"$R_{PF}$"), (ax_NF, r"$R_{NF}$")]:
        ax.axhline(0, color="gray", lw=0.8, ls="--")
        ax.set_title(f"x/c={xc:.1f}  {title}", fontsize=8)
        ax.set_xlabel(r"$y^+$", fontsize=8)
        ax.set_xscale("log")
        ax.set_ylim(-1, 1)
        ax.grid(True, alpha=0.3, which="both")

    if j == 0:
        axes[0, 0].set_ylabel("$R$", fontsize=9)
        axes[1, 0].set_ylabel("$R$", fontsize=9)
        axes[0, 0].legend(fontsize=7)

plt.tight_layout()
# out = os.path.join(OUTPUT_DIR, "alpha_study_wall_normal_profiles.png")
# plt.savefig(out, dpi=150, bbox_inches="tight")
# plt.close()
# print(f"  Saved: {out}")
plt.show()


# ============================================================================
# Summary table printed to terminal
# ============================================================================
print("\n" + "=" * 70)
print("SUMMARY TABLE")
print("=" * 70)
print(f"{'':>6s} {'alpha':>5s} {'N_PF':>8s} {'N_NF':>8s} "
      f"{'f_PF%':>7s} {'f_NF%':>7s} "
      f"{'pk|R_PF|':>9s} {'pk|R_NF|':>9s} {'convergent':>10s}")
print("-" * 75)

for xc in X_C_LOCATIONS:
    for alpha in ALPHA_VALUES:
        d = data[alpha][xc]
        if d is None:
            print(f"  x/c={xc:.1f}  alpha={alpha:.1f}  MISSING")
            continue
        conv = "YES" if (d["N_PF"] >= N_MIN_ADEQUATE and d["N_NF"] >= N_MIN_ADEQUATE) else "NO"
        print(
            f"  x/c={xc:.1f}  {alpha:4.1f}  {d['N_PF']:8d}  {d['N_NF']:8d}"
            f"  {d['f_PF']*100:6.2f}%  {d['f_NF']*100:6.2f}%"
            f"  {d['peak_R_PF']:8.4f}  {d['peak_R_NF']:8.4f}  {conv:>10s}"
        )
    print()

print(f"\nAll figures saved to: {OUTPUT_DIR}")
print("=" * 70)
print("ALPHA CONVERGENCE STUDY COMPLETE")
print("=" * 70)
