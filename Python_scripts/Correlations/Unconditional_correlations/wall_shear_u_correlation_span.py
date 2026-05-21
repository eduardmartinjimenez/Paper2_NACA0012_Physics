"""
Spanwise correlation decay profiles of unconditional wall-shear/velocity correlation.

This script extracts and compares spanwise correlation profiles R(Δz) from the
new unconditional correlation datasets:

    R_tau_u = R(tau'_w, u')

For each selected reference chordwise location and both AoAs, the script:
1. Loads wall_shear_correlation_unconditional_xc_*.h5 (full 3D field).
2. Extracts the spanwise correlation curve at the surface point.
3. Recenters the lag coordinate so Δz=0 is in the middle.
4. Creates comparison figures showing:
   - Spanwise decay profiles at different x/c locations (one panel per AoA)
   - AoA comparison at each x/c location

The z-index dimension (Nz=128) represents spanwise separation Δz, where z-index=0
corresponds to the reference z-plane and z-index=k corresponds to spanwise separation
Δz = z[k,0,0] - z[0,0,0].

Notes
-----
If file naming changes, edit build_result_file_candidates(). Currently accepts:
    wall_shear_correlation_unconditional_xc_0.500.h5
    wall_shear_correlation_unconditional_xc0.500.h5
"""

import os
import h5py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# Optional LaTeX styling
USE_LATEX = True
if USE_LATEX:
    plt.rc("text", usetex=True)
    plt.rc("font", size=14, family="serif")
    plt.rc("text.latex", preamble=r"\usepackage{amsmath} \usepackage{amssymb}")

# =============================================================================
# Configuration
# =============================================================================

CASES = {
    "AoA5": {
        "AOA_DEG": 5.0,
        "BASE_RESULTS_DIR": (
            "/home/jofre/Members/Eduard/Paper2/Simulations/"
            "NACA_0012_AOA5_Re50000_1716x1662x128/Mean_data/"
            "Wall_shear_correlations/test_3"
        ),
    },
    "AoA12": {
        "AOA_DEG": 12.0,
        "BASE_RESULTS_DIR": (
            "/home/jofre/Members/Eduard/Paper2/Simulations/"
            "NACA_0012_AOA12_Re50000_1716x1662x128/Mean_data/"
            "Wall_shear_correlations/test_5"
        ),
    },
}

# Common output directory for comparison figures
OUTPUT_DIR = (
    "/home/jofre/Members/Eduard/Paper2/Simulations/"
    "NACA_0012_AOA12_Re50000_1716x1662x128/Mean_data/"
    "Wall_shear_correlations/comparison_AOA5_AOA12/Figures"
)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Target chord locations
X_C_LOCATIONS = [0.5, 0.7, 0.9]

# Plot settings
SAVE_EPS = True
SHOW_FIGURES = True

# =============================================================================
# Helpers
# =============================================================================


def build_result_file_candidates(base_dir, x_c):
    """Return possible unconditional correlation filenames for one target x/c."""
    return [
        os.path.join(base_dir, f"wall_shear_correlation_unconditional_xc_{x_c:.3f}.h5"),
        os.path.join(base_dir, f"wall_shear_correlation_unconditional_xc{x_c:.3f}.h5"),
    ]


def resolve_result_file(base_dir, x_c):
    """Find the result file for one x/c target."""
    candidates = build_result_file_candidates(base_dir, x_c)
    for path in candidates:
        if os.path.exists(path):
            return path
    return None


def load_full_correlation(filepath):
    """Load the full 3D correlation field and coordinates."""
    with h5py.File(filepath, "r") as f:
        if "R" not in f:
            raise KeyError(f"Dataset 'R' not found in {filepath}")

        R = f["R"][:]  # (Nz, Ny, Nx)
        x = f["x"][:]  # (Nz, Ny, Nx)
        y = f["y"][:]  # (Nz, Ny, Nx)
        z = f["z"][:]  # (Nz, Ny, Nx)

        x_c_actual = float(f.attrs["x_c_actual"])
        y_actual = float(f.attrs["y_actual"])
        n_samples = int(f.attrs.get("N_samples", -1))
        n_snapshots = int(f.attrs.get("N_snapshots", -1))
        tau_w_rms = float(f.attrs.get("tau_w_rms", np.nan))
        tau_w_mean = float(f.attrs.get("tau_w_mean", np.nan))

    return {
        "R": np.asarray(R),
        "x": np.asarray(x),
        "y": np.asarray(y),
        "z": np.asarray(z),
        "x_c_actual": x_c_actual,
        "y_actual": y_actual,
        "N_samples": n_samples,
        "N_snapshots": n_snapshots,
        "tau_w_rms": tau_w_rms,
        "tau_w_mean": tau_w_mean,
    }


def extract_spanwise_profile(data):
    """Extract spanwise correlation profile at the surface point (y closest to wall)."""
    R = data["R"]  # (Nz, Ny, Nx)
    x = data["x"]  # (Nz, Ny, Nx)
    y = data["y"]  # (Nz, Ny, Nx)
    z = data["z"]  # (Nz, Ny, Nx)

    x_c_actual = data["x_c_actual"]
    y_actual = data["y_actual"]

    Nz, Ny, Nx = R.shape

    # Find x-index closest to x_c_actual (use first row)
    x_at_first_row = x[0, 0, :]  # (Nx,)
    ix_ref = np.argmin(np.abs(x_at_first_row - x_c_actual))

    # Find y-index closest to y_actual (use first z and reference x)
    y_at_ref_location = y[0, :, ix_ref]  # (Ny,)
    iy_ref = np.argmin(np.abs(y_at_ref_location - y_actual))

    # Extract spanwise correlation curve at this (iy_ref, ix_ref) point
    corr_at_point = R[:, iy_ref, ix_ref]  # (Nz,)

    # Compute centered spanwise lag coordinate
    z_ref = z[0, 0, 0]  # Reference z-coordinate
    dz_all = np.array([z[iz, 0, 0] - z_ref for iz in range(Nz)])

    # Recenter lag coordinate for periodic correlation
    dz_spacing = dz_all[1] - dz_all[0] if Nz > 1 else 1.0
    centered_indices = np.arange(-Nz // 2, Nz // 2)
    dz_centered = centered_indices * dz_spacing
    shift = Nz // 2

    # Recenter correlation curve
    corr_centered = np.roll(corr_at_point, shift)

    # Normalize by chord
    c_ref = 1.0
    dz_centered_normalized = dz_centered / c_ref

    return {
        "dz_centered": dz_centered_normalized,
        "corr_centered": corr_centered,
        "ix_ref": ix_ref,
        "iy_ref": iy_ref,
        "y_actual_found": y_at_ref_location[iy_ref],
        "x_actual_found": x_at_first_row[ix_ref],
        "dz_spacing": dz_spacing,
        "Nz": Nz,
    }


# =============================================================================
# Main workflow
# =============================================================================

print("=" * 78)
print("UNCONDITIONAL WALL-SHEAR / STREAMWISE-VELOCITY SPANWISE CORRELATION DECAY")
print("=" * 78)
print(f"Output directory:  {OUTPUT_DIR}")
print(f"Target x/c:        {X_C_LOCATIONS}")

profiles = {}

for case_name, case_cfg in CASES.items():
    aoa_deg = case_cfg["AOA_DEG"]
    base_results_dir = case_cfg["BASE_RESULTS_DIR"]

    profiles[case_name] = {}

    print("\n" + "=" * 78)
    print(f"LOADING CASE: {case_name} (AoA = {aoa_deg:.1f} deg)")
    print("=" * 78)
    print(f"Results directory: {base_results_dir}")

    for x_c_target in X_C_LOCATIONS:
        filepath = resolve_result_file(base_results_dir, x_c_target)

        if filepath is None:
            print(f"[WARNING] No file found for {case_name}, target x/c = {x_c_target:.3f}")
            for candidate in build_result_file_candidates(base_results_dir, x_c_target):
                print(f"          tried: {candidate}")
            continue

        print(f"\nLoading {case_name}, x/c target {x_c_target:.3f}: {os.path.basename(filepath)}")
        data = load_full_correlation(filepath)
        profile = extract_spanwise_profile(data)

        profiles[case_name][x_c_target] = {
            **data,
            **profile,
            "filepath": filepath,
            "x_c_target": x_c_target,
            "AOA_DEG": aoa_deg,
            "case_name": case_name,
        }

        print(
            f"  wall point: x/c={data['x_c_actual']:.5f}, y={data['y_actual']:.5f}"
        )
        print(
            f"  extracted point: x/c={profile['x_actual_found']:.5f}, "
            f"y={profile['y_actual_found']:.5f}"
        )
        print(
            f"  correlation range: [{np.nanmin(profile['corr_centered']):.4f}, "
            f"{np.nanmax(profile['corr_centered']):.4f}]"
        )

# Remove empty cases
profiles = {case_name: case_profiles for case_name, case_profiles in profiles.items() if case_profiles}

if not profiles:
    raise RuntimeError("No unconditional wall-shear correlation files were loaded.")

x_c_sorted = sorted(X_C_LOCATIONS)

# Color scheme: colors by chordwise location
color_by_xc = {
    0.5: "red",
    0.7: "blue",
    0.9: "green",
}

# Fallback colors
fallback_colors = plt.cm.tab10(np.linspace(0.0, 1.0, len(x_c_sorted)))
for x_c, fallback_color in zip(x_c_sorted, fallback_colors):
    color_by_xc.setdefault(x_c, fallback_color)

# Line styles: identify angle of attack
linestyle_by_case = {
    "AoA5": "-",
    "AoA12": "--",
}

# =============================================================================
# Plot: Spanwise correlation decay with AoA & x/c comparison
# =============================================================================

print("\nCreating spanwise correlation decay plot...")

fig, ax = plt.subplots(figsize=(7.2, 5.4))

for case_name, case_profiles in profiles.items():
    linestyle = linestyle_by_case.get(case_name, "-")

    for x_c in x_c_sorted:
        if x_c not in case_profiles:
            continue

        data = case_profiles[x_c]
        dz = data["dz_centered"]
        corr = data["corr_centered"]

        ax.plot(
            dz,
            corr,
            linewidth=2.2,
            linestyle=linestyle,
            color=color_by_xc[x_c],
            label=None,
        )

ax.set_xlabel(r"$\Delta z / c$", fontsize=12)
ax.set_ylabel(r"$R_{\tau_w^\prime u^\prime}$", fontsize=12)

# Create legend with color (x/c) and linestyle (AoA)
color_handles = [
    Line2D([0], [0], color=color_by_xc[x_c], linewidth=2.5, linestyle="-",
           label=rf"$x/c={x_c:.1f}$")
    for x_c in x_c_sorted
]


ax.legend(handles=color_handles, fontsize=11, loc="best", frameon=False, ncol=2)

fig.tight_layout()

plot_png = os.path.join(
    OUTPUT_DIR,
    "R_tau_u_unconditional_spanwise_decay_AOA5_AOA12.png",
)
fig.savefig(plot_png, dpi=300, bbox_inches="tight")
print(f"  Saved: {plot_png}")

if SAVE_EPS:
    plot_eps = plot_png.replace(".png", ".eps")
    fig.savefig(plot_eps, dpi=300, bbox_inches="tight")
    print(f"  Saved: {plot_eps}")

# =============================================================================
# Summary
# =============================================================================

print("\n" + "=" * 78)
print("SUMMARY")
print("=" * 78)

for case_name, case_profiles in profiles.items():
    aoa_deg = CASES[case_name]["AOA_DEG"]
    print(f"\nLoaded profiles for {case_name} (AoA = {aoa_deg:.1f}°):")

    for x_c in x_c_sorted:
        if x_c not in case_profiles:
            continue

        data = case_profiles[x_c]

        print(
            f"  x/c={x_c:.3f}: "
            f"wall point (x={data['x_c_actual']:.5f}, y={data['y_actual']:.5f}), "
            f"extracted (x={data['x_actual_found']:.5f}, y={data['y_actual_found']:.5f}), "
            f"N_samples={data['N_samples']}, "
            f"tau_w_mean={data['tau_w_mean']:.6e}"
        )

print("\nOutput files:")
print(f"  {plot_png}")

if SHOW_FIGURES:
    plt.show()
else:
    plt.close("all")
