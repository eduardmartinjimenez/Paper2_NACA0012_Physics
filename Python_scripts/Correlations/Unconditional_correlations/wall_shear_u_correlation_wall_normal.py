"""
Wall-normal decay profiles of unconditional wall-shear/velocity correlation.

This script extracts wall-normal profiles from the in-plane Delta z = 0 maps of
new unconditional correlation datasets:

    R_tau_u = R(tau'_w, u')

For each selected reference chordwise location, the script:
1. Loads wall_shear_correlation_unconditional_xc_*.h5.
2. Extracts the Delta z = 0 plane.
3. Rotates the 2D mesh into an AoA-aligned frame.
4. Builds a KDTree on the rotated in-plane mesh.
5. Samples a vertical wall-normal line starting from the wall reference point.
6. Keeps only eta >= 0 and R > 0 for the decay plot.
7. Saves:
   - Plot 0: KDTree extraction-line diagnostic.
   - Plot 1: Wall-normal decay profiles R vs eta/c for all x/c locations.

Notes
-----
- The result z-index 0 is assumed to correspond to Delta z = 0, consistent with
  the current unconditional correlation computation based on circular FFT.
- If your file naming changes, edit build_result_file_candidates(). The function
  currently accepts both:
      wall_shear_correlation_unconditional_xc_0.500.h5
      wall_shear_correlation_unconditional_xc0.500.h5
"""

import os
import h5py
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree
from scipy.ndimage import gaussian_filter1d
from matplotlib.lines import Line2D

# Optional LaTeX styling. Disable if the cluster/head node has no LaTeX install.
USE_LATEX = True
if USE_LATEX:
    plt.rc("text", usetex=True)
    plt.rc("font", size=16, family="serif")
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
        "WALL_NORMAL_LENGTH": 0.08,
    },
    "AoA12": {
        "AOA_DEG": 12.0,
        "BASE_RESULTS_DIR": (
            "/home/jofre/Members/Eduard/Paper2/Simulations/"
            "NACA_0012_AOA12_Re50000_1716x1662x128/Mean_data/"
            "Wall_shear_correlations/test_5"
        ),
        "WALL_NORMAL_LENGTH": 0.25,
    },
}

# Common output directory for the comparison figures.
OUTPUT_DIR = (
    "/home/jofre/Members/Eduard/Paper2/Simulations/"
    "NACA_0012_AOA12_Re50000_1716x1662x128/Mean_data/"
    "Wall_shear_correlations/comparison_AOA5_AOA12/Figures"
)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Target chord locations. Edit as needed.
X_C_LOCATIONS = [0.5, 0.7, 0.9]

# Wall-normal extraction settings in chord units.
WALL_NORMAL_LENGTH = 0.25
N_SAMPLE_POINTS = 400

# Optional quality control. Set to None to disable.
# This prevents isolated KDTree jumps if the requested line exits the cropped mesh.
MAX_KDTREE_DISTANCE = None
# Example: MAX_KDTREE_DISTANCE = 2.0e-3

# Plot settings
ONLY_POSITIVE_CORRELATION = True
SAVE_EPS = True
SHOW_FIGURES = True
SMOOTH_SIGMA = 1

# =============================================================================
# Wall-unit scaling
# =============================================================================

Re_c = 50000.0
rho_ref = 1.0
u_infty = 1.0
c_ref = 1.0

nu_ref = u_infty * c_ref / Re_c

# Correlation value used to define a practical decay height.
# This is not a mathematical zero, it is a reporting threshold.
R_DECAY_THRESHOLD = 0.05

# =============================================================================
# Helpers
# =============================================================================

def rotate_to_flow_frame(x, y, angle_rad):
    """
    Rotate coordinates into a flow-aligned frame.

    x_prime is streamwise-aligned and y_prime is cross-stream/wall-normal-like.
    This follows the same convention used in the previous KDTree wall-normal
    extraction script:

        x' =  x cos(alpha) + y sin(alpha)
        y' = -x sin(alpha) + y cos(alpha)
    """
    ca = np.cos(angle_rad)
    sa = np.sin(angle_rad)
    x_prime = x * ca + y * sa
    y_prime = -x * sa + y * ca
    return x_prime, y_prime


def build_result_file_candidates(base_dir, x_c):
    """
    Return possible unconditional correlation filenames for one target x/c.

    The underscore version is the one currently written by the unconditional
    computation script. The no-underscore version is included because it was
    requested explicitly in the new plotting specification.
    """
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


def load_dz0_unconditional_map(filepath):
    """Load R_tau_u and mesh coordinates at Delta z = 0."""
    with h5py.File(filepath, "r") as f:
        if "R" not in f:
            raise KeyError(f"Dataset 'R' not found in {filepath}")

        R = f["R"]
        x = f["x"]
        y = f["y"]

        if R.ndim != 3:
            raise ValueError(f"Expected R with shape (Nz, Ny, Nx), got {R.shape}")

        R_2d = R[0, :, :]
        x_2d = x[0, :, :]
        y_2d = y[0, :, :]

        x_c_actual = float(f.attrs["x_c_actual"])
        y_actual = float(f.attrs["y_actual"])
        n_samples = int(f.attrs.get("N_samples", -1))
        n_snapshots = int(f.attrs.get("N_snapshots", -1))
        tau_w_rms = float(f.attrs.get("tau_w_rms", np.nan))
        tau_w_mean = float(f.attrs.get("tau_w_mean", np.nan))

    return {
        "R_2d": np.asarray(R_2d),
        "x_2d": np.asarray(x_2d),
        "y_2d": np.asarray(y_2d),
        "x_c_actual": x_c_actual,
        "y_actual": y_actual,
        "N_samples": n_samples,
        "N_snapshots": n_snapshots,
        "tau_w_rms": tau_w_rms,
        "tau_w_mean": tau_w_mean,
    }


def extract_wall_normal_profile(data, angle_rad, wall_normal_length):
    """Extract one KDTree wall-normal profile from a Delta z = 0 map."""
    R_2d = data["R_2d"]
    x_2d = data["x_2d"]
    y_2d = data["y_2d"]
    x_ref = data["x_c_actual"]
    y_ref = data["y_actual"]

    # Rotate full in-plane mesh and wall reference point.
    x_rot, y_rot = rotate_to_flow_frame(x_2d, y_2d, angle_rad)
    x_ref_rot, y_ref_rot = rotate_to_flow_frame(x_ref, y_ref, angle_rad)

    Ny, Nx = x_rot.shape
    x_flat = x_rot.ravel()
    y_flat = y_rot.ravel()

    valid_mesh = np.isfinite(x_flat) & np.isfinite(y_flat) & np.isfinite(R_2d.ravel())
    flat_valid_indices = np.flatnonzero(valid_mesh)

    tree = cKDTree(np.column_stack((x_flat[valid_mesh], y_flat[valid_mesh])))

    # Query a vertical line in the rotated frame.
    eta_query = np.linspace(0.0, wall_normal_length, N_SAMPLE_POINTS)
    x_query = np.full_like(eta_query, x_ref_rot)
    y_query = y_ref_rot + eta_query

    distances, local_indices = tree.query(np.column_stack((x_query, y_query)))
    flat_indices = flat_valid_indices[local_indices]

    j_indices = flat_indices // Nx
    i_indices = flat_indices % Nx

    # Remove duplicate grid cells while preserving the first occurrence along eta.
    ij_pairs = np.column_stack((j_indices, i_indices))
    _, unique_query_indices = np.unique(ij_pairs, axis=0, return_index=True)
    unique_query_indices = np.sort(unique_query_indices)

    j_u = j_indices[unique_query_indices]
    i_u = i_indices[unique_query_indices]
    dist_u = distances[unique_query_indices]

    x_profile = x_rot[j_u, i_u]
    y_profile = y_rot[j_u, i_u]
    eta = y_profile - y_ref_rot
    R_profile = R_2d[j_u, i_u]

    # Sort by actual extracted eta. KDTree duplicate removal can otherwise leave
    # tiny non-monotonic jumps on skewed or cropped meshes.
    order = np.argsort(eta)
    x_profile = x_profile[order]
    y_profile = y_profile[order]
    eta = eta[order]
    R_profile = R_profile[order]
    dist_u = dist_u[order]
    j_u = j_u[order]
    i_u = i_u[order]

    mask = np.isfinite(eta) & np.isfinite(R_profile) & (eta >= 0.0)

    if ONLY_POSITIVE_CORRELATION:
        mask &= R_profile > 0.0

    if MAX_KDTREE_DISTANCE is not None:
        mask &= dist_u <= MAX_KDTREE_DISTANCE

    tau_w_mean = float(data.get("tau_w_mean", np.nan))

    if np.isfinite(tau_w_mean):
        u_tau = np.sqrt(np.abs(tau_w_mean) / rho_ref)
    else:
        u_tau = np.nan

    if np.isfinite(u_tau) and u_tau > 0.0:
        y_plus = eta * u_tau / nu_ref
    else:
        y_plus = np.full_like(eta, np.nan)

    return {
        "x_prime": x_profile,
        "y_prime": y_profile,
        "eta": eta,
        "R_profile": R_profile,
        "eta_plot": eta[mask],
        "y_plus": y_plus,
        "y_plus_plot": y_plus[mask],
        "R_profile_plot": R_profile[mask],
        "u_tau": u_tau,
        "nu_ref": nu_ref,
        "distances": dist_u,
        "i_indices": i_u,
        "j_indices": j_u,
        "plot_mask": mask,
        "x_2d_rot": x_rot,
        "y_2d_rot": y_rot,
        "x_ref_rot": float(x_ref_rot),
        "y_ref_rot": float(y_ref_rot),
    }

# =============================================================================
# Main workflow
# =============================================================================

print("=" * 78)
print("UNCONDITIONAL WALL-SHEAR / STREAMWISE-VELOCITY WALL-NORMAL DECAY")
print("=" * 78)
print(f"Results directory: {[cfg['BASE_RESULTS_DIR'] for cfg in CASES.values()]}")
print(f"Output directory:  {OUTPUT_DIR}")
print(f"AoA:               {list(CASES.values())[0]['AOA_DEG']:.1f} deg")
print(f"Target x/c:         {X_C_LOCATIONS}")
print(f"Only R > 0:         {ONLY_POSITIVE_CORRELATION}")

profiles = {}

for case_name, case_cfg in CASES.items():
    aoa_deg = case_cfg["AOA_DEG"]
    aoa_rad = np.deg2rad(aoa_deg)
    base_results_dir = case_cfg["BASE_RESULTS_DIR"]
    wall_normal_length = case_cfg["WALL_NORMAL_LENGTH"]

    profiles[case_name] = {}

    print("\n" + "=" * 78)
    print(f"LOADING CASE: {case_name}")
    print("=" * 78)
    print(f"Results directory: {base_results_dir}")
    print(f"AoA:               {aoa_deg:.1f} deg")
    print(f"Target x/c:         {X_C_LOCATIONS}")

    for x_c_target in X_C_LOCATIONS:
        filepath = resolve_result_file(base_results_dir, x_c_target)

        if filepath is None:
            print(f"[WARNING] No file found for {case_name}, target x/c = {x_c_target:.3f}")
            for candidate in build_result_file_candidates(base_results_dir, x_c_target):
                print(f"          tried: {candidate}")
            continue

        print(f"\nLoading {case_name}, x/c target {x_c_target:.3f}: {os.path.basename(filepath)}")
        data = load_dz0_unconditional_map(filepath)
        profile = extract_wall_normal_profile(data, aoa_rad, wall_normal_length)

        profiles[case_name][x_c_target] = {
            **data,
            **profile,
            "filepath": filepath,
            "x_c_target": x_c_target,
            "AOA_DEG": aoa_deg,
            "WALL_NORMAL_LENGTH": wall_normal_length,
            "case_name": case_name,
        }

        print(
            f"  actual wall point: x/c={data['x_c_actual']:.5f}, y={data['y_actual']:.5f}"
        )
        print(
            f"  extracted points: {len(profile['eta'])} unique, "
            f"plotted points: {len(profile['eta_plot'])}, "
            f"max KDTree distance: {np.nanmax(profile['distances']):.4e}"
        )
        print(
            f"  R range on profile: [{np.nanmin(profile['R_profile']):.4f}, "
            f"{np.nanmax(profile['R_profile']):.4f}]"
        )

# Remove empty cases
profiles = {case_name: case_profiles for case_name, case_profiles in profiles.items() if case_profiles}

if not profiles:
    raise RuntimeError("No unconditional wall-shear correlation files were loaded.")

x_c_sorted = sorted(X_C_LOCATIONS)

# Colors identify chordwise location, consistent with PDF/statistics plots.
color_by_xc = {
    0.5: "red",
    0.7: "blue",
    0.9: "green",
}

# Fallback colors if additional x/c locations are added later.
fallback_colors = plt.cm.tab10(np.linspace(0.0, 1.0, len(x_c_sorted)))
for x_c, fallback_color in zip(x_c_sorted, fallback_colors):
    color_by_xc.setdefault(x_c, fallback_color)

# Line styles identify angle of attack.
# Consistent convention:
#   solid  -> AoA = 5 deg
#   dashed -> AoA = 12 deg
linestyle_by_case = {
    "AoA5": "-",
    "AoA12": "--",
}

# =============================================================================
# Plot 0: extraction-line verification, one panel per AoA
# =============================================================================

print("\nCreating Plot 0: extraction-line verification...")

n_cases = len(profiles)
fig0, axes0 = plt.subplots(
    1,
    n_cases,
    figsize=(6.0 * n_cases, 6.0),
    squeeze=False,
)

axes0 = axes0.ravel()

for ax0, (case_name, case_profiles) in zip(axes0, profiles.items()):
    aoa_deg = CASES[case_name]["AOA_DEG"]

    # Plot all cropped meshes for this AoA.
    for x_c in x_c_sorted:
        if x_c not in case_profiles:
            continue

        data = case_profiles[x_c]
        ax0.scatter(
            data["x_2d_rot"].ravel(),
            data["y_2d_rot"].ravel(),
            s=0.15,
            c="lightgray",
            alpha=0.18,
            rasterized=True,
            linewidths=0,
        )

    for x_c in x_c_sorted:
        if x_c not in case_profiles:
            continue

        data = case_profiles[x_c]
        color = color_by_xc[x_c]

        ax0.plot(
            [data["x_ref_rot"], data["x_ref_rot"]],
            [data["y_ref_rot"], data["y_ref_rot"] + data["WALL_NORMAL_LENGTH"]],
            linestyle="--",
            linewidth=1.5,
            color=color,
            alpha=0.85,
            label=rf"$x/c={x_c:.2f}$",
        )

        ax0.scatter(
            data["x_prime"],
            data["y_prime"],
            s=16,
            color=color,
            edgecolors="black",
            linewidth=0.25,
            alpha=0.85,
            zorder=3,
        )

        ax0.plot(
            data["x_ref_rot"],
            data["y_ref_rot"],
            marker="*",
            markersize=12,
            color=color,
            markeredgecolor="black",
            markeredgewidth=0.6,
            zorder=4,
        )

    ax0.set_xlabel(r"$x^\prime/c$")
    ax0.set_ylabel(r"$y^\prime/c$")
    ax0.set_aspect("equal", adjustable="box")
    ax0.legend(fontsize=11, loc="best", frameon=False)

fig0.suptitle(r"KDTree wall-normal extraction lines, $\Delta z=0$", y=1.02)
fig0.tight_layout()

plot0_png = os.path.join(
    OUTPUT_DIR,
    "R_tau_u_unconditional_wall_normal_kdtree_extraction_AOA5_AOA12.png",
)
fig0.savefig(plot0_png, dpi=300, bbox_inches="tight")
print(f"  Saved: {plot0_png}")

if SAVE_EPS:
    plot0_eps = plot0_png.replace(".png", ".eps")
    fig0.savefig(plot0_eps, dpi=300, bbox_inches="tight")
    print(f"  Saved: {plot0_eps}")

# =============================================================================
# Plot 1: wall-normal decay in outer units, AoA comparison
# =============================================================================

print("\nCreating Plot 1: wall-normal decay profiles in outer units...")
fig1, ax1 = plt.subplots(figsize=(7.2, 5.4))

for case_name, case_profiles in profiles.items():
    linestyle = linestyle_by_case.get(case_name, "-")
    aoa_deg = CASES[case_name]["AOA_DEG"]

    for x_c in x_c_sorted:
        if x_c not in case_profiles:
            continue

        data = case_profiles[x_c]
        eta = data["eta_plot"]
        R_profile = data["R_profile_plot"]

        if len(eta) == 0:
            print(f"  [WARNING] No positive-correlation points for {case_name}, x/c={x_c:.3f}")
            continue

        R_profile_smooth = gaussian_filter1d(R_profile, sigma=SMOOTH_SIGMA)

        ax1.plot(
            eta,
            R_profile_smooth,
            linewidth=2.2,
            linestyle=linestyle,
            color=color_by_xc[x_c],
            label=None,
        )

ax1.axhline(0.0, color="0.4", linewidth=0.8, alpha=0.6)
ax1.set_xlabel(r"$\eta/c$")
ax1.set_ylabel(r"$R_{\tau_w^\prime u^\prime}$")
ax1.set_xlim(left=0.0)
ax1.set_ylim(bottom=0.0)
color_handles = [
    Line2D([0], [0], color=color_by_xc[x_c], linewidth=2.5, linestyle="-",
           label=rf"$x/c={x_c:.1f}$")
    for x_c in x_c_sorted
]

ax1.legend(handles=color_handles, fontsize=11, loc="best", frameon=False)

fig1.tight_layout()

plot1_png = os.path.join(
    OUTPUT_DIR,
    "R_tau_u_unconditional_wall_normal_decay_eta_AOA5_AOA12.png",
)
fig1.savefig(plot1_png, dpi=300, bbox_inches="tight")
print(f"  Saved: {plot1_png}")

if SAVE_EPS:
    plot1_eps = plot1_png.replace(".png", ".eps")
    fig1.savefig(plot1_eps, dpi=300, bbox_inches="tight")
    print(f"  Saved: {plot1_eps}")

# =============================================================================
# Plot 2: wall-normal decay in local wall units, AoA comparison
# =============================================================================

print("\nCreating Plot 2: wall-normal decay profiles in wall units...")
fig2, ax2 = plt.subplots(figsize=(7.2, 5.4))

decay_summary = {}
YPLUS_PLOT_MIN = 1.0e-1

for case_name, case_profiles in profiles.items():
    linestyle = linestyle_by_case.get(case_name, "-")
    decay_summary[case_name] = {}

    for x_c in x_c_sorted:
        if x_c not in case_profiles:
            continue

        data = case_profiles[x_c]

        y_plus = data["y_plus_plot"]
        R_profile = data["R_profile_plot"]

        valid = np.isfinite(y_plus) & np.isfinite(R_profile) & (y_plus > 0.0)

        if not np.any(valid):
            print(f"  [WARNING] No valid wall-unit points for {case_name}, x/c={x_c:.3f}")
            continue

        y_plus_valid = y_plus[valid]
        R_valid = R_profile[valid]
        eta_valid = data["eta_plot"][valid]

        # Sort by y+ to avoid possible KDTree ordering artifacts.
        sort_idx = np.argsort(y_plus_valid)
        y_plus_valid = y_plus_valid[sort_idx]
        R_valid = R_valid[sort_idx]
        eta_valid = eta_valid[sort_idx]

        # Optional visual near-wall anchor.
        # This is not a new DNS point.
        if y_plus_valid[0] > YPLUS_PLOT_MIN:
            y_plus_plot = np.insert(y_plus_valid, 0, YPLUS_PLOT_MIN)
            R_plot = np.insert(R_valid, 0, R_valid[0])
        else:
            y_plus_plot = y_plus_valid
            R_plot = R_valid

        R_plot_smooth = gaussian_filter1d(R_plot, sigma=SMOOTH_SIGMA)

        ax2.plot(
            y_plus_plot,
            R_plot_smooth,
            linewidth=2.2,
            linestyle=linestyle,
            color=color_by_xc[x_c],
            label=None,
        )

        # Practical decay height: first y+ where R falls below threshold.
        below = np.where(R_valid <= R_DECAY_THRESHOLD)[0]

        if len(below) > 0:
            idx_decay = below[0]
            y_plus_decay = y_plus_valid[idx_decay]
            eta_decay = eta_valid[idx_decay]
            R_decay = R_valid[idx_decay]
        else:
            y_plus_decay = np.nan
            eta_decay = np.nan
            R_decay = np.nan

        decay_summary[case_name][x_c] = {
            "u_tau": data["u_tau"],
            "y_plus_decay": y_plus_decay,
            "eta_decay": eta_decay,
            "R_decay": R_decay,
        }


ax2.axhline(0.0, color="0.4", linewidth=0.8, alpha=0.6)
ax2.set_xlabel(r"$y^+$")
ax2.set_ylabel(r"$R_{\tau_w^\prime u^\prime}$")
ax2.set_xscale("log")
ax2.set_xlim(left=YPLUS_PLOT_MIN)
ax2.set_ylim(bottom=0.0)
color_handles = [
    Line2D([0], [0], color=color_by_xc[x_c], linewidth=2.5, linestyle="-",
           label=rf"$x/c={x_c:.1f}$")
    for x_c in x_c_sorted
]

ax2.legend(handles=color_handles, fontsize=11, loc="best", frameon=False)

fig2.tight_layout()

plot2_png = os.path.join(
    OUTPUT_DIR,
    "R_tau_u_unconditional_wall_normal_decay_yplus_AOA5_AOA12.png",
)
fig2.savefig(plot2_png, dpi=300, bbox_inches="tight")
print(f"  Saved: {plot2_png}")

if SAVE_EPS:
    plot2_eps = plot2_png.replace(".png", ".eps")
    fig2.savefig(plot2_eps, dpi=300, bbox_inches="tight")
    print(f"  Saved: {plot2_eps}")

# =============================================================================
# Summary
# =============================================================================

print("\n" + "=" * 78)
print("SUMMARY")
print("=" * 78)

for case_name, case_profiles in profiles.items():
    print(f"\nLoaded profiles for {case_name}:")

    for x_c in x_c_sorted:
        if x_c not in case_profiles:
            continue

        data = case_profiles[x_c]

        print(
            f"  target x/c={x_c:.3f}, actual x/c={data['x_c_actual']:.5f}, "
            f"N_samples={data['N_samples']}, N_snapshots={data['N_snapshots']}, "
            f"tau_w_mean={data['tau_w_mean']:.6e}, "
            f"u_tau={data['u_tau']:.6e}, "
            f"points plotted={len(data['eta_plot'])}"
        )

print("\nPractical decay heights:")
print(f"  threshold: R <= {R_DECAY_THRESHOLD:.3f}")

for case_name, case_decay in decay_summary.items():
    print(f"\n  {case_name}:")

    for x_c in x_c_sorted:
        if x_c not in case_decay:
            continue

        item = case_decay[x_c]

        if np.isfinite(item["y_plus_decay"]):
            print(
                f"    x/c={x_c:.3f}: "
                f"eta/c={item['eta_decay']:.6e}, "
                f"y+={item['y_plus_decay']:.3f}, "
                f"R={item['R_decay']:.4f}"
            )
        else:
            print(
                f"    x/c={x_c:.3f}: "
                f"R did not fall below {R_DECAY_THRESHOLD:.3f} within the extracted line"
            )

print("\nOutput files:")
print(f"  {plot0_png}")
print(f"  {plot1_png}")
print(f"  {plot2_png}")

if SHOW_FIGURES:
    plt.show()
else:
    plt.close("all")