import os
import numpy as np
import h5py
import matplotlib.pyplot as plt
from matplotlib import ticker
from matplotlib.ticker import FuncFormatter
from matplotlib import patheffects
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from scipy.interpolate import interp1d

# Shared plotting defaults
plt.rc("text", usetex=True)
plt.rc("font", size=14, family="serif")
plt.rc("text.latex", preamble=r"\usepackage{amsmath} \usepackage{amssymb}")

# ---------------------------------------------------------------------------
# Minimal loaders (reuse HDF5 structure from existing scripts)
# ---------------------------------------------------------------------------

def load_velocity_profile_data_dense(filepath):
    if not os.path.exists(filepath):
        print(f"Missing dense file: {filepath}")
        return None

    with h5py.File(filepath, "r") as f:
        u_infty = f.attrs["u_infty"]
        alpha = f.attrs["alpha"]
        C = f.attrs["C"]
        x_c_locations_dense = f["x_c_locations_dense"][...]

        x_rot = f["x_rot"][...]
        y_rot = f["y_rot"][...]
        u_rot = f["u_rot"][...]
        v_rot = f["v_rot"][...]

        interface_indices_i = f["interface_indices_i"][...]
        interface_indices_j = f["interface_indices_j"][...]
        x_int_rot = f["x_int_rot"][...]
        y_int_rot = f["y_int_rot"][...]
        suction_mask = f["suction_mask"][...]
        pressure_mask = f["pressure_mask"][...]

        wall_normal_profiles_dense = []
        profiles_group = f["wall_normal_profiles_dense"]
        for idx in range(len(profiles_group)):
            prof_group = profiles_group[f"profile_{idx}"]
            profile = {
                "x_c": prof_group.attrs["x_c"],
                "x_start": prof_group.attrs["x_start"],
                "y_start": prof_group.attrs["y_start"],
                "y_end": prof_group.attrs["y_end"],
                "x_prime": prof_group["x_prime"][...],
                "y_prime": prof_group["y_prime"][...],
                "u_rot": prof_group["u_rot"][...],
                "v_rot": prof_group["v_rot"][...],
            }
            wall_normal_profiles_dense.append(profile)

        isoline_data = {
            "x_c": f["isoline_data"]["x_c"][...],
            "y_c": f["isoline_data"]["y_c"][...],
        }

    return {
        "x_rot": x_rot,
        "y_rot": y_rot,
        "u_rot": u_rot,
        "v_rot": v_rot,
        "interface_indices_i": interface_indices_i,
        "interface_indices_j": interface_indices_j,
        "x_int_rot": x_int_rot,
        "y_int_rot": y_int_rot,
        "suction_mask": suction_mask,
        "pressure_mask": pressure_mask,
        "wall_normal_profiles_dense": wall_normal_profiles_dense,
        "isoline_data": isoline_data,
        "u_infty": u_infty,
        "alpha": alpha,
        "C": C,
        "x_c_locations_dense": x_c_locations_dense,
    }


def load_velocity_rms_profiles(filepath):
    if not os.path.exists(filepath):
        print(f"RMS file missing: {filepath}")
        return None

    with h5py.File(filepath, "r") as f:
        rms_profiles = []
        profiles_group = f["rms_profiles"]
        for idx in range(len(profiles_group)):
            prof_group = profiles_group[f"profile_{idx}"]
            profile = {
                "x_c": prof_group.attrs["x_c"],
                "x_start": prof_group.attrs["x_start"],
                "y_start": prof_group.attrs["y_start"],
                "y_end": prof_group.attrs["y_end"],
                "x_prime": prof_group["x_prime"][...],
                "y_prime": prof_group["y_prime"][...],
                "u_rms": prof_group["u_rms"][...],
                "v_rms": prof_group["v_rms"][...],
                "w_rms": prof_group["w_rms"][...],
            }
            rms_profiles.append(profile)

        snapshot_count = f.attrs.get("snapshot_count", None)

    return {
        "rms_profiles": rms_profiles,
        "snapshot_count": snapshot_count,
    }


def build_rms_by_xc(rms_profiles):
    if rms_profiles is None:
        return {}
    return {round(p["x_c"], 6): p for p in rms_profiles}


def build_2d_grid_from_profiles(profiles, rms_by_xc, u_infty, n_interp_y=150):
    x_list, y_list, u_rms_list = [], [], []
    for profile in profiles:
        x_c = round(profile["x_c"], 6)
        rms_profile = rms_by_xc.get(x_c)
        if rms_profile is None:
            continue
        y_vals = profile["y_prime"]
        x_val = profile["x_start"]
        u_rms_vals = rms_profile["u_rms"] / u_infty
        if len(y_vals) > 1 and len(u_rms_vals) == len(y_vals):
            x_list.append(x_val)
            y_list.append(y_vals)
            u_rms_list.append(u_rms_vals)

    if len(x_list) == 0:
        print("Warning: No RMS data available for contour plot")
        return None, None, None

    x_min, x_max = min(x_list), max(x_list)
    y_min, y_max = min(np.concatenate(y_list)), max(np.concatenate(y_list))
    x_grid = np.linspace(x_min, x_max, len(x_list))
    y_grid = np.linspace(y_min, y_max, n_interp_y)
    X_grid, Y_grid = np.meshgrid(x_grid, y_grid)

    U_rms_grid = np.zeros_like(X_grid)
    for i, (x_val, y_vals, u_rms_vals) in enumerate(zip(x_list, y_list, u_rms_list)):
        sort_idx = np.argsort(y_vals)
        y_sorted = y_vals[sort_idx]
        u_rms_sorted = u_rms_vals[sort_idx]
        f = interp1d(y_sorted, u_rms_sorted, kind="linear", fill_value="extrapolate")
        U_rms_grid[:, i] = f(y_grid)

    return X_grid, Y_grid, U_rms_grid


# ---------------------------------------------------------------------------
# Plot helpers
# ---------------------------------------------------------------------------

def recompute_zero_velocity_isoline(profiles, eps_zero=1e-6):
    """Recompute u'=0 isoline from wall-normal profiles using robust sign-change.
    Returns arrays of x/c and y/c, capped to x/c <= 1.0 and filtered for NaNs.
    """
    x_vals = []
    y_vals = []
    for profile in profiles:
        y = profile["y_prime"]
        u = profile["u_rot"]
        # Sort by wall-normal distance
        sort_idx = np.argsort(y)
        y_sorted = y[sort_idx]
        u_sorted = u[sort_idx]

        # Sign array with tolerance
        signs = np.zeros_like(u_sorted, dtype=np.int8)
        signs[u_sorted > eps_zero] = 1
        signs[u_sorted < -eps_zero] = -1

        # Crossing from negative to positive (separation bubble signature)
        crossings = np.where((signs[:-1] == -1) & (signs[1:] == 1))[0]
        if crossings.size > 0:
            k = crossings[0]
            # Linear interpolation for zero crossing between k and k+1
            u1, u2 = u_sorted[k], u_sorted[k+1]
            y1, y2 = y_sorted[k], y_sorted[k+1]
            if u2 != u1:
                y0 = y1 + (0.0 - u1) * (y2 - y1) / (u2 - u1)
            else:
                y0 = y1
            x_vals.append(profile["x_c"])  # x/c location of the profile
            y_vals.append(y0)

    if len(x_vals) == 0:
        return np.array([]), np.array([])

    x_arr = np.array(x_vals)
    y_arr = np.array(y_vals)
    # Cap to chord length and remove NaNs
    mask = (x_arr <= 1.0) & np.isfinite(y_arr)
    return x_arr[mask], y_arr[mask]

def add_colorbar(ax, mappable, label, tick_size=12, label_size=14, shrink=1.0):
    # Use same API as individual scripts for consistency
    cbar = plt.colorbar(mappable, ax=ax, fraction=0.046, pad=0.04, shrink=shrink, aspect=20)
    cbar.ax.set_title(label, fontsize=label_size, pad=8)
    cbar.ax.tick_params(labelsize=tick_size)
    cbar.ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, p: f"{x:.3f}"))
    return cbar

def load_reference_separation_reattachment(dat_file):
    if not dat_file or not os.path.exists(dat_file):
        return None
    xs, ys = [], []
    with open(dat_file, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.replace(",", " ").split()
            if len(parts) < 2:
                continue
            try:
                x = float(parts[0])
                y = float(parts[1])
                xs.append(x)
                ys.append(y)
            except ValueError:
                continue
    if len(xs) == 0:
        return None
    return {"x": np.array(xs), "y": np.array(ys)}


def load_reference_data_file(dat_file):
    """Load data from a single .dat file."""
    x_vals = []
    y_vals = []
    with open(dat_file, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 2:
                continue
            # European decimal commas replaced by dots
            x_val = float(parts[0].replace(",", "."))
            y_val = float(parts[1].replace(",", "."))
            x_vals.append(x_val)
            y_vals.append(y_val)

    if x_vals:
        return {
            "x": np.array(x_vals),
            "y": np.array(y_vals),
            "filename": dat_file,
        }
    else:
        return None


def load_reference_profiles_by_pattern(ref_data_path, pattern, x_c_token_position=4):
    """Generic loader for reference profile files grouped by x/c location.
    
    Exactly matches the approach from Mean_velocity_profiles_dense.py
    """
    import glob
    ref_profiles = {}

    dat_files = sorted(glob.glob(os.path.join(ref_data_path, pattern)))
    for dat_file in dat_files:
        base = os.path.basename(dat_file)
        # Extract the 0XX token (e.g., 015 -> 0.15)
        try:
            x_c_token = base.split("_")[x_c_token_position]
            x_c = float(x_c_token) / 100
        except Exception:
            # Skip files that don't follow the naming convention
            continue

        data = load_reference_data_file(dat_file)
        if data is not None:
            ref_profiles[x_c] = data

    return ref_profiles


def plot_case(ax, data_path, rms_path, x_c_locations_original, xlim, ylim,
              recompute_isoline=False, mark_reattachment=True,
              ref_sep_reatt_path=None,
              ref_profiles_dir=None,
              colorbar_shrink=1.0):
    data = load_velocity_profile_data_dense(data_path)
    if data is None:
        return None

    rms_data = load_velocity_rms_profiles(rms_path) if rms_path else None
    rms_by_xc = build_rms_by_xc(rms_data["rms_profiles"]) if rms_data else {}

    X_grid, Y_grid, U_rms_grid = build_2d_grid_from_profiles(
        data["wall_normal_profiles_dense"], rms_by_xc, data["u_infty"], n_interp_y=150
    )
    if X_grid is None:
        return None

    levels = np.linspace(0, np.nanmax(U_rms_grid) * 1.0, 8)[1:]
    contourf = ax.contourf(X_grid, Y_grid, U_rms_grid, levels=levels, cmap="YlOrRd", alpha=0.85)

    # Airfoil surface
    x_suction_rot = data["x_int_rot"][data["suction_mask"]]
    y_suction_rot = data["y_int_rot"][data["suction_mask"]]
    x_pressure_rot = data["x_int_rot"][data["pressure_mask"]]
    y_pressure_rot = data["y_int_rot"][data["pressure_mask"]]
    ax.plot(np.sort(x_suction_rot), y_suction_rot[np.argsort(x_suction_rot)], "-", color="black", linewidth=0.5, alpha=0.9)
    ax.plot(np.sort(x_pressure_rot), y_pressure_rot[np.argsort(x_pressure_rot)], "-", color="black", linewidth=0.5, alpha=0.9)

    # Zero-velocity isoline
    if recompute_isoline:
        x_iso, y_iso = recompute_zero_velocity_isoline(data["wall_normal_profiles_dense"])
        if len(x_iso) > 0:
            ax.plot(x_iso, y_iso, "b-.", linewidth=1.2, label=r"$u' = 0$")
    else:
        iso = data["isoline_data"]
        if len(iso["x_c"]) > 0:
            ax.plot(iso["x_c"], iso["y_c"], "b-.", linewidth=1.2, label=r"$u' = 0$")

    # Reference separation-reattachment line (AoA5)
    if ref_sep_reatt_path:
        ref_sep = load_reference_separation_reattachment(ref_sep_reatt_path)
        if ref_sep is not None:
            ax.plot(ref_sep["x"] - 0.003, ref_sep["y"] - 0.021,
                    'g--', linewidth=1.8, alpha=0.8,
                    label='Reference $u\' = 0$')

    # Overlay velocity profiles at requested locations
    scale_factor = 0.05
    for target in x_c_locations_original:
        diffs = np.abs(np.array([p["x_c"] for p in data["wall_normal_profiles_dense"]]) - target)
        idx = int(np.argmin(diffs))
        profile = data["wall_normal_profiles_dense"][idx]
        eta = profile["y_prime"] - profile["y_start"]
        sort_idx = np.argsort(eta)
        eta_sorted = eta[sort_idx]
        u_norm_sorted = (profile["u_rot"] / data["u_infty"])[sort_idx]
        x_profile = profile["x_start"] + u_norm_sorted * scale_factor
        y_profile = profile["y_start"] + eta_sorted
        ax.plot(x_profile, y_profile, "-", color="black", linewidth=1.2, alpha=0.9)
        ax.plot([profile["x_start"], profile["x_start"]], [profile["y_start"], profile["y_start"] + eta.max()], "--", color="gray", alpha=0.6, linewidth=0.8)

    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_aspect("equal", adjustable="box")
    ax.tick_params(labelsize=12)
    ax.set_xlabel(r"x/c", fontsize=16)
    ax.set_ylabel(r"y/c", fontsize=16)
    
    # Format tick labels to show 0 without decimals and 1 decimal for other values
    def format_ticks(x, pos):
        if x == 0:
            return '0'
        return f'{x:.1f}'
    
    ax.xaxis.set_major_formatter(FuncFormatter(format_ticks))
    ax.yaxis.set_major_formatter(FuncFormatter(format_ticks))

    # Reference mean velocity profiles (AoA5 style) — small black points
    if ref_profiles_dir:
        ref_profiles = load_reference_profiles_by_pattern(
            ref_profiles_dir,
            pattern="Re5e4_AOA5_U_mean_*_Jardin_2025_2.dat",
            x_c_token_position=4,
        )
        # Draw reference exactly as in working script
        ref_label_added = False
        if ref_profiles:
            for target in x_c_locations_original:
                # Check if this exact x/c location has reference data
                if target in ref_profiles:
                    ref = ref_profiles[target]
                    ax.plot(
                        ref["x"],
                        ref["y"],
                        'o',
                        color='black',
                        markersize=3,
                        alpha=1,
                        label=("Reference U_mean" if not ref_label_added else None)
                    )
                    ref_label_added = True

    # Colorbar with consistent size
    cbar = add_colorbar(ax, contourf, r"$u'_{rms}/U_\infty$", tick_size=12, label_size=14, shrink=colorbar_shrink)
    return cbar


if __name__ == "__main__":
    # Paths (existing cached outputs)
    AOA5_DATA = "/home/jofre/Members/Eduard/Paper2/Simulations/NACA_0012_AOA5_Re50000_1716x1662x128/Mean_data/AoA5_Re50000_velocity_profiles_dense_data.h5"
    AOA5_RMS = "/home/jofre/Members/Eduard/Paper2/Simulations/NACA_0012_AOA5_Re50000_1716x1662x128/Mean_data/AoA5_Re50000_velocity_RMS_profiles_data_mpi_2.h5"
    AOA5_XC_ORIG = np.array([0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.95])

    AOA12_DATA = "/home/jofre/Members/Eduard/Paper2/Simulations/NACA_0012_AOA12_Re50000_1716x1662x128/Mean_data/AoA12_Re50000_velocity_profiles_dense_data_2.h5"
    AOA12_RMS = "/home/jofre/Members/Eduard/Paper2/Simulations/NACA_0012_AOA12_Re50000_1716x1662x128/Mean_data/AoA12_Re50000_velocity_RMS_profiles_data_mpi_2.h5"
    AOA12_XC_ORIG = np.array([0.05, 0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.95])

    fig, axes = plt.subplots(
        2,
        1,
        figsize=(12, 9),
        gridspec_kw={"height_ratios": [1.0, 1.9]},
        constrained_layout=True,
    )

    # Colorbar shrink factors: with height_ratios [1.0, 1.9], 
    # to match physical colorbar size, use shrink = 1.0/1.9 = 0.526 for bottom
    cbar1 = plot_case(
        axes[0],
        AOA5_DATA,
        AOA5_RMS,
        AOA5_XC_ORIG,
        xlim=(-0.05, 1.05),
        ylim=(-0.12, 0.12),
        recompute_isoline=False,
        mark_reattachment=True,
        ref_sep_reatt_path="/home/jofre/Members/Eduard/Paper2/Python_scripts/Mean_data/U_mean_profile_data/Re5e4_AOA5_separtion-reattachment.dat",
        ref_profiles_dir="/home/jofre/Members/Eduard/Paper2/Python_scripts/Mean_data/U_mean_profile_data/",
        colorbar_shrink=1.0,
    )

    cbar2 = plot_case(
        axes[1],
        AOA12_DATA,
        AOA12_RMS,
        AOA12_XC_ORIG,
        xlim=(-0.05, 1.05),
        ylim=(-0.25, 0.22),
        recompute_isoline=True,
        mark_reattachment=False,
        ref_profiles_dir=None,
        colorbar_shrink=0.526,
    )

    # Save combined plot
    out_dir = "/home/jofre/Members/Eduard/Paper2/Figures"
    os.makedirs(out_dir, exist_ok=True)
    out_png = os.path.join(out_dir, "combined_mean_velocity_profiles.png")
    out_eps = os.path.join(out_dir, "combined_mean_velocity_profiles.eps")
    plt.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.savefig(out_eps, dpi=300, bbox_inches="tight")
    print(f"Saved combined plots to: {out_png}")

    # Generate individual plots with exact same dimensions as original working scripts
    # Plot 1: AoA5 (y_range=0.24, x_range=1.10, ratio=0.218, height increased to 3.0 for labels)
    # fig1, ax1 = plt.subplots(figsize=(12, 2.618))
    fig1, ax1 = plt.subplots(figsize=(12, 3))
    cbar1_individual = plot_case(
        ax1,
        AOA5_DATA,
        AOA5_RMS,
        AOA5_XC_ORIG,
        xlim=(-0.05, 1.05),
        ylim=(-0.12, 0.12),
        recompute_isoline=False,
        mark_reattachment=True,
        ref_sep_reatt_path="/home/jofre/Members/Eduard/Paper2/Python_scripts/Mean_data/U_mean_profile_data/Re5e4_AOA5_separtion-reattachment.dat",
        ref_profiles_dir="/home/jofre/Members/Eduard/Paper2/Python_scripts/Mean_data/U_mean_profile_data/",
        colorbar_shrink=1.0,
    )
    #fig1.subplots_adjust(left=0.1, right=0.92, top=0.95, bottom=0.18)
    out_png_aoa5 = os.path.join(out_dir, "mean_velocity_profiles_AoA5.png")
    out_eps_aoa5 = os.path.join(out_dir, "mean_velocity_profiles_AoA5.eps")
    fig1.savefig(out_png_aoa5, dpi=300)
    fig1.savefig(out_eps_aoa5, dpi=300)
    print(f"Saved AoA5 plot to: {out_png_aoa5}")

    # Plot 2: AoA12 (y_range=0.47, x_range=1.10, ratio=0.427, height=3.0*1.958=5.875 to match AoA5 ratio)
    fig2, ax2 = plt.subplots(figsize=(12, 5.875))
    cbar2_individual = plot_case(
        ax2,
        AOA12_DATA,
        AOA12_RMS,
        AOA12_XC_ORIG,
        xlim=(-0.05, 1.05),
        ylim=(-0.25, 0.22),
        recompute_isoline=True,
        mark_reattachment=False,
        ref_profiles_dir=None,
        colorbar_shrink=(3.0/5.875),  # Match colorbar physical size to AoA5
    )
    #fig2.subplots_adjust(left=0.1, right=0.92, top=0.95, bottom=0.18)
    out_png_aoa12 = os.path.join(out_dir, "mean_velocity_profiles_AoA12.png")
    out_eps_aoa12 = os.path.join(out_dir, "mean_velocity_profiles_AoA12.eps")
    fig2.savefig(out_png_aoa12, dpi=300)
    fig2.savefig(out_eps_aoa12, dpi=300)
    print(f"Saved AoA12 plot to: {out_png_aoa12}")

    plt.show()

    

"""
%%BoundingBox: 0 21 864 359
%%HiResBoundingBox: 0.000000 21.363636 864.000000 358.909090
"""
