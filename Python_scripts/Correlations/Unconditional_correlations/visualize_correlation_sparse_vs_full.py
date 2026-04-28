import os
import sys
import h5py
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy.interpolate import griddata
from glob import glob

# ============================================================================
# Configuration
# ============================================================================

# Base directory for correlation results
CORRELATION_DIR_SPARSE = "/home/jofre/Members/Eduard/Paper2/Simulations/NACA_0012_AOA5_Re50000_1716x1662x128/Mean_data/Wall_shear_correlations/test_3_sparse/"
CORRELATION_DIR_FULL = "/home/jofre/Members/Eduard/Paper2/Simulations/NACA_0012_AOA5_Re50000_1716x1662x128/Mean_data/Wall_shear_correlations/test_3/"

# Chord locations to compare
X_C_LOCATIONS = [0.3, 0.5, 0.7, 0.9]

# Z-slice indices to visualize (relative separation Δz)
Z_SLICE_INDICES = [0, 5, 10, 15]

# Y-slice index for xy-plane visualization
Y_SLICE_IDX = None  # Will be set to middle y-index if not specified

# ============================================================================
# Load Correlation Files
# ============================================================================

def load_correlation_file(filepath):
    """Load correlation results from HDF5 file."""
    try:
        with h5py.File(filepath, 'r') as f:
            data = {
                'r': f['R'][:],
                'u_rms': f['u_rms'][:],
                'x': f['x'][:],
                'y': f['y'][:],
                'z': f['z'][:],
                'attrs': dict(f.attrs)
            }
        return data
    except Exception as e:
        print(f"Error loading {filepath}: {e}")
        return None


def interpolate_sparse_to_full_grid(sparse_data, full_data):
    """
    Interpolate sparse correlation field to full resolution grid.
    Returns interpolated field and relative error field.
    """
    r_sparse = sparse_data['r']
    r_full = full_data['r']

    x_sparse = sparse_data['x']
    y_sparse = sparse_data['y']
    z_sparse = sparse_data['z']

    x_full = full_data['x']
    y_full = full_data['y']
    z_full = full_data['z']

    Nz, Ny_sparse, Nx_sparse = r_sparse.shape
    Nz_full, Ny_full, Nx_full = r_full.shape

    # Interpolate for each z-slice
    r_interp = np.zeros_like(r_full)
    error_field = np.zeros_like(r_full)

    for k in range(min(Nz, Nz_full)):
        # Extract 2D grids
        x_s_2d = x_sparse[k, :, :]  # (Ny_sparse, Nx_sparse)
        y_s_2d = y_sparse[k, :, :]
        r_s_2d = r_sparse[k, :, :]

        x_f_2d = x_full[k, :, :]  # (Ny_full, Nx_full)
        y_f_2d = y_full[k, :, :]

        # Flatten for griddata
        points = np.column_stack((x_s_2d.ravel(), y_s_2d.ravel()))
        values = r_s_2d.ravel()
        xi = np.column_stack((x_f_2d.ravel(), y_f_2d.ravel()))

        # Interpolate using linear method
        r_interp_2d = griddata(points, values, xi, method='linear', fill_value=0.0)
        r_interp_2d = np.where(np.isnan(r_interp_2d), 0.0, r_interp_2d)
        r_interp[k, :, :] = r_interp_2d.reshape(x_f_2d.shape)

        # Compute relative error
        denominator = np.abs(r_full[k, :, :]) + 1e-12
        error_field[k, :, :] = np.abs(r_interp[k, :, :] - r_full[k, :, :]) / denominator

    return r_interp, error_field


# ============================================================================
# Main Visualization Loop
# ============================================================================

print("=" * 70)
print("SPARSE VS FULL-RESOLUTION CORRELATION VISUALIZATION")
print("=" * 70)

for x_c_target in X_C_LOCATIONS:
    print(f"\n{'=' * 70}")
    print(f"Processing x/c = {x_c_target:.2f}")
    print(f"{'=' * 70}")

    # Find files
    sparse_file = os.path.join(
        CORRELATION_DIR_SPARSE,
        f"wall_shear_correlation_unconditional_sparse_xc_{x_c_target:.3f}_sx5_sy25_sz1.h5"
    )

    full_file = os.path.join(
        CORRELATION_DIR_FULL,
        f"wall_shear_correlation_unconditional_xc_{x_c_target:.3f}.h5"
    )

    # Check if files exist
    if not os.path.exists(sparse_file):
        print(f"  WARNING: Sparse file not found: {sparse_file}")
        continue

    if not os.path.exists(full_file):
        print(f"  WARNING: Full file not found: {full_file}")
        continue

    # Load data
    print(f"  Loading sparse data...")
    sparse_data = load_correlation_file(sparse_file)

    print(f"  Loading full-resolution data...")
    full_data = load_correlation_file(full_file)

    if sparse_data is None or full_data is None:
        print(f"  ERROR: Could not load data")
        continue

    # Print metadata
    print(f"\n  Sparse data shape: {sparse_data['r'].shape}")
    print(f"  Full data shape: {full_data['r'].shape}")
    print(f"  Sparse stride_x: {sparse_data['attrs'].get('stride_x', 'N/A')}")
    print(f"  Sparse stride_y: {sparse_data['attrs'].get('stride_y', 'N/A')}")
    print(f"  Sparse stride_z: {sparse_data['attrs'].get('stride_z', 'N/A')}")

    Nz, Ny_sparse, Nx_sparse = sparse_data['r'].shape
    Nz_full, Ny_full, Nx_full = full_data['r'].shape

    memory_reduction = (Ny_full * Nx_full) / (Ny_sparse * Nx_sparse)
    print(f"  Memory reduction factor: {memory_reduction:.2f}x")

    # Interpolate sparse to full grid
    print(f"  Interpolating sparse data to full grid...")
    r_interp, error_field = interpolate_sparse_to_full_grid(sparse_data, full_data)

    # ====================================================================
    # Figure 1: Z-Slices Comparison
    # ====================================================================
    print(f"  Creating z-slice comparison figure...")

    n_z_slices = len(Z_SLICE_INDICES)
    fig, axes = plt.subplots(n_z_slices, 3, figsize=(18, 5*n_z_slices))

    if n_z_slices == 1:
        axes = axes.reshape(1, -1)

    for row, k in enumerate(Z_SLICE_INDICES):
        if k >= Nz_full:
            continue

        # Full resolution
        im0 = axes[row, 0].contourf(full_data['x'][k, :, :],
                                     full_data['y'][k, :, :],
                                     np.abs(full_data['r'][k, :, :]),
                                     levels=20, cmap='RdYlBu_r')
        axes[row, 0].set_title(f'Full Resolution (Δz={k})', fontsize=12, fontweight='bold')
        axes[row, 0].set_xlabel('x/c')
        axes[row, 0].set_ylabel('y/c')
        axes[row, 0].set_aspect('equal')
        plt.colorbar(im0, ax=axes[row, 0], label='|R|')

        # Interpolated sparse
        im1 = axes[row, 1].contourf(full_data['x'][k, :, :],
                                     full_data['y'][k, :, :],
                                     np.abs(r_interp[k, :, :]),
                                     levels=20, cmap='RdYlBu_r')
        axes[row, 1].set_title(f'Interpolated Sparse (Δz={k})', fontsize=12, fontweight='bold')
        axes[row, 1].set_xlabel('x/c')
        axes[row, 1].set_ylabel('y/c')
        axes[row, 1].set_aspect('equal')
        plt.colorbar(im1, ax=axes[row, 1], label='|R|')

        # Difference
        diff = full_data['r'][k, :, :] - r_interp[k, :, :]
        im2 = axes[row, 2].contourf(full_data['x'][k, :, :],
                                     full_data['y'][k, :, :],
                                     diff,
                                     levels=20, cmap='seismic')
        axes[row, 2].set_title(f'Difference (Δz={k})', fontsize=12, fontweight='bold')
        axes[row, 2].set_xlabel('x/c')
        axes[row, 2].set_ylabel('y/c')
        axes[row, 2].set_aspect('equal')
        plt.colorbar(im2, ax=axes[row, 2], label='R_full - R_interp')

    plt.suptitle(f'Correlation Field Comparison: x/c = {x_c_target:.2f}',
                 fontsize=16, fontweight='bold', y=1.00)
    plt.tight_layout()
    print(f"    Displaying z-slice comparison figure...")
    plt.show()

    # ====================================================================
    # Figure 2: XY-Plane Slices (at different z)
    # ====================================================================
    print(f"  Creating xy-plane slices figure...")

    if Y_SLICE_IDX is None:
        y_idx = Ny_full // 2
    else:
        y_idx = Y_SLICE_IDX

    z_indices_xy = [0, Nz_full//4, Nz_full//2, 3*Nz_full//4]
    n_z_xy = len(z_indices_xy)

    fig, axes = plt.subplots(n_z_xy, 2, figsize=(14, 5*n_z_xy))

    if n_z_xy == 1:
        axes = axes.reshape(1, -1)

    for row, k in enumerate(z_indices_xy):
        if k >= Nz_full:
            continue

        # Full resolution xy-slice
        im0 = axes[row, 0].contourf(full_data['x'][k, :, :],
                                     full_data['y'][k, :, :],
                                     np.abs(full_data['r'][k, :, :]),
                                     levels=20, cmap='RdYlBu_r')
        axes[row, 0].contour(full_data['x'][k, :, :],
                             full_data['y'][k, :, :],
                             np.abs(full_data['r'][k, :, :]),
                             levels=5, colors='black', alpha=0.3, linewidths=0.5)
        axes[row, 0].set_title(f'Full Resolution (Δz={k})', fontsize=12, fontweight='bold')
        axes[row, 0].set_xlabel('x/c')
        axes[row, 0].set_ylabel('y/c')
        axes[row, 0].set_aspect('equal')
        plt.colorbar(im0, ax=axes[row, 0], label='|R|')

        # Sparse data with sampling points overlay
        im1 = axes[row, 1].contourf(full_data['x'][k, :, :],
                                     full_data['y'][k, :, :],
                                     np.abs(r_interp[k, :, :]),
                                     levels=20, cmap='RdYlBu_r')
        axes[row, 1].contour(full_data['x'][k, :, :],
                             full_data['y'][k, :, :],
                             np.abs(r_interp[k, :, :]),
                             levels=5, colors='black', alpha=0.3, linewidths=0.5)

        # Overlay sparse sample points
        if k < sparse_data['r'].shape[0]:
            axes[row, 1].scatter(sparse_data['x'][k, :, :],
                                sparse_data['y'][k, :, :],
                                s=3, c='black', alpha=0.4, label='Sparse samples')

        axes[row, 1].set_title(f'Interpolated Sparse + Sample Points (Δz={k})', fontsize=12, fontweight='bold')
        axes[row, 1].set_xlabel('x/c')
        axes[row, 1].set_ylabel('y/c')
        axes[row, 1].set_aspect('equal')
        plt.colorbar(im1, ax=axes[row, 1], label='|R|')
        if k < sparse_data['r'].shape[0]:
            axes[row, 1].legend(loc='upper right', fontsize=9)

    plt.suptitle(f'XY-Plane Correlation Comparison: x/c = {x_c_target:.2f}',
                 fontsize=16, fontweight='bold', y=1.00)
    plt.tight_layout()
    print(f"    Displaying xy-plane comparison figure...")
    plt.show()

    # ====================================================================
    # Figure 3: Error Analysis
    # ====================================================================
    print(f"  Creating error analysis figure...")

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Panel 1: Peak |R| along z
    r_max_full = np.max(np.abs(full_data['r']), axis=(1, 2))
    r_max_interp = np.max(np.abs(r_interp), axis=(1, 2))

    axes[0, 0].plot(range(Nz_full), r_max_full, 'o-', label='Full resolution', linewidth=2, markersize=4)
    axes[0, 0].plot(range(Nz_full), r_max_interp, 's-', label='Interpolated sparse', linewidth=2, markersize=4)
    axes[0, 0].set_xlabel('Δz (spanwise separation)')
    axes[0, 0].set_ylabel('Peak |R|')
    axes[0, 0].set_title('Peak Correlation vs Spanwise Separation', fontweight='bold')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # Panel 2: Mean absolute error
    mae_per_z = np.mean(np.abs(error_field), axis=(1, 2))
    axes[0, 1].plot(range(Nz_full), mae_per_z, 'o-', color='red', linewidth=2, markersize=4)
    axes[0, 1].set_xlabel('Δz (spanwise separation)')
    axes[0, 1].set_ylabel('Mean Absolute Error (\%)')
    axes[0, 1].set_title('Relative Error vs Spanwise Separation', fontweight='bold')
    axes[0, 1].grid(True, alpha=0.3)

    # Panel 3: 2D error field (middle z-plane)
    k_mid = Nz_full // 2
    im3 = axes[1, 0].contourf(full_data['x'][k_mid, :, :],
                               full_data['y'][k_mid, :, :],
                               error_field[k_mid, :, :] * 100,
                               levels=20, cmap='hot')
    axes[1, 0].set_xlabel('x/c')
    axes[1, 0].set_ylabel('y/c')
    axes[1, 0].set_title(f'Relative Error Field (Δz={k_mid})', fontweight='bold')
    axes[1, 0].set_aspect('equal')
    plt.colorbar(im3, ax=axes[1, 0], label='Error (\%)')

    # Panel 4: Statistics
    axes[1, 1].axis('off')
    stats_text = f"""
    STATISTICAL COMPARISON (x/c = {x_c_target:.2f})
    ═══════════════════════════════════════════

    Spatial Resolution:
      Full:    {Ny_full} × {Nx_full} × {Nz_full}
      Sparse:  {Ny_sparse} × {Nx_sparse} × {Nz_full}
      Reduction: {memory_reduction:.1f}x

    Correlation Statistics (Full):
      Peak |R|:  {np.max(np.abs(full_data['r'])):.4f}
      Mean |R|:  {np.mean(np.abs(full_data['r'])):.4f}
      Std |R|:   {np.std(np.abs(full_data['r'])):.4f}

    Correlation Statistics (Interpolated Sparse):
      Peak |R|:  {np.max(np.abs(r_interp)):.4f}
      Mean |R|:  {np.mean(np.abs(r_interp)):.4f}
      Std |R|:   {np.std(np.abs(r_interp)):.4f}

    Interpolation Error:
      Mean Abs Error: {np.mean(error_field):.4f}
      Max Abs Error:  {np.max(error_field):.4f}
      RMS Error:      {np.sqrt(np.mean(error_field**2)):.4f}

    Peak Error by Δz:
      Min: {np.min(mae_per_z):.4f}
      Max: {np.max(mae_per_z):.4f}
      Mean: {np.mean(mae_per_z):.4f}
    """

    axes[1, 1].text(0.05, 0.95, stats_text, transform=axes[1, 1].transAxes,
                   fontfamily='monospace', fontsize=10, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.suptitle(f'Error Analysis: x/c = {x_c_target:.2f}',
                 fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()
    print(f"    Displaying error analysis figure...")
    plt.show()

    # ====================================================================
    # Figure 4: Sparse Sampling Grid Visualization
    # ====================================================================
    print(f"  Creating sparse sampling grid visualization...")

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Full grid
    axes[0].scatter(full_data['x'][0, :, :], full_data['y'][0, :, :],
                   s=1, c='blue', alpha=0.3, label='Full resolution points')
    axes[0].set_xlabel('x/c', fontsize=12)
    axes[0].set_ylabel('y/c', fontsize=12)
    axes[0].set_title(f'Full Resolution Grid ({Ny_full}×{Nx_full} points)', fontsize=13, fontweight='bold')
    axes[0].legend(fontsize=10)
    axes[0].set_aspect('equal')
    axes[0].grid(True, alpha=0.3)

    # Sparse grid overlay
    axes[1].scatter(full_data['x'][0, :, :], full_data['y'][0, :, :],
                   s=1, c='lightblue', alpha=0.2, label='Full resolution points')
    axes[1].scatter(sparse_data['x'][0, :, :], sparse_data['y'][0, :, :],
                   s=10, c='red', alpha=0.7, label=f'Sparse points ({Ny_sparse}×{Nx_sparse})')
    axes[1].set_xlabel('x/c', fontsize=12)
    axes[1].set_ylabel('y/c', fontsize=12)
    axes[1].set_title(f'Sparse Sampling Grid Overlay (Reduction: {memory_reduction:.1f}x)',
                     fontsize=13, fontweight='bold')
    axes[1].legend(fontsize=10)
    axes[1].set_aspect('equal')
    axes[1].grid(True, alpha=0.3)

    plt.suptitle(f'Spatial Sampling Comparison: x/c = {x_c_target:.2f}',
                 fontsize=16, fontweight='bold')
    plt.tight_layout()
    print(f"    Displaying sparse sampling grid figure...")
    plt.show()

print("\n" + "=" * 70)
print("VISUALIZATION COMPLETE")
print("=" * 70)
