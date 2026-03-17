import os
import glob
import re
import h5py
import numpy as np
import gc

# ============================================================================
# Configuration
# ============================================================================

# Root directory that contains the batch_XXXXXXX folders
STEADY_STATE_DIR = "/media/disc2/jofre/Members/Eduard/NACA_0012_AOA5_Re50000_1716x1662x128/Steady_state"

# Mesh file is stored separately, outside the batch folders
MESH_DIR      = "/home/jofre/Members/Eduard/Paper2/Simulations/NACA_0012_AOA5_Re50000_1716x1662x128/Geometrical_data/"
MESH_BASENAME = "3d_NACA0012_Re50000_AoA5"

# Basename used in snapshot filenames inside each batch
SNAP_BASENAME = "3d_NACA0012_Re50000_AoA5"

# Spanwise plane to extract (0-based index along the z axis of the mesh)
Z_PLANE_IDX = 65

# Zero-padding width for snapshot iteration in output filenames
# Example output: 3d_NACA0012_Re50000_AoA5_2D_06350000.h5
ITER_PAD_DIGITS = 8

# ============================================================================
# Helpers
# ============================================================================

def find_batches(root_dir):
    """Return sorted list of batch_XXXXXXX directories."""
    batches = sorted(
        d for d in glob.glob(os.path.join(root_dir, "batch_*"))
        if os.path.isdir(d) and re.match(r"batch_\d+$", os.path.basename(d))
    )
    return batches


def find_instants(batch_dir, basename):
    """Return sorted snapshot step numbers found in batch_dir."""
    steps = []
    for fpath in glob.glob(os.path.join(batch_dir, f"{basename}_*-COMP-DATA.h5")):
        m = re.search(r"_(\d+)-COMP-DATA\.h5$", fpath)
        if m:
            steps.append(int(m.group(1)))
    steps.sort()
    return steps


def save_xdmf_2d(output_dir, output_base, shape_2d, mesh_h5_name):
    """Write a 2D XDMF sidecar file for ParaView/VisIt."""
    xdmf_path    = os.path.join(output_dir, f"{output_base}.xdmf")
    data_h5_name = f"{output_base}.h5"
    dims         = f"{shape_2d[0]} {shape_2d[1]}"

    with open(xdmf_path, "w") as f:
        f.write("<?xml version='1.0' ?>\n")
        f.write("<!DOCTYPE Xdmf SYSTEM 'Xdmf.dtd' []>\n")
        f.write("<Xdmf Version='2.0'>\n")
        f.write("  <Domain>\n")
        f.write(f"    <Grid Name='{output_base}' GridType='Uniform'>\n")
        f.write(f"      <Topology TopologyType='2DSMesh' Dimensions='{dims}'/>\n")
        f.write(f"      <Geometry GeometryType='X_Y'>\n")
        for coord in ("x", "y"):
            f.write(f"        <DataItem Name='{coord}' Dimensions='{dims}' "
                    f"NumberType='Float' Precision='8' Format='HDF'>\n")
            f.write(f"            {mesh_h5_name}:/{coord}\n")
            f.write(f"        </DataItem>\n")
        f.write(f"      </Geometry>\n")
        for field in ("u", "v", "w", "p"):
            f.write(f"      <Attribute Name='{field}' AttributeType='Scalar' Center='Node'>\n")
            f.write(f"        <DataItem Dimensions='{dims}' "
                    f"NumberType='Float' Precision='4' Format='HDF'>\n")
            f.write(f"            {data_h5_name}:/{field}\n")
            f.write(f"        </DataItem>\n")
            f.write(f"      </Attribute>\n")
        f.write(f"    </Grid>\n")
        f.write(f"  </Domain>\n")
        f.write(f"</Xdmf>\n")

    return xdmf_path


# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":

    # ------------------------------------------------------------------
    # Load mesh once (shared across all batches and snapshots)
    # ------------------------------------------------------------------
    mesh_h5_path = os.path.join(MESH_DIR, f"{MESH_BASENAME}-CROP-MESH.h5")
    if not os.path.exists(mesh_h5_path):
        raise FileNotFoundError(f"Mesh file not found: {mesh_h5_path}")

    print(f"Loading mesh from: {mesh_h5_path}")
    with h5py.File(mesh_h5_path, "r") as mesh:
        Nz, Ny, Nx      = mesh["x"].shape                     # axis order: (z, y, x)
        x_2d            = mesh["x"][Z_PLANE_IDX, :, :]        # (Ny, Nx)
        y_2d            = mesh["y"][Z_PLANE_IDX, :, :]        # (Ny, Nx)
        compressed_topo = mesh["compressed_topology"][:]       # (N, 3)  → [iz, iy, ix]

    print(f"  Full mesh shape (Nz, Ny, Nx) : ({Nz}, {Ny}, {Nx})")
    print(f"  Extracting z-plane index : {Z_PLANE_IDX}  (valid range 0–{Nz-1})")

    # Pre-filter topology to rows belonging to the target z-plane only.
    # topo[:, 0] = z-index, topo[:, 1] = y-index, topo[:, 2] = x-index
    z_mask   = compressed_topo[:, 0] == Z_PLANE_IDX
    y_idx_2d = compressed_topo[z_mask, 1]   # (M,)
    x_idx_2d = compressed_topo[z_mask, 2]   # (M,)

    print(f"  Fluid points in z-plane {Z_PLANE_IDX}: {z_mask.sum()} / {len(z_mask)}\n")

    del compressed_topo
    gc.collect()

    # ------------------------------------------------------------------
    # Discover batch directories
    # ------------------------------------------------------------------
    batches = find_batches(STEADY_STATE_DIR)
    if not batches:
        raise RuntimeError(f"No batch_XXXXXXX directories found in:\n  {STEADY_STATE_DIR}")

    print(f"Found {len(batches)} batch(es):")
    for b in batches:
        print(f"  {os.path.basename(b)}")
    print()

    # ------------------------------------------------------------------
    # Process each batch
    # ------------------------------------------------------------------
    for batch_dir in batches:
        batch_name = os.path.basename(batch_dir)
        output_dir = os.path.join(batch_dir, "2d_snapshots")
        os.makedirs(output_dir, exist_ok=True)

        # Save the 2D mesh slice into the output folder once (reuse if present)
        mesh_2d_name = f"{MESH_BASENAME}-CROP-MESH-2D-Z{Z_PLANE_IDX}.h5"
        mesh_2d_path = os.path.join(output_dir, mesh_2d_name)
        if not os.path.exists(mesh_2d_path):
            print(f"[{batch_name}] Saving 2D mesh slice → {mesh_2d_path}")
            with h5py.File(mesh_2d_path, "w") as mf:
                mf.create_dataset("x", data=x_2d, dtype="float64")
                mf.create_dataset("y", data=y_2d, dtype="float64")

        instants = find_instants(batch_dir, SNAP_BASENAME)
        if not instants:
            print(f"[{batch_name}] No COMP-DATA snapshots found — skipping.\n")
            continue

        print(f"[{batch_name}] {len(instants)} snapshot(s) found: {instants}")

        for inst in instants:
            comp_path = os.path.join(batch_dir, f"{SNAP_BASENAME}_{inst}-COMP-DATA.h5")
            inst_str  = str(inst).zfill(ITER_PAD_DIGITS)
            output_base = f"{SNAP_BASENAME}_2D_{inst_str}"
            out_h5    = os.path.join(output_dir, f"{output_base}.h5")

            if os.path.exists(out_h5):
                print(f"  [{inst}] Already exists, skipping.")
                continue

            if not os.path.exists(comp_path):
                print(f"  [{inst}] COMP-DATA file not found: {comp_path} — skipping.")
                continue

            print(f"  [{inst}] Reading {comp_path}")
            with h5py.File(comp_path, "r") as df:
                u_comp = df["u_compressed"][:]
                v_comp = df["v_compressed"][:]
                w_comp = df["w_compressed"][:]
                p_comp = df["p_compressed"][:]

            # Allocate 2D output arrays (Ny, Nx) and fill only the z-plane points
            u_2d = np.full((Ny, Nx), np.nan, dtype="float32")
            v_2d = np.full((Ny, Nx), np.nan, dtype="float32")
            w_2d = np.full((Ny, Nx), np.nan, dtype="float32")
            p_2d = np.full((Ny, Nx), np.nan, dtype="float32")

            u_2d[y_idx_2d, x_idx_2d] = u_comp[z_mask]
            v_2d[y_idx_2d, x_idx_2d] = v_comp[z_mask]
            w_2d[y_idx_2d, x_idx_2d] = w_comp[z_mask]
            p_2d[y_idx_2d, x_idx_2d] = p_comp[z_mask]

            del u_comp, v_comp, w_comp, p_comp
            gc.collect()

            print(f"  [{inst}] Saving {out_h5}")
            with h5py.File(out_h5, "w") as of:
                of.create_dataset("u", data=u_2d, dtype="float32")
                of.create_dataset("v", data=v_2d, dtype="float32")
                of.create_dataset("w", data=w_2d, dtype="float32")
                of.create_dataset("p", data=p_2d, dtype="float32")

            del u_2d, v_2d, w_2d, p_2d
            gc.collect()

            xdmf_path = save_xdmf_2d(output_dir, output_base,
                                     (Ny, Nx), mesh_2d_name)
            print(f"  [{inst}] Saved XDMF → {xdmf_path}")

        print()

    print("Done.")
