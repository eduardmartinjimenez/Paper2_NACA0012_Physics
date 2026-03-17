import argparse
import os
import sys
import h5py
from stl import mesh

from Geometrical_plots import plot_projection_interface_points_and_stl


# DEFAULT_GEOMETRY_FILE = (
#     "/home/jofre/Members/Eduard/Paper2/Simulations/"
#     "NACA_0012_AOA5_Re50000_1716x1662x128/Geometrical_data/"
#     "3d_NACA0012_Re50000_AoA5_Geometrical_Data.h5"
# )

# DEFAULT_MESH_FILE = (
#     "/home/jofre/Members/Eduard/Paper2/Simulations/"
#     "NACA_0012_AOA5_Re50000_1716x1662x128/Geometrical_data/"
#     "3d_NACA0012_Re50000_AoA5-CROP-MESH.h5"
# )

# DEFAULT_STL_FILE = (
#     "/home/jofre/Members/Eduard/Paper2/Simulations/"
#     "NACA_0012_AOA5_Re50000_1716x1662x128/Geometrical_data/naca0012.stl"
# )

DEFAULT_GEOMETRY_FILE = (
    "/home/jofre/Members/Eduard/Paper2/Simulations/"
    "Test_Re1e4/Geometrical_data/"
    "3d_NACA0012_Test_Geometrical_Data.h5"
)

DEFAULT_MESH_FILE = (
    "/home/jofre/Members/Eduard/Paper2/Simulations/"
    "Test_Re1e4/"
    "3d_NACA0012_Re10000_AoA5-CROP-MESH.h5"
)

DEFAULT_STL_FILE = (
    "/home/jofre/Members/Eduard/Paper2/Simulations/"
    "NACA_0012_AOA5_Re50000_1716x1662x128/Geometrical_data/naca0012.stl"
)

DATA_LOADER_MODULE_PATH = "/home/jofre/Members/Eduard/Paper2/Python_scripts/Data_loader"

if DATA_LOADER_MODULE_PATH not in sys.path:
    sys.path.append(DATA_LOADER_MODULE_PATH)

from data_loader_functions import CompressedSnapshotLoader


def print_dataset_info(name, dataset):
    print(f"{name}:")
    print(f"  shape: {dataset.shape}")
    print(f"  dtype: {dataset.dtype}")

    if dataset.size == 0:
        print("  values: <empty>")
        return

    data = dataset[()]

    if data.ndim == 1:
        print(f"  first 10 values: {data[:10]}")
    else:
        n_rows = min(5, data.shape[0])
        print(f"  first {n_rows} rows:")
        print(data[:n_rows])


def load_and_print_geometrical_data(file_path):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Geometrical data file not found: {file_path}")

    print(f"Loading geometrical data from: {file_path}")
    loaded_data = {}
    with h5py.File(file_path, "r") as h5_file:
        print("\nAvailable datasets:")
        for dataset_name in h5_file.keys():
            dataset = h5_file[dataset_name]
            print_dataset_info(dataset_name, dataset)
            loaded_data[dataset_name] = dataset[()]
            print()

    return loaded_data


def plot_geometrical_data(loaded_data, mesh_file_path, stl_file_path):
    required_keys = ["interface_indices_i", "interface_indices_j", "proj_points"]
    for key in required_keys:
        if key not in loaded_data:
            print(f"Skipping plot: missing dataset '{key}' in HDF5 file.")
            return

    if not os.path.exists(mesh_file_path):
        print(f"Skipping plot: mesh file not found at {mesh_file_path}")
        return

    if not os.path.exists(stl_file_path):
        print(f"Skipping plot: STL file not found at {stl_file_path}")
        return

    print("Loading mesh and STL for plotting...")
    loader = CompressedSnapshotLoader(mesh_file_path)
    triangles = mesh.Mesh.from_file(stl_file_path).vectors

    interface_indices_i = loaded_data["interface_indices_i"].astype(int).tolist()
    interface_indices_j = loaded_data["interface_indices_j"].astype(int).tolist()
    proj_points = loaded_data["proj_points"]

    print("Displaying projection/interface/STL plot...")
    plot_projection_interface_points_and_stl(
        loader.x,
        loader.y,
        interface_indices_i,
        interface_indices_j,
        proj_points,
        triangles,
    )


def parse_args():
    parser = argparse.ArgumentParser(
        description="Load and print geometrical data from an HDF5 file."
    )
    parser.add_argument(
        "--file",
        default=DEFAULT_GEOMETRY_FILE,
        help=f"Path to geometrical HDF5 file (default: {DEFAULT_GEOMETRY_FILE})",
    )
    parser.add_argument(
        "--mesh-file",
        default=DEFAULT_MESH_FILE,
        help=f"Path to compressed mesh HDF5 file used for plotting (default: {DEFAULT_MESH_FILE})",
    )
    parser.add_argument(
        "--stl-file",
        default=DEFAULT_STL_FILE,
        help=f"Path to STL geometry file used for plotting (default: {DEFAULT_STL_FILE})",
    )
    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="Disable plotting and only print HDF5 dataset information.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    data = load_and_print_geometrical_data(arguments.file)
    if not arguments.no_plot:
        plot_geometrical_data(data, arguments.mesh_file, arguments.stl_file)