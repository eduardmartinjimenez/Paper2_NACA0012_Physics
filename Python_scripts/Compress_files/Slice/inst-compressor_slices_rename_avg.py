import os, glob, re
import h5py
import numpy as np
import gc

# Define input file path and base name
INPUT_FILE_PATH = "/home/jofre/Members/Eduard/Paper2/Simulations/NACA_0012_AOA5_Re50000_1716x1662x128/Slices_data/slices_batch_1/slice_9_compr/last_slice/"
INPUT_FILE_BASENAME = "slice_9_output"

# Define ouput folder path
OUTPUT_FILE_PATH = "/home/jofre/Members/Eduard/Paper2/Simulations/NACA_0012_AOA5_Re50000_1716x1662x128/Slices_data/slices_batch_1/slice_9_compr/last_slice/"
# Rename ONLY the output files. If None, it uses INPUT_FILE_BASENAME.
OUTPUT_FILE_BASENAME = "slice_9"

# Define crop region:
X_MIN = -0.5
Y_MIN = 0.0
Z_MIN = -1.0

X_MAX = 8.0
Y_MAX = 1.0
Z_MAX = 1.0

def read_instants():
    steps = []
    print(INPUT_FILE_PATH)
    for file in glob.glob(INPUT_FILE_PATH + f"{INPUT_FILE_BASENAME}_*.h5"):
        steps.append(int(re.split("[_ .]", file)[-2]))
    steps.sort()
    return steps


def crop_3d_by_physical_coords(data, coord_array, min_coords, max_coords):
    """
    Crop a 3D array using physical coordinate bounds from a coordinate tensor.

    Parameters:
        data (np.ndarray): The original 3D array with shape (D, H, W).
        coord_array (np.ndarray): Array of shape (D, H, W, 3) with (X, Y, Z) coordinates per voxel.
        min_coords (tuple): Minimum (X, Y, Z) values as a 3-tuple.
        max_coords (tuple): Maximum (X, Y, Z) values as a 3-tuple.

    Returns:
        cropped_data (np.ndarray): Cropped 3D array.
        crop_bounds (tuple): ((z_min, z_max), (y_min, y_max), (x_min, x_max)) index bounds.
    """
    if data.shape != coord_array.shape[:3]:
        raise ValueError(
            "coord_array must have shape (D, H, W, 3) matching data shape (D, H, W)."
        )

    # Create mask for each axis
    mask = np.ones(data.shape, dtype=bool)
    for axis, (min_val, max_val) in enumerate(zip(min_coords, max_coords)):
        # print("axis=",axis," min=",min_val," max=",max_val)
        axis_vals = coord_array[..., axis]
        # print("ax vals ",axis_vals)
        mask &= (axis_vals >= min_val) & (axis_vals <= max_val)
        # print("mask ",axis_vals)

    # Find the bounding box of the region where mask is True
    indices = np.argwhere(mask)
    if indices.size == 0:
        raise ValueError("No voxels found within the given coordinate range.")

    z_min, y_min, x_min = indices.min(axis=0)
    z_max, y_max, x_max = indices.max(axis=0) + 1  # +1 to include upper bounds

    cropped_data = data[z_min:z_max, y_min:y_max, x_min:x_max]
    # return cropped_data, ((z_min, z_max), (y_min, y_max), (x_min, x_max))
    return cropped_data


def compress_3d_array(data: np.ndarray, mask: np.ndarray):
    """
    Eliminates positions in a 3D NumPy array based on a mask and returns:
    - The values of the non-eliminated positions
    - Their indices in the original array

    Parameters:
    - data (np.ndarray): A 3D array of data.
    - mask (np.ndarray): A 3D boolean array of the same shape as `data`.
                         Positions where `mask` is True will be eliminated.

    Returns:
    - compressed_values (np.ndarray): 1D array of values from `data` where mask is False.
    - compressed_topo (np.ndarray): 2D array of shape (N, 3), with the indices of these values.
    """
    if data.shape != mask.shape:
        raise ValueError("`data` and `mask` must have the same shape")

    # Get the indices where mask is False (i.e., not eliminated)
    compressed_topo = np.argwhere(~mask)

    # Extract the values at those indices
    compressed_values = data[~mask]

    return compressed_values, compressed_topo


if __name__ == "__main__":
    # Decide final output basename
    OUT_BASE = OUTPUT_FILE_BASENAME if OUTPUT_FILE_BASENAME else INPUT_FILE_BASENAME

    instants = read_instants()
    print("Available file instants:", instants)

    if os.path.exists(OUTPUT_FILE_PATH):
        print(f"Data folder exists! {OUTPUT_FILE_PATH}")
    else:
        print(f"Data folder does not exist: {OUTPUT_FILE_PATH}")

    for inst in instants:

        # check whether we've already compressed this instant
        output_data_path = os.path.join(
            OUTPUT_FILE_PATH,
            f"{OUT_BASE}_{inst}-COMP-DATA.h5"
        )
        if os.path.exists(output_data_path):
            print(f"► Snapshot {inst} already compressed, skipping.")
            continue

        INPUT_FULL_PATH = os.path.join(
            INPUT_FILE_PATH, f"{INPUT_FILE_BASENAME}_{inst}.h5"
        )

        # Check if the data file exists
        if os.path.exists(INPUT_FULL_PATH):
            print(f"reading file: {INPUT_FULL_PATH}")
        else:
            print(f"Data file does not exist: {INPUT_FULL_PATH}")

        # Import data file
        data_file = h5py.File(INPUT_FULL_PATH, "r")

        ### Import 3D Data
        x_data = data_file["x"][:, :, :]
        y_data = data_file["y"][:, :, :]
        z_data = data_file["z"][:, :, :]
        tag_ibm_data = data_file["tag_IBM"][:, :, :]

        u_data = data_file["u"][:, :, :]
        v_data = data_file["v"][:, :, :]
        w_data = data_file["w"][:, :, :]
        p_data = data_file["P"][:, :, :]

        u_avg_data = data_file["avg_u"][:, :, :]
        v_avg_data = data_file["avg_v"][:, :, :]
        w_avg_data = data_file["avg_w"][:, :, :]
        p_avg_data = data_file["avg_P"][:, :, :]

        min_coords = (X_MIN, Y_MIN, Z_MIN)
        max_coords = (X_MAX, Y_MAX, Z_MAX)

        coords_array = np.stack(
            (x_data, y_data, z_data), axis=-1
        )  # Shape: (D, H, W, 3)

        ### Crop 3D spatial data to the specified domain
        tag_ibm_cropped = crop_3d_by_physical_coords(
            tag_ibm_data, coords_array, min_coords, max_coords
        )
        x_cropped = crop_3d_by_physical_coords(
            x_data, coords_array, min_coords, max_coords
        )
        y_cropped = crop_3d_by_physical_coords(
            y_data, coords_array, min_coords, max_coords
        )
        z_cropped = crop_3d_by_physical_coords(
            z_data, coords_array, min_coords, max_coords
        )
        u_cropped = crop_3d_by_physical_coords(
            u_data, coords_array, min_coords, max_coords
        )
        v_cropped = crop_3d_by_physical_coords(
            v_data, coords_array, min_coords, max_coords
        )
        w_cropped = crop_3d_by_physical_coords(
            w_data, coords_array, min_coords, max_coords
        )
        p_cropped = crop_3d_by_physical_coords(
            p_data, coords_array, min_coords, max_coords
        )

        u_avg_cropped = crop_3d_by_physical_coords(
            u_avg_data, coords_array, min_coords, max_coords
        )
        v_avg_cropped = crop_3d_by_physical_coords(
            v_avg_data, coords_array, min_coords, max_coords
        )
        w_avg_cropped = crop_3d_by_physical_coords(
            w_avg_data, coords_array, min_coords, max_coords
        )
        p_avg_cropped = crop_3d_by_physical_coords(
            p_avg_data, coords_array, min_coords, max_coords
        )

        del x_data, y_data, z_data, coords_array, u_data, v_data, w_data, p_data, u_avg_data, v_avg_data, w_avg_data, p_avg_data
        gc.collect()

        ### Compressing data by removing solid points tag_ibm==1
        mask = tag_ibm_cropped > 0.9999

        u_compressed, compressed_topo = compress_3d_array(u_cropped, mask)
        v_compressed, _ = compress_3d_array(v_cropped, mask)
        w_compressed, _ = compress_3d_array(w_cropped, mask)
        p_compressed, _ = compress_3d_array(p_cropped, mask)

        u_avg_compressed, _ = compress_3d_array(u_avg_cropped, mask)
        v_avg_compressed, _ = compress_3d_array(v_avg_cropped, mask)
        w_avg_compressed, _ = compress_3d_array(w_avg_cropped, mask)
        p_avg_compressed, _ = compress_3d_array(p_avg_cropped, mask)

        del u_cropped, v_cropped, w_cropped, p_cropped, mask, u_avg_cropped, v_avg_cropped, w_avg_cropped, p_avg_cropped
        gc.collect()

        ### Saving compressed mesh and topology (only the first file).
        OUTPUT_MESH_PATH = os.path.join(
            OUTPUT_FILE_PATH, f"{OUT_BASE}-CROP-MESH.h5"
        )

        if not os.path.exists(OUTPUT_MESH_PATH):
            print("saving file: ", OUTPUT_MESH_PATH)
            output_file = h5py.File(OUTPUT_MESH_PATH, "w")
            output_file.create_dataset("x", data=x_cropped, dtype="float32")
            output_file.create_dataset("y", data=y_cropped, dtype="float32")
            output_file.create_dataset("z", data=z_cropped, dtype="float32")
            output_file.create_dataset("tag_IBM", data=tag_ibm_cropped, dtype="float32")
            output_file.create_dataset(
                "compressed_topology", data=compressed_topo, dtype="int"
            )
            output_file.close()

        del x_cropped, y_cropped, z_cropped, tag_ibm_cropped, compressed_topo
        gc.collect()

        ### Saving compressed data.
        OUTPUT_DATA_PATH = os.path.join(
            OUTPUT_FILE_PATH, f"{OUT_BASE}_{inst}-COMP-DATA.h5"
        )

        print("saving file: ", OUTPUT_DATA_PATH)
        output_file = h5py.File(OUTPUT_DATA_PATH, "w")
        output_file.create_dataset("u_compressed", data=u_compressed, dtype="float32")
        output_file.create_dataset("v_compressed", data=v_compressed, dtype="float32")
        output_file.create_dataset("w_compressed", data=w_compressed, dtype="float32")
        output_file.create_dataset("p_compressed", data=p_compressed, dtype="float32")
        output_file.create_dataset(
            "u_avg_compressed", data=u_avg_compressed, dtype="float32"
        )
        output_file.create_dataset(
            "v_avg_compressed", data=v_avg_compressed, dtype="float32"
        )
        output_file.create_dataset(
            "w_avg_compressed", data=w_avg_compressed, dtype="float32"
        )
        output_file.create_dataset(
            "p_avg_compressed", data=p_avg_compressed, dtype="float32"
        )
        output_file.close()

        del u_compressed, v_compressed, w_compressed, p_compressed, u_avg_compressed, v_avg_compressed, w_avg_compressed, p_avg_compressed
        gc.collect()
