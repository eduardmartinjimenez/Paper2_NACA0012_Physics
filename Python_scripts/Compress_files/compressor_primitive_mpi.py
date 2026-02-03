import os, glob, re
import h5py
import numpy as np
import gc
from mpi4py import MPI
from compress_tools import (crop_3d_by_physical_coords, compress_3d_array)

# Initialize MPI
mpi_comm = MPI.COMM_WORLD # Get the global communicator
mpi_rank = mpi_comm.Get_rank() # Get the rank of the current process
mpi_size = mpi_comm.Get_size() # Get the total number of processes

# Print MPI info
if mpi_rank == 0:
    print("=" * 60)
    print(f"MPI Configuration:")
    print(f"  Total processes: {mpi_size}")
    print("=" * 60)

# Define input file path and base name
INPUT_FILE_PATH = "/home/jofre/Members/Eduard/Simulations/NACA_0012_AOA85_Re10000_502x443x64/batch_5/"
INPUT_FILE_BASENAME = "3d_ibm_stl_naca0012_aoa85_Re10000"

# Define output folder path
OUTPUT_FILE_PATH = "/home/jofre/Members/Eduard/Simulations/NACA_0012_AOA85_Re10000_502x443x64/batch_5/compressed_snapshots"
# Rename ONLY the output files. If None, it uses INPUT_FILE_BASENAME.
OUTPUT_FILE_BASENAME = "3d_NACA0012_Re10000_AoA85"

# Define crop region:
X_MIN = -0.5
Y_MIN = -0.5
Z_MIN = -1.0

X_MAX = 8.0
Y_MAX = 1.5
Z_MAX = 1.0

def get_files_by_extension(folder_path, extensions):
    """
    Returns a list of file paths in the given folder matching the specified extensions.
    """
    if isinstance(extensions, str):
        extensions = (extensions,)
    else:
        extensions = tuple(extensions)

    file_paths = []
    for file in os.listdir(folder_path):
        full_path = os.path.join(folder_path, file)
        if os.path.isfile(full_path) and file.lower().endswith(
            tuple(f".{ext.lower()}" for ext in extensions)
        ):
            file_paths.append(full_path)

    return file_paths

def distribute_file_list(folder_path, extensions, comm):
    """
    Root rank scans 'folder_path' for files with given extensions, then distributes
    the list in (almost) equal chunks to all ranks.

    Returns:
        local_files (list of str): List of file paths assigned to this rank.
    """
    rank = comm.Get_rank()
    size = comm.Get_size()

    if rank == 0: # Root rank
        file_list = get_files_by_extension(folder_path, extensions) # Get all files
        # Sort for deterministic ordering
        file_list.sort()
        N = len(file_list)
        print(f"[Rank 0] Found {N} files in {folder_path}")

        # Base number of files per process
        base = N // size
        # remainder distributed one by one to the first ranks
        rem = N % size

        # Compute chunk sizes
        counts = [base + (1 if r < rem else 0) for r in range(size)]

        # Compute displacements (start indices)
        displs = [sum(counts[:r]) for r in range(size)]
    else:
        file_list = None
        counts = None
        displs = None

    # Broadcast counts and displacements to all ranks
    counts = comm.bcast(counts, root=0)
    displs = comm.bcast(displs, root=0)

    # Now actually send the file-list slices
    if rank == 0:
        local_files = file_list[displs[0] : displs[0] + counts[0]]
        for r in range(1, size):
            start = displs[r]
            end = start + counts[r]
            comm.send(file_list[start:end], dest=r, tag=77)
    else:
        local_files = comm.recv(source=0, tag=77)
    
    snapshot_numbers = [int(re.split("[_ .]", os.path.basename(f))[-2]) for f in local_files]
    print(f"[Rank {rank}] Received {len(local_files)} files, Snapshot numbers: {snapshot_numbers}")
    return local_files 


if __name__ == "__main__":

    # Ensure output directory exists 
    if mpi_rank == 0:
        os.makedirs(OUTPUT_FILE_PATH, exist_ok=True)
    mpi_comm.Barrier()

    OUT_BASE = OUTPUT_FILE_BASENAME if OUTPUT_FILE_BASENAME else INPUT_FILE_BASENAME

    # Each rank gets its share of .h5 files from the input path
    local_files = distribute_file_list(INPUT_FILE_PATH, "h5", mpi_comm)

    min_coords = (X_MIN, Y_MIN, Z_MIN)
    max_coords = (X_MAX, Y_MAX, Z_MAX)

    for input_full_path in local_files:
        # Extract the instant from the filename
        fname = os.path.basename(input_full_path)

        inst = int(re.split("[_ .]", fname)[-2]) #

        # Check whether we've already compressed this instant
        output_data_path = os.path.join(
            OUTPUT_FILE_PATH, f"{OUT_BASE}_{inst}-COMP-DATA.h5"
        )
        if os.path.exists(output_data_path):
            print(f"[Rank {mpi_rank}] ► Snapshot {inst} already compressed, skipping.")
            continue

        if os.path.exists(input_full_path):
            print(f"[Rank {mpi_rank}] Reading file: {input_full_path}")
        else:
            print(f"[Rank {mpi_rank}] Data file does not exist: {input_full_path}")
            continue

        # Import data file
        data_file = h5py.File(input_full_path, "r")

        # Import 3D coordinates
        x_data = data_file["x"][:, :, :]
        y_data = data_file["y"][:, :, :]
        z_data = data_file["z"][:, :, :]

        coords_array = np.stack((x_data, y_data, z_data), axis=-1)  # (D, H, W, 3)

        # Import tag_IBM data and crop
        tag_ibm_data = data_file["tag_IBM"][:, :, :]
        tag_ibm_cropped = crop_3d_by_physical_coords(
            tag_ibm_data, coords_array, min_coords, max_coords
        )
        del tag_ibm_data
        gc.collect()

        # Import u, v, w, p
        u_data = data_file["u"][:, :, :]
        v_data = data_file["v"][:, :, :]
        w_data = data_file["w"][:, :, :]
        p_data = data_file["P"][:, :, :]

        # Crop fields
        u_cropped = crop_3d_by_physical_coords(u_data, coords_array, min_coords, max_coords)
        v_cropped = crop_3d_by_physical_coords(v_data, coords_array, min_coords, max_coords)
        w_cropped = crop_3d_by_physical_coords(w_data, coords_array, min_coords, max_coords)
        p_cropped = crop_3d_by_physical_coords(p_data, coords_array, min_coords, max_coords)

        del u_data, v_data, w_data, p_data, coords_array
        gc.collect()

        # Mask for solid points
        mask = tag_ibm_cropped > 0.9999

        # Compress fields
        u_compressed, compressed_topo = compress_3d_array(u_cropped, mask)
        del u_cropped, tag_ibm_cropped
        gc.collect()

        v_compressed, _ = compress_3d_array(v_cropped, mask)
        del v_cropped
        gc.collect()

        w_compressed, _ = compress_3d_array(w_cropped, mask)
        del w_cropped
        gc.collect()

        p_compressed, _ = compress_3d_array(p_cropped, mask)
        del p_cropped, mask, compressed_topo
        gc.collect()

        # Saving compressed data.
        print(f"[Rank {mpi_rank}] Saving file: {output_data_path}")
        with h5py.File(output_data_path, "w") as output_file:
            output_file.create_dataset("u_compressed", data=u_compressed, dtype="float32")
            output_file.create_dataset("v_compressed", data=v_compressed, dtype="float32")
            output_file.create_dataset("w_compressed", data=w_compressed, dtype="float32")
            output_file.create_dataset("p_compressed", data=p_compressed, dtype="float32")

        del u_compressed, v_compressed, w_compressed, p_compressed
        gc.collect()
        data_file.close()

    # Synchronize all processes
    mpi_comm.Barrier()
    if mpi_rank == 0:
        print("All snapshots compressed successfully!")