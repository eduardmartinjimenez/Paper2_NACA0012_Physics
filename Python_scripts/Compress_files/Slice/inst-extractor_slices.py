import os, glob, re
import h5py
import numpy as np
import gc

# Define file path and name

FILE_PATH = "/gpfs/scratch/upc108/EDU/NACA_0012_AOA12_Re50000_1716x1662x128/slices_data/slice_test/compressed_slices/"
FILE_BASENAME = "slice_1"

# REDUCED_TOPOLOGY_PATH = os.path.join(INPUT_BASE_PATH, INPUT_BASENAME+"-reduced-topology.h5")
# INPUT_REDUCED_DATA_PATH = os.path.join(INPUT_BASE_PATH, INPUT_BASENAME+"_"+INPUT_REDUCED_DATA_INST+"-reduced-data.h5")


def read_instants():
    steps = []
    print(FILE_PATH)
    for file in glob.glob(FILE_PATH + f"{FILE_BASENAME}_*-COMP-DATA.h5"):
        steps.append(int(re.split("[_-]", file)[-3]))
    steps.sort()
    return steps


def extract_compressed_data(
    target: np.ndarray, values: np.ndarray, indices: np.ndarray
):
    """
    Sets values from a 1D array into a 3D target array at positions specified by indices.

    Parameters:
    - target (np.ndarray): 3D array where values will be set.
    - values (np.ndarray): 1D array of values to insert.
    - indices (np.ndarray): 2D array of shape (N, 3), containing the 3D indices where values will be set.

    Returns:
    - None (modifies target in-place)
    """
    if len(values) != len(indices):
        raise ValueError("Length of values and indices must be the same.")
    if indices.shape[1] != 3:
        raise ValueError("Indices must be of shape (N, 3) for 3D positions.")

    # Unpack indices to 3 separate arrays for advanced indexing
    x_idx, y_idx, z_idx = indices[:, 0], indices[:, 1], indices[:, 2]

    target[x_idx, y_idx, z_idx] = values


def save_xdmf(inst, data_shape):
    OUTPUT_FILE_PATH = os.path.join(FILE_PATH, f"{FILE_BASENAME}_{inst}-EXTR-DATA.xdmf")
    f = open(OUTPUT_FILE_PATH, "w")

    shape = f"{data_shape[0]} {data_shape[1]} {data_shape[2]}"
    data_source_file = f"{FILE_BASENAME}_{inst}-EXTR-DATA.h5"
    mesh_source_file = f"{FILE_BASENAME}-CROP-MESH.h5"

    f.write(f"<?xml version='{1.0}' ?>")
    f.write("\n")
    f.write(f"<!DOCTYPE Xdmf SYSTEM 'Xdmf.dtd' []>")
    f.write("\n")
    f.write(f"<Xdmf Version='{2.0}'>")
    f.write("\n")
    f.write(f"  <Domain>")
    f.write("\n")
    f.write(f"    <Grid Name='{FILE_BASENAME}-cropped-mesh' GridType='Uniform'>")
    f.write("\n")
    f.write(f"      <Topology TopologyType='3DSMesh' Dimensions='{shape}'/>")
    f.write("\n")
    f.write(f"      <Geometry GeometryType='X_Y_Z'>")
    f.write("\n")
    f.write(
        f"        <DataItem Name='x' Dimensions='{shape}' NumberType='Float' Precision='8' Format='HDF'>"
    )
    f.write("\n")
    f.write(f"            {mesh_source_file}:/x")
    f.write("\n")
    f.write(f"        </DataItem>")
    f.write("\n")
    f.write(
        f"        <DataItem Name='y' Dimensions='{shape}' NumberType='Float' Precision='8' Format='HDF'>"
    )
    f.write("\n")
    f.write(f"            {mesh_source_file}:/y")
    f.write("\n")
    f.write(f"        </DataItem>")
    f.write("\n")
    f.write(
        f"        <DataItem Name='z' Dimensions='{shape}' NumberType='Float' Precision='8' Format='HDF'>"
    )
    f.write("\n")
    f.write(f"            {mesh_source_file}:/z")
    f.write("\n")
    f.write(f"        </DataItem>")
    f.write("\n")
    f.write(f"      </Geometry>")
    f.write("\n")
    f.write(f"      <Attribute Name='tag_IBM' AttributeType='Scalar' Center='Node'>")
    f.write("\n")
    f.write(
        f"        <DataItem Dimensions='{shape}' NumberType='Float' Precision='8' Format='HDF'>"
    )
    f.write("\n")
    f.write(f"            {mesh_source_file}:/tag_IBM")
    f.write("\n")
    f.write(f"        </DataItem>")
    f.write("\n")
    f.write(f"      </Attribute>")
    f.write("\n")
    f.write(f"      <Attribute Name='u' AttributeType='Scalar' Center='Node'>")
    f.write("\n")
    f.write(
        f"        <DataItem Dimensions='{shape}' NumberType='Float' Precision='8' Format='HDF'>"
    )
    f.write("\n")
    f.write(f"            {data_source_file}:/u")
    f.write("\n")
    f.write(f"        </DataItem>")
    f.write("\n")
    f.write(f"      </Attribute>")
    f.write("\n")
    f.write(f"      <Attribute Name='v' AttributeType='Scalar' Center='Node'>")
    f.write("\n")
    f.write(
        f"        <DataItem Dimensions='{shape}' NumberType='Float' Precision='8' Format='HDF'>"
    )
    f.write("\n")
    f.write(f"            {data_source_file}:/v")
    f.write("\n")
    f.write(f"        </DataItem>")
    f.write("\n")
    f.write(f"      </Attribute>")
    f.write("\n")
    f.write(f"      <Attribute Name='w' AttributeType='Scalar' Center='Node'>")
    f.write("\n")
    f.write(
        f"        <DataItem Dimensions='{shape}' NumberType='Float' Precision='8' Format='HDF'>"
    )
    f.write("\n")
    f.write(f"            {data_source_file}:/w")
    f.write("\n")
    f.write(f"        </DataItem>")
    f.write("\n")
    f.write(f"      </Attribute>")
    f.write("\n")
    f.write(f"      <Attribute Name='P' AttributeType='Scalar' Center='Node'>")
    f.write("\n")
    f.write(
        f"        <DataItem Dimensions='{shape}' NumberType='Float' Precision='8' Format='HDF'>"
    )
    f.write("\n")
    f.write(f"            {data_source_file}:/p")
    f.write("\n")
    f.write(f"        </DataItem>")
    f.write("\n")
    f.write(f"      </Attribute>")
    f.write("\n")
    f.write(f"      <Attribute Name='avg_u' AttributeType='Scalar' Center='Node'>")
    f.write("\n")
    f.write(
        f"        <DataItem Dimensions='{shape}' NumberType='Float' Precision='8' Format='HDF'>"
    )
    f.write("\n")
    f.write(f"            {data_source_file}:/avg_u")
    f.write("\n")
    f.write(f"        </DataItem>")
    f.write("\n")
    f.write(f"      </Attribute>")
    f.write("\n")
    f.write(f"      <Attribute Name='avg_v' AttributeType='Scalar' Center='Node'>")
    f.write("\n")
    f.write(
        f"        <DataItem Dimensions='{shape}' NumberType='Float' Precision='8' Format='HDF'>"
    )
    f.write("\n")
    f.write(f"            {data_source_file}:/avg_v")
    f.write("\n")
    f.write(f"        </DataItem>")
    f.write("\n")
    f.write(f"      </Attribute>")
    f.write("\n")
    f.write(f"      <Attribute Name='avg_w' AttributeType='Scalar' Center='Node'>")
    f.write("\n")
    f.write(
        f"        <DataItem Dimensions='{shape}' NumberType='Float' Precision='8' Format='HDF'>"
    )
    f.write("\n")
    f.write(f"            {data_source_file}:/avg_w")
    f.write("\n")
    f.write(f"        </DataItem>")
    f.write("\n")
    f.write(f"      </Attribute>")
    f.write("\n")
    f.write(f"      <Attribute Name='avg_P' AttributeType='Scalar' Center='Node'>")
    f.write("\n")
    f.write(
        f"        <DataItem Dimensions='{shape}' NumberType='Float' Precision='8' Format='HDF'>"
    )
    f.write("\n")
    f.write(f"            {data_source_file}:/avg_P")
    f.write("\n")
    f.write(f"        </DataItem>")
    f.write("\n")
    f.write(f"      </Attribute>")
    f.write("\n")
    f.write(f"    </Grid>")
    f.write("\n")
    f.write(f"  </Domain>")
    f.write("\n")
    f.write(f"</Xdmf>")


    f.close()


if __name__ == "__main__":
    instants = read_instants()
    print(instants)

    print("File to extract:", instants)

    MESH_FILE_PATH = os.path.join(FILE_PATH, f"{FILE_BASENAME}-CROP-MESH.h5")

    if os.path.exists(MESH_FILE_PATH):
        print(f"Opening Mesh file {MESH_FILE_PATH}")
    else:
        print(f"Mesh file does not exist: {MESH_FILE_PATH}")

    mesh = h5py.File(MESH_FILE_PATH, "r")

    x = mesh["x"][:, :, :]
    y = mesh["y"][:, :, :]
    z = mesh["z"][:, :, :]
    tag_ibm = mesh["tag_IBM"][:, :, :]
    compressed_topo = mesh["compressed_topology"][:]

    del tag_ibm
    gc.collect()

    mesh.close()
    gc.collect()

    for inst in instants:
        INPUT_FILE_PATH = os.path.join(
            FILE_PATH, f"{FILE_BASENAME}_{inst}-COMP-DATA.h5"
        )

        if os.path.exists(INPUT_FILE_PATH):
            print(f"Opening file {INPUT_FILE_PATH}")
        else:
            print(f"File does not exist: {INPUT_FILE_PATH}")

        # Import data file
        data_file = h5py.File(INPUT_FILE_PATH, "r")

        u_compressed = data_file["u_compressed"][:]
        v_compressed = data_file["v_compressed"][:]
        w_compressed = data_file["w_compressed"][:]
        p_compressed = data_file["p_compressed"][:]

        data_file.close()
        del data_file
        gc.collect()

        u = np.full_like(x, np.nan)
        v = np.full_like(x, np.nan)
        w = np.full_like(x, np.nan)
        p = np.full_like(x, np.nan)

        # Extracting compressed data to expanded fields
        extract_compressed_data(u, u_compressed, compressed_topo)
        extract_compressed_data(v, v_compressed, compressed_topo)
        extract_compressed_data(w, w_compressed, compressed_topo)
        extract_compressed_data(p, p_compressed, compressed_topo)

        del u_compressed, v_compressed, w_compressed, p_compressed
        gc.collect()

        # Saving expanded fields
        OUTPUT_FILE_PATH = os.path.join(
            FILE_PATH, f"{FILE_BASENAME}_{inst}-EXTR-DATA.h5"
        )

        print("Saving file: ", OUTPUT_FILE_PATH)
        output_file = h5py.File(OUTPUT_FILE_PATH, "w")
        output_file.create_dataset("u", data=u, dtype="float32")
        output_file.create_dataset("v", data=v, dtype="float32")
        output_file.create_dataset("w", data=w, dtype="float32")
        output_file.create_dataset("p", data=p, dtype="float32")

        del u, v, w, p

        gc.collect()

        save_xdmf(inst, x.shape)
        output_file.close()
