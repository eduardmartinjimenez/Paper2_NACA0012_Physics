import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import ConvexHull

def plot_projection_interface_points_and_stl(
    x_data, y_data, interface_indices_i, interface_indices_j, proj_points, triangles
):
    """
    Plot the interface points and their corresponding projection points on the x-y plane,
    and overlay the STL boundary computed with ConvexHull.

    Parameters:
        x_data (np.ndarray): 3D array containing the x-coordinates of the grid.
        y_data (np.ndarray): 3D array containing the y-coordinates of the grid.
        interface_indices_i (list): List of i indices of interface points.
        interface_indices_j (list): List of j indices of interface points.
        proj_points (np.ndarray): Array of projection points (N x 3), where each row is [x, y, z].

    Returns:
        None
    """
    # Extract x and y coordinates of interface points
    x_interface = [
        x_data[1, j, i] for i, j in zip(interface_indices_i, interface_indices_j)
    ]
    y_interface = [
        y_data[1, j, i] for i, j in zip(interface_indices_i, interface_indices_j)
    ]

    # Extract x and y coordinates of projection points, ignoring NaN values
    proj_x = proj_points[:, 0]
    proj_y = proj_points[:, 1]

    # Create the figure and axis
    fig, ax = plt.subplots(figsize=(8, 6))

    # Plot interface and projection points
    ax.scatter(x_interface, y_interface, color="red", s=10, label="Interface Points")
    ax.scatter(proj_x, proj_y, color="blue", s=10, label="Projection Points", alpha=0.6)

    # Draw lines connecting interface points to their projections
    for i in range(len(x_interface)):
        ax.plot(
            [x_interface[i], proj_x[i]],
            [y_interface[i], proj_y[i]],
            color="gray",
            linestyle="--",
            linewidth=0.8,
        )
    # Compute and Plot the STL Boundary
    # Extract all unique (x, y) points from the STL triangles
    all_points = triangles.reshape(-1, 3)[:, :2]  # Keep only the x, y coordinates
    unique_points = np.unique(all_points, axis=0)

    # Compute the convex hull to determine the outer boundary
    hull = ConvexHull(unique_points)

    # Plot each edge of the convex hull
    for i, simplex in enumerate(hull.simplices):
        # Only label the first segment for the legend
        label = "STL Boundary" if i == 0 else None
        ax.plot(
            unique_points[simplex, 0],
            unique_points[simplex, 1],
            "k-",
            linewidth=0.25,
            label=label,
        )
    # Labels and formatting
    ax.set_xlabel("X Coordinate")
    ax.set_ylabel("Y Coordinate")
    ax.set_title("Interface Points, Projection Points and STL Boundary")
    ax.legend()
    ax.grid(True)
    ax.axis("equal")  # Equal aspect ratio for better visualization
    plt.show()


def plot_projection_interface_points_stl_and_ref_points(
    x_data, y_data,
    interface_indices_i, interface_indices_j,
    proj_points, triangles,
    highlight_indices=None  # NEW: list of indices to highlight
):
    """
    Plot the interface points and their corresponding projection points on the x-y plane,
    and overlay the STL boundary. Optionally highlight selected reference points.

    Parameters:
        x_data, y_data: 3D coordinate arrays
        interface_indices_i, interface_indices_j: grid indices of interface points
        proj_points: (N, 3) array of projection points
        triangles: STL mesh triangles
        highlight_indices: List of integer indices (in proj_points) to highlight
    """
    # Extract interface coordinates
    x_interface = [x_data[1, j, i] for i, j in zip(interface_indices_i, interface_indices_j)]
    y_interface = [y_data[1, j, i] for i, j in zip(interface_indices_i, interface_indices_j)]

    # Extract projection coordinates
    proj_x = proj_points[:, 0]
    proj_y = proj_points[:, 1]

    # Create figure
    fig, ax = plt.subplots(figsize=(8, 6))

    # Plot all interface and projection points
    ax.scatter(x_interface, y_interface, color="red", s=10, label="Interface Points")
    ax.scatter(proj_x, proj_y, color="blue", s=10, alpha=0.6, label="Projection Points")

    # Draw connection lines
    for i in range(len(x_interface)):
        ax.plot(
            [x_interface[i], proj_x[i]],
            [y_interface[i], proj_y[i]],
            color="gray", linestyle="--", linewidth=0.8
        )

    # Highlight selected reference projection points
    if highlight_indices is not None:
        ax.scatter(
            proj_x[highlight_indices],
            proj_y[highlight_indices],
            color="green", s=50, marker="*", label="Reference Points", zorder=5
        )

    # STL boundary
    all_points = triangles.reshape(-1, 3)[:, :2]
    unique_points = np.unique(all_points, axis=0)
    hull = ConvexHull(unique_points)

    for i, simplex in enumerate(hull.simplices):
        label = "STL Boundary" if i == 0 else None
        ax.plot(
            unique_points[simplex, 0],
            unique_points[simplex, 1],
            "k-", linewidth=0.25, label=label
        )

    ax.set_xlabel("X Coordinate")
    ax.set_ylabel("Y Coordinate")
    ax.set_title("Interface Points, Projection Points, and STL Boundary")
    ax.legend()
    ax.grid(True)
    ax.axis("equal")
    plt.show()


def plot_projection_interface_points_stl_and_ref_points2(
    x_data, y_data,
    interface_indices_i, interface_indices_j,
    proj_points, triangles,
    highlight_indices=None,  # Optional: indices in proj_points to highlight
    correlation_indices_ij=None  # NEW: list of (i, j) tuples to show target grid points
):

    # Extract interface coordinates
    x_interface = [x_data[1, j, i] for i, j in zip(interface_indices_i, interface_indices_j)]
    y_interface = [y_data[1, j, i] for i, j in zip(interface_indices_i, interface_indices_j)]

    # Projection points
    proj_x = proj_points[:, 0]
    proj_y = proj_points[:, 1]

    fig, ax = plt.subplots(figsize=(8, 6))

    # Plot all interface and projection points
    ax.scatter(x_interface, y_interface, color="red", s=10, label="Interface Points")
    ax.scatter(proj_x, proj_y, color="blue", s=10, alpha=0.6, label="Projection Points")

    # Draw connection lines
    for i in range(len(x_interface)):
        ax.plot(
            [x_interface[i], proj_x[i]],
            [y_interface[i], proj_y[i]],
            color="gray", linestyle="--", linewidth=0.8
        )

    # Highlight selected projection points
    if highlight_indices is not None:
        ax.scatter(
            proj_x[highlight_indices],
            proj_y[highlight_indices],
            color="green", s=50, marker="*", label="Reference Proj. Points", zorder=5
        )

    # Highlight actual matched grid correlation points
    if correlation_indices_ij is not None:
        x_corr = [x_data[1, j, i] for i, j in correlation_indices_ij]
        y_corr = [y_data[1, j, i] for i, j in correlation_indices_ij]
        ax.scatter(
            x_corr, y_corr,
            color="magenta", s=50, marker="x", label="Matched Grid Points", zorder=6
        )

    # STL boundary
    all_points = triangles.reshape(-1, 3)[:, :2]
    unique_points = np.unique(all_points, axis=0)
    hull = ConvexHull(unique_points)

    for i, simplex in enumerate(hull.simplices):
        label = "STL Boundary" if i == 0 else None
        ax.plot(
            unique_points[simplex, 0],
            unique_points[simplex, 1],
            "k-", linewidth=0.25, label=label
        )

    ax.set_xlabel("X Coordinate")
    ax.set_ylabel("Y Coordinate")
    ax.set_title("Interface Points, Projection Points, and STL Boundary")
    ax.legend()
    ax.grid(True)
    ax.axis("equal")
    plt.show()
