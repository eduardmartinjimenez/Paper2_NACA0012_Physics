import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


def load_lift_data(filepath):
    """
    Load lift coefficient data from CSV.

    Parameters:
        filepath (str): Full path to the CSV file.

    Returns:
        t (np.ndarray): Time array.
        Cl (np.ndarray): Lift coefficient array.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Data file does not exist: {filepath}")

    df = pd.read_csv(filepath)
    df.columns = df.columns.str.strip()
    return df["# time"].to_numpy(), df["C_l"].to_numpy()


# AOA 5


def plot_lift_vs_time_aoa5(t, Cl):
    """Plot instantaneous lift coefficient vs. time."""
    plt.figure(figsize=(10, 5))
    plt.plot(t, Cl, label="Cl (lift coeff)", color="b")
    plt.xlabel("Time [s]", fontsize=12)
    plt.ylabel("Lift coefficient", fontsize=12)
    plt.title("Lift coefficient vs. Time", fontsize=14)
    # Horizontal line at cl = 0.57
    plt.axhline(y=0.57, color="green", linestyle="--", linewidth=1.5)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def plot_lift_with_mean_aoa5(t, Cl, t_threshold=4.0):
    """Plot lift coefficient with mean after threshold time."""
    Cl_mean = Cl[t > t_threshold].mean()

    plt.figure(figsize=(10, 5))
    plt.plot(t, Cl, label="Cl (lift coeff)", color="b")
    plt.axhline(
        Cl_mean,
        color="r",
        linestyle="--",
        linewidth=1.5,
        label=rf"$\overline{{C_l}}$ (t > {t_threshold})",
    )
    # Horizontal line at cl = 0.57
    plt.axhline(y=0.57, color="green", linestyle="--", linewidth=1.5)
    plt.text(
        t[-1],
        Cl_mean,
        f"{Cl_mean:.3f}",
        color="r",
        fontsize=10,
        verticalalignment="bottom",
        horizontalalignment="right",
    )
    plt.xlabel("Time [s]", fontsize=12)
    plt.ylabel("Lift coefficient", fontsize=12)
    plt.title("Lift coefficient vs. Time with Mean", fontsize=14)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def plot_lift_and_accumulated_mean_aoa5(t, Cl, t_start=4.0):
    """Plot instantaneous and accumulated mean lift coefficient from t_start."""
    mask = t >= t_start
    t_steady = t[mask]
    Cl_steady = Cl[mask]

    Cl_cumsum = np.cumsum(Cl_steady)
    Cl_mean_accumulated = Cl_cumsum / np.arange(1, len(Cl_steady) + 1)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    ax1.plot(t_steady, Cl_steady, color="blue", label="Instantaneous $C_l$")
    ax1.set_ylabel("Lift Coefficient $C_l$", fontsize=12)
    ax1.set_title(f"Instantaneous Lift Coefficient from t = {t_start} s", fontsize=14)
    # Horizontal line at cl = 0.57
    ax1.axhline(y=0.57, color="green", linestyle="--", linewidth=1.5)
    ax1.grid(True)
    ax1.legend()

    ax2.plot(
        t_steady, Cl_mean_accumulated, color="green", label="Accumulated Mean $C_l$"
    )
    ax2.set_xlabel("Time [s]", fontsize=12)
    ax2.set_ylabel("Accumulated Mean $C_l$", fontsize=12)
    # Horizontal line at cl = 0.57
    ax2.axhline(y=0.57, color="green", linestyle="--", linewidth=1.5)
    ax2.set_title(
        f"Accumulated Mean Lift Coefficient from t = {t_start} s", fontsize=14
    )
    ax2.grid(True)
    ax2.legend()

    plt.tight_layout()
    plt.show()


# AOA 7.5

def plot_lift_vs_time_aoa75(t, Cl):
    """Plot instantaneous lift coefficient vs. time."""
    plt.figure(figsize=(10, 5))
    plt.plot(t, Cl, label="Cl (lift coeff)", color="b")
    plt.xlabel("Time [s]", fontsize=12)
    plt.ylabel("Lift coefficient", fontsize=12)
    plt.title("Lift coefficient vs. Time", fontsize=14)
    # Horizontal line at cl = 0.74
    plt.axhline(y=0.74, color="green", linestyle="--", linewidth=1.5)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def plot_lift_with_mean_aoa75(t, Cl, t_threshold=4.0):
    """Plot lift coefficient with mean after threshold time."""
    Cl_mean = Cl[t > t_threshold].mean()

    plt.figure(figsize=(10, 5))
    plt.plot(t, Cl, label="Cl (lift coeff)", color="b")
    plt.axhline(
        Cl_mean,
        color="r",
        linestyle="--",
        linewidth=1.5,
        label=rf"$\overline{{C_l}}$ (t > {t_threshold})",
    )
    # Horizontal line at cl = 0.74
    plt.axhline(y=0.74, color="green", linestyle="--", linewidth=1.5)
    plt.text(
        t[-1],
        Cl_mean,
        f"{Cl_mean:.3f}",
        color="r",
        fontsize=10,
        verticalalignment="bottom",
        horizontalalignment="right",
    )
    plt.xlabel("Time [s]", fontsize=12)
    plt.ylabel("Lift coefficient", fontsize=12)
    plt.title("Lift coefficient vs. Time with Mean", fontsize=14)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def plot_lift_and_accumulated_mean_aoa75(t, Cl, t_start=4.0):
    """Plot instantaneous and accumulated mean lift coefficient from t_start."""
    mask = t >= t_start
    t_steady = t[mask]
    Cl_steady = Cl[mask]

    Cl_cumsum = np.cumsum(Cl_steady)
    Cl_mean_accumulated = Cl_cumsum / np.arange(1, len(Cl_steady) + 1)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    ax1.plot(t_steady, Cl_steady, color="blue", label="Instantaneous $C_l$")
    ax1.set_ylabel("Lift Coefficient $C_l$", fontsize=12)
    ax1.set_title(f"Instantaneous Lift Coefficient from t = {t_start} s", fontsize=14)
    # Horizontal line at cl = 0.74
    ax1.axhline(y=0.74, color="green", linestyle="--", linewidth=1.5)
    ax1.grid(True)
    ax1.legend()

    ax2.plot(
        t_steady, Cl_mean_accumulated, color="green", label="Accumulated Mean $C_l$"
    )
    ax2.set_xlabel("Time [s]", fontsize=12)
    ax2.set_ylabel("Accumulated Mean $C_l$", fontsize=12)
    # Horizontal line at cl = 0.74
    ax2.axhline(y=0.74, color="green", linestyle="--", linewidth=1.5)
    ax2.set_title(
        f"Accumulated Mean Lift Coefficient from t = {t_start} s", fontsize=14
    )
    ax2.grid(True)
    ax2.legend()

    plt.tight_layout()
    plt.show()


# AOA 9.25


def plot_lift_vs_time_aoa925(t, Cl):
    """Plot instantaneous lift coefficient vs. time."""
    plt.figure(figsize=(10, 5))
    plt.plot(t, Cl, label="Cl (lift coeff)", color="b")
    plt.xlabel("Time [s]", fontsize=12)
    plt.ylabel("Lift coefficient", fontsize=12)
    plt.title("Lift coefficient vs. Time", fontsize=14)
    # Horizontal line at cl = 0.8
    plt.axhline(y=0.8, color="green", linestyle="--", linewidth=1.5)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def plot_lift_with_mean_aoa925(t, Cl, t_threshold=4.0):
    """Plot lift coefficient with mean after threshold time."""
    Cl_mean = Cl[t > t_threshold].mean()

    plt.figure(figsize=(10, 5))
    plt.plot(t, Cl, label="Cl (lift coeff)", color="b")
    plt.axhline(
        Cl_mean,
        color="r",
        linestyle="--",
        linewidth=1.5,
        label=rf"$\overline{{C_l}}$ (t > {t_threshold})",
    )
    # Horizontal line at cl = 0.8
    plt.axhline(y=0.8, color="green", linestyle="--", linewidth=1.5)
    plt.text(
        t[-1],
        Cl_mean,
        f"{Cl_mean:.3f}",
        color="r",
        fontsize=10,
        verticalalignment="bottom",
        horizontalalignment="right",
    )
    plt.xlabel("Time [s]", fontsize=12)
    plt.ylabel("Lift coefficient", fontsize=12)
    plt.title("Lift coefficient vs. Time with Mean", fontsize=14)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def plot_lift_and_accumulated_mean_aoa925(t, Cl, t_start=4.0):
    """Plot instantaneous and accumulated mean lift coefficient from t_start."""
    mask = t >= t_start
    t_steady = t[mask]
    Cl_steady = Cl[mask]

    Cl_cumsum = np.cumsum(Cl_steady)
    Cl_mean_accumulated = Cl_cumsum / np.arange(1, len(Cl_steady) + 1)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    ax1.plot(t_steady, Cl_steady, color="blue", label="Instantaneous $C_l$")
    ax1.set_ylabel("Lift Coefficient $C_l$", fontsize=12)
    ax1.set_title(f"Instantaneous Lift Coefficient from t = {t_start} s", fontsize=14)
    # Horizontal line at cl = 0.81
    ax1.axhline(y=0.81, color="green", linestyle="--", linewidth=1.5)
    ax1.grid(True)
    ax1.legend()

    ax2.plot(
        t_steady, Cl_mean_accumulated, color="green", label="Accumulated Mean $C_l$"
    )
    ax2.set_xlabel("Time [s]", fontsize=12)
    ax2.set_ylabel("Accumulated Mean $C_l$", fontsize=12)
    # Horizontal line at cl = 0.8
    ax2.axhline(y=0.8, color="green", linestyle="--", linewidth=1.5)
    ax2.set_title(
        f"Accumulated Mean Lift Coefficient from t = {t_start} s", fontsize=14
    )
    ax2.grid(True)
    ax2.legend()

    plt.tight_layout()
    plt.show()


# AOA 12


def plot_lift_vs_time_aoa12(t, Cl):
    """Plot instantaneous lift coefficient vs. time."""
    plt.figure(figsize=(10, 5))
    plt.plot(t, Cl, label="Cl (lift coeff)", color="b")
    plt.xlabel("Time [s]", fontsize=12)
    plt.ylabel("Lift coefficient", fontsize=12)
    plt.title("Lift coefficient vs. Time", fontsize=14)
    # Horizontal line at cl = 0.7
    plt.axhline(y=0.7, color="green", linestyle="--", linewidth=1.5)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def plot_lift_with_mean_aoa12(t, Cl, t_threshold=4.0):
    """Plot lift coefficient with mean after threshold time."""
    Cl_mean = Cl[t > t_threshold].mean()

    plt.figure(figsize=(10, 5))
    plt.plot(t, Cl, label="Cl (lift coeff)", color="b")
    plt.axhline(
        Cl_mean,
        color="r",
        linestyle="--",
        linewidth=1.5,
        label=rf"$\overline{{C_l}}$ (t > {t_threshold})",
    )
    # Horizontal line at cl = 0.7
    #plt.axhline(y=0.70, color="green", linestyle="--", linewidth=1.5)
    plt.text(
        t[-1],
        Cl_mean,
        f"{Cl_mean:.3f}",
        color="r",
        fontsize=10,
        verticalalignment="bottom",
        horizontalalignment="right",
    )
    plt.xlabel("Time [s]", fontsize=12)
    plt.ylabel("Lift coefficient", fontsize=12)
    plt.title("Lift coefficient vs. Time with Mean", fontsize=14)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def plot_lift_and_accumulated_mean_aoa12(t, Cl, t_start=4.0):
    """Plot instantaneous and accumulated mean lift coefficient from t_start."""
    mask = t >= t_start
    t_steady = t[mask]
    Cl_steady = Cl[mask]

    Cl_cumsum = np.cumsum(Cl_steady)
    Cl_mean_accumulated = Cl_cumsum / np.arange(1, len(Cl_steady) + 1)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    ax1.plot(t_steady, Cl_steady, color="blue", label="Instantaneous $C_l$")
    ax1.set_ylabel("Lift Coefficient $C_l$", fontsize=12)
    ax1.set_title(f"Instantaneous Lift Coefficient from t = {t_start} s", fontsize=14)
    # Horizontal line at cl = 0.7
    ax1.axhline(y=0.70, color="green", linestyle="--", linewidth=1.5)
    ax1.grid(True)
    ax1.legend()

    ax2.plot(
        t_steady, Cl_mean_accumulated, color="green", label="Accumulated Mean $C_l$"
    )
    ax2.set_xlabel("Time [s]", fontsize=12)
    ax2.set_ylabel("Accumulated Mean $C_l$", fontsize=12)
    # Horizontal line at cl = 0.7
    ax2.axhline(y=0.7, color="green", linestyle="--", linewidth=1.5)
    ax2.set_title(
        f"Accumulated Mean Lift Coefficient from t = {t_start} s", fontsize=14
    )
    ax2.grid(True)
    ax2.legend()

    plt.tight_layout()
    plt.show()
