import numpy as np
import matplotlib.pyplot as plt

# LaTeX style
plt.rc('text', usetex=True)
plt.rc('font', size=16, family='serif')
plt.rc('text.latex', preamble=r'\usepackage{amsmath} \usepackage{amssymb}')

def naca0012_coordinates(num_points=1000):
    """
    Generate NACA0012 airfoil coordinates.
    NACA0012: symmetric airfoil, 12% thickness
    """
    # Thickness distribution for NACA airfoil
    c = 1.0  # chord length
    t = 0.12  # thickness ratio

    x = np.linspace(0, 1, num_points)

    # Thickness distribution
    yt = 5 * t * (
        0.2969 * np.sqrt(x) -
        0.1260 * x -
        0.3516 * x**2 +
        0.2843 * x**3 -
        0.1015 * x**4
    )

    # For symmetric airfoil, combine upper and lower surfaces
    x_upper = x
    y_upper = yt

    x_lower = x
    y_lower = -yt

    # Combine for closed airfoil
    x_airfoil = np.concatenate([x_upper, x_lower[::-1]])
    y_airfoil = np.concatenate([y_upper, y_lower[::-1]])

    return x_airfoil, y_airfoil

# Define probe locations
probes = {
    'P0': (0.5, 0.075),
    'P1': (0.5, 0.14),
    'P2': (0.7, 0.06),
    'P3': (0.7, 0.09),
    'P4': (0.7, 0.2),
    'P5': (0.9, 0.03),
    'P6': (0.9, 0.06),
    'P7': (0.9, 0.09),
    'P8': (0.9, 0.18),
    'P9': (0.9, 0.27)
}

# Generate NACA0012 coordinates
x_airfoil, y_airfoil = naca0012_coordinates()

# Create figure
fig, ax = plt.subplots(figsize=(10, 5))

# Plot airfoil surface
ax.plot(x_airfoil, y_airfoil, 'k-', linewidth=2, label='NACA0012')

# Plot probe locations
for probe_name, (x, y) in probes.items():
    ax.plot(x, y, 'ko', markersize=6)
    # Offset label slightly above the point
    ax.text(x + 0.02, y, probe_name, fontsize=10, ha='center')

# Set axis properties
ax.set_xlim(-0.05, 1.05)
ax.set_ylim(-0.07, 0.3)
ax.set_xlabel('x/c', fontsize=12)
ax.set_ylabel('y/c', fontsize=12)
ax.set_aspect('equal')
ax.axhline(y=0, color='k', linewidth=0.5)
ax.axvline(x=0, color='k', linewidth=0.5)

plt.tight_layout()
plt.savefig('/home/jofre/Members/Eduard/Paper2/Figures/probe_locations.png', dpi=300, bbox_inches='tight')
plt.savefig('/home/jofre/Members/Eduard/Paper2/Figures/probe_locations.eps', dpi=300, bbox_inches='tight')
plt.show()

print("Visualization saved to probe_locations.png")
