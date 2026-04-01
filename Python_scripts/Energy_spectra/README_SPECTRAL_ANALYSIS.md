# Energy Spectra Script - Complete Guide

## What is This Script Doing?

This script computes **energy spectral densities** of velocity fluctuations from time-series data extracted at specific locations in the flow around an airfoil. It analyzes how the kinetic energy of turbulent velocity fluctuations is distributed across different frequencies.

---

## Key Concepts (Explained Simply)

### 1. **Spectral Analysis - The Big Picture**

Imagine you have a time-series signal of velocity measurements:

```
Time:     0    1    2    3    4    5    6    ...
Velocity: 1.1  1.3  0.9  1.2  1.0  1.4  0.8  ...
```

This signal contains fluctuations at many different time scales (fast wiggles and slow wiggles mixed together). **Spectral analysis** separates these into individual frequency components, just like a prism separates white light into rainbow colors.

### 2. **Periodogram - Energy at Each Frequency**

A **periodogram** is a function `E(f)` that tells you: *"How much energy is in the velocity fluctuations at frequency f?"*

- Low frequency (f → 0): slow variations over time
- High frequency: rapid wiggles
- The area under the curve = total energy (variance)

### 3. **Fast Fourier Transform (FFT / RFFT)**

The **FFT** is the mathematical tool we use to convert from time-domain to frequency-domain:

```
Time series u(t) ──[FFT]──→ Spectrum E(f)
```

We use `rfft` (real FFT) because our velocity data is real-valued, and `rfft` is more efficient than regular `fft`.

### 4. **Variance and Spectrum - Connected**

Variance (RMS² of fluctuations) and spectral energy are two ways of measuring the same thing:

```
var(u) = integral over all frequencies of E(f) df
```

If you measure variance in time-domain and compute it from the spectrum, they should match (this script verifies this!).

---

## What This Script Actually Does

### **Step 1: Load Data**
- Loads the airfoil geometry (surface points)
- Loads slice mesh at a specific x-location
- Verifies the mesh is a valid 2D x-plane cross-section

### **Step 2: Select Probe Locations**
- You specify target y-coordinates where you want to measure velocity
- The script finds the closest grid points to those targets
- These become your "virtual measurement points" (probes)

### **Step 3: Extract Time Series**
- Loads all snapshot files (in time order)
- At each probe location, extracts velocity u(t) and v(t) at each time step
- Keeps ALL spanwise (z-direction) data, not just one line
- Result: Time series of shape `(n_times, n_z)` for each probe

### **Step 4: Rotate Coordinates**
- Converts velocity from grid coordinates (x,y) to **freestream-aligned** coordinates:
  - **u_stream**: velocity along the incoming flow direction
  - **v_cross**: velocity perpendicular to flow direction
- This is needed because the airfoil is at angle of attack (AOA)

### **Step 5: Compute Energy Spectra**
For each probe and each spanwise position (z):

1. Remove the temporal mean: `u'(t) = u(t) - mean(u)`
2. Apply FFT: `U_fft = rfft(u')`
3. Compute one-sided periodogram: `E(f) = (2 dt / N) |U_fft|²`
4. Correct DC and Nyquist components (counted only once)

### **Step 6: Average Over Spanwise Direction**
- For each frequency, average the spectrum across all z-positions
- Result: Representative spectrum at that probe location

### **Step 7: Validate Results**
- Compare time-domain variance with spectral variance
- If they don't match, something is wrong!
- Compute relative errors and warn if > 5%

### **Step 8: Create Visualizations**
- Probe location plot: shows where you're measuring
- Spectral plots: log-log plots of E(f) vs nondimensional frequency f*

### **Step 9: Save Results**
- HDF5 file with all spectra, metadata, and validation metrics

---

## Physical Interpretation - What Do the Results Mean?

### Example Spectrum

```
Energy
  |     ***
  |    * * *
  |   *     *
  |  *       *
  | *         *    Low frequency: large structures (slower)
  |*           *   High frequency: small structures (faster)
  |_________________________
     f_low    f_mid    f_high
```

**Reading the plot:**
- **Peak at low f**: Energy concentrated in large-scale turbulent structures
- **Tail at high f**: Energy in small-scale turbulence, decays rapidly (usually ~ f^-5/3)
- **Steep drop-off**: Indicates viscous damping (viscosity kills fast oscillations)

### Nondimensional Frequency f*

The x-axis uses "convective frequency":

```
f* = f × (chord length) / (freestream velocity)
```

This makes the spectrum independent of your specific simulation parameters and comparable to other studies.

### Two Components: E_uu vs E_vv

The script computes two spectra:
- **E_uu (streamwise)**: spectrum of velocity fluctuations along freestream direction
- **E_vv (cross-stream)**: spectrum of velocity fluctuations perpendicular to freestream

In turbulence, these are usually different because the flow is anisotropic (directional).

---

## Configuration - What You Need to Set

### Required Settings

```python
# 1. Slice location
SLICES_PATH = "path/to/slice_9/"      # Which slice to analyze

# 2. Probe positions
Y_LOCATIONS = [0.1, 0.15, 0.25]       # y-coordinates where you measure

# 3. Physical parameters
AOA_deg = 12.0                         # Angle of attack (degrees)
dt_iteration = 2.0e-06                # Time per iteration (seconds)

# 4. Reference parameters (from simulation)
u_infty = 1.0                          # Freestream velocity
c = 1.0                                # Chord length
Re_c = 50000                           # Reynolds number
```

### How to Adapt for Different Slices

The script automatically infers `slice_id` from the path:
```python
SLICES_PATH = "/path/to/slice_1/"     # → slice_id = "slice_1"
SLICES_PATH = "/path/to/slice_9/"     # → slice_id = "slice_9"
```

All output files are named with the slice_id, so you can run multiple slices without conflicts!

---

## Understanding the Outputs

### **File 1: airfoil_probe_locations_slice_9.png**

Shows where you're measuring:
- **Blue/Red points**: Airfoil surface
- **Green dashed line**: Slice plane (x-location)
- **Diamond markers (◇)**: Your requested y-coordinates
- **Circle markers (○)**: Actual grid points selected
- **Dashed lines**: Connect requested to actual (if different)

### **File 2: energy_spectra_uv_slice_9.png**

Two log-log plots:

**Top (E_uu)**: Streamwise energy spectrum
```
|  ____
|  \     \   ← Peak energy at medium frequencies
|   \     \___
|    \        \__  ← Drops off at high frequencies
└──────────────────
  f*
```

**Bottom (E_vv)**: Cross-stream energy spectrum
- Similar to E_uu but usually smaller amplitude

**Offset stacking**: Curves are shifted vertically for visibility (actual data unchanged)

### **File 3: energy_spectra_data_slice_9.h5**

HDF5 database containing:

```
Global Attributes:
├─ slice_id: "slice_9"
├─ AOA_deg: 12.0
├─ dt_save: 1.6e-03 (time between snapshots)
├─ n_samples: 21898 (number of snapshots)
├─ n_z: 128 (spanwise resolution)
└─ ...

Probe Groups:
├─ probe_00/
│  ├─ frequencies: [0, df, 2df, ...]      ← Dimensional frequencies (Hz)
│  ├─ f_star: [0, df*, 2df*, ...]         ← Nondimensional frequencies
│  ├─ E_uu: [E(f=0), E(f=df), ...]        ← Streamwise spectrum (z-averaged)
│  ├─ E_vv: [E(f=0), E(f=df), ...]        ← Cross-stream spectrum (z-averaged)
│  ├─ E_uu_z: (n_freqs, 128)              ← Full z-resolved spectrum
│  ├─ rel_error_u_percent: 0.5            ← Validation error (%)
│  └─ ...
├─ probe_01/
│  └─ ...
└─ probe_02/
   └─ ...
```

**Key metadata per probe:**
```
y_target: 0.1                    # What you requested
y_actual: 0.099996               # What the grid has
y_distance_error: 0.000004       # Mismatch
var_u_time: 0.00093             # Variance (time domain)
var_u_spectral: 0.000931        # Variance (from spectrum)
rel_error_u_percent: 0.23       # Should be < 5%
```

---

## Typical Workflow

### **First Time: Explore One Slice**
```python
Y_LOCATIONS = [0.05, 0.1, 0.15, 0.2]    # Multiple probe heights
SLICES_PATH = "/path/to/slice_9/"

# Run script → examine PNG visualizations
# Read HDF5 to post-process results
```

### **Systematic Study: Multiple Slices**
```python
for slice_num in [1, 3, 5, 7, 9]:
    SLICES_PATH = f"/path/to/slice_{slice_num}/"
    # Run script → automatically generates slice_{slice_num}_data.h5
```

### **Publication-Ready: Pick Best Heights**
```python
# Based on initial exploration, select specific y-locations
Y_LOCATIONS = [0.08, 0.12, 0.18]        # Refined selection

# Re-run with final configuration
```

---

## Validation - How Do You Know Results Are Correct?

### Check 1: Variance Match
Look for warning messages:
```
Probe 0 (y=0.1):
  u: var_time=9.34e-04, var_spectral=9.02e-04, rel_error=0.34%
  v: var_time=4.86e-04, var_spectral=4.87e-04, rel_error=0.21%
```

✅ **Good**: rel_error < 5%
❌ **Bad**: rel_error > 10% (suggests numerical issues)

### Check 2: Probe Locations
Open the PNG and verify:
- All probe circles (○) are in the fluid region (above the airfoil)
- Diamonds (◇) and circles overlap (or are very close)
- No probes inside the airfoil!

### Check 3: Spectrum Shape
- Should decay at high frequencies (usually ~f^-5/3 slope in inertial range)
- Should have a peak somewhere in the middle (characteristic flow frequency)
- Should NOT have noise/spikes (indicates aliasing or sampling problem)

---

## Common Questions

### Q1: Why are there two spectrum files (E_uu and E_vv)?

**A:** The flow is anisotropic. Streamwise fluctuations (along flow) usually have more energy than perpendicular fluctuations because the mean flow constrains perpendicular motion.

### Q2: What does "z-averaged" mean?

**A:** We compute the spectrum independently at each spanwise position (z), then average across all z. This gives a representative spectrum for that probe location.

### Q3: Why use nondimensional frequency f*?

**A:** Makes results comparable across different simulations:
- Different chord lengths → different physical frequencies
- Different velocities → different physical frequencies
- f* removes these effects and makes spectra from different cases comparable

### Q4: What if my variance error is > 5%?

**A:** Usually means:
- Numerical FFT errors (rare for large n_samples)
- Missing data (files didn't load completely)
- Windowing/edge effects
- Check the warning message for probe location

### Q5: Can I compare my spectra to published data?

**A:** Yes! Plot your E_uu vs f* and compare to literature. Make sure to note:
- Reynolds number
- AOA
- Airfoil type
- Measurement method

---

## Advanced: Understanding the Code

### Time Step Computation
```
Physical time between snapshots:
dt_save = (delta_iter) × (dt_iteration)
         = 800 iterations × 2e-6 s/iteration
         = 1.6e-3 seconds
```

### Frequency Resolution
```
Smallest resolvable frequency:
df = fs / n_samples = (1/dt_save) / n_samples

Nyquist frequency (max frequency):
f_nyquist = fs/2 = 1/(2*dt_save)
```

### One-Sided Spectrum Normalization
```
E(f) = (2 × dt_save / n_samples) × |FFT(u)|²

Except at f=0 (DC) and f=Nyquist (divide by 2)
This ensures: integral(E*df) ≈ var(u)
```

---

## Next Steps

1. **Run the script** on your data
2. **Examine the PNG plots** - do they look reasonable?
3. **Read the HDF5 file** with Python and plot custom comparisons
4. **Compare results** to similar studies in literature
5. **Vary probe locations** to understand spatial variation

---

## References & Further Reading

- **FFT fundamentals**: Oppenheim & Schafer, "Discrete-Time Signal Processing"
- **Turbulent spectra**: Pope, "Turbulent Flows" (Chapter on energy spectra)
- **Wall turbulence**: Smits, Marusic & Hutchins, "Annual Review of Fluid Mechanics" v47
- **Airfoil turbulence**: Lüthi et al., "Some aspects of high-frequency boundary layer turbulence"

---

## Script Structure at a Glance

```
LOAD GEOMETRY AND MESH
    ↓
VERIFY SLICE STRUCTURE (single x-plane?)
    ↓
SORT FILES BY ITERATION NUMBER (not alphabetical!)
    ↓
SELECT PROBE LOCATIONS (find closest grid points)
    ↓
VISUALIZE PROBE SETUP
    ↓
EXTRACT TIME SERIES (load all snapshots)
    ↓
ROTATE TO FREESTREAM COORDINATES (by AOA)
    ↓
COMPUTE RFFT AT EACH PROBE AND Z-POSITION
    ↓
AVERAGE SPECTRA OVER Z
    ↓
VALIDATE VARIANCE (time vs. spectral)
    ↓
CREATE PLOTS (log-log spectra)
    ↓
SAVE TO HDF5
    ↓
SUMMARY
```

---

**Created**: 2026-03-25
**Script Version**: v2 (Refactored with multi-slice support)
**Author**: Claude Code Assistant
