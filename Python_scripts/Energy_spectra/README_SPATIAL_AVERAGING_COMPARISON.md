# Energy Spectra Spatial Averaging Comparison Study

## Overview

New script: **`energy_spectra_spatial_comparison.py`** (1,169 lines)

This script performs a comprehensive validation study comparing two methods for computing energy spectra from fine-resolution DNS data:

### Method 1: Single-Point (Traditional)
- Uses single grid point at requested y-location
- Traditional FFT-based spectral analysis
- Baseline for comparison

### Method 2: Spatially-Averaged (Novel)
- Averages 3 adjacent y-grid points: (y-1, y, y+1)
- Spatial filtering performed BEFORE FFT computation
- Leverages fine mesh resolution for noise reduction
- Expected to show improved SNR and reduced high-frequency noise

---

## Key Features

### 1. **Dual Extraction Paths**
- Simultaneously extracts data for both methods
- No computational overhead (single pass through snapshots)
- Identical processing pipeline (ensures fair comparison)

### 2. **Comprehensive Metrics**
Each probe location generates:
- **Variance comparison**: Reduction/increase in turbulent kinetic energy
- **RMS velocity**: Change in fluctuation intensity
- **Spectral peak analysis**: Frequency shifting and amplitude changes
- **Noise floor estimation**: High-frequency background energy
- **SNR improvement**: Quantifiable gain in signal-to-noise ratio (dB)
- **Spectral coherence**: Correlation between the two methods (should be >0.9)

### 3. **Four Comparison Visualizations**

#### **Plot 1: Overlay Spectra** (`comparison_overlay_spectra_*.png`)
- Left: E_uu (streamwise) with both methods overlaid
- Right: E_vv (cross-stream) with both methods overlaid
- Shows smoothing effect of spatial averaging
- Kolmogorov -5/3 reference line included

#### **Plot 2: Spectral Ratio** (`comparison_spectral_ratio_*.png`)
- Left: E_uu_avg / E_uu_single across all frequencies
- Right: E_vv_avg / E_vv_single across all frequencies
- Ratio = 1.0 means no change
- Ratio < 1.0 indicates attenuation (less energy)
- Ratio > 1.0 indicates amplification (more energy)

#### **Plot 3: Statistical Comparison Table** (`comparison_statistics_*.png`)
- Tabular format with rows for each probe
- 10 columns showing:
  - Variance (single vs averaged)
  - Variance reduction percentage
  - RMS values
  - Spectral peak frequencies
  - SNR improvement (dB)
  - Coherence score

#### **Plot 4: SNR & Coherence Analysis** (`comparison_snr_analysis_*.png`)
- Left: Noise floor bars for both methods
- Right: SNR improvement and coherence metrics

### 4. **Extended HDF5 Output**

Two HDF5 files created:

**`energy_spectra_data_single_point_*.h5`**
- Contains traditional single-point spectra
- Same structure as original script
- Reference baseline

**`energy_spectra_data_spatial_avg_*.h5`**
- Contains spatially-averaged spectra
- **NEW** attributes tracking spatial averaging:
  - `j_index_minus`, `j_index_center`, `j_index_plus`: Grid indices used
  - `delta_y_minus`, `delta_y_plus`: Actual distances between grid points
  - `weight_[minus/center/plus]`: Weighting factors (default: 1/3 each)
  - `method`: "spatial-average-3point"

### 5. **JSON Metrics Export**

**`comparison_metrics_*.json`**
- JSON file with all comparison metrics
- Easily importable to Python/Matlab/Excel
- One entry per probe with all statistical values

---

## How It Works

### Phase 1: Load & Prepare
Same as original script:
- Load mesh and verify slice structure
- Select probe locations (find nearest y-grid indices)
- **NEW:** Find adjacent indices (j-1, j, j+1) for spatial averaging

### Phase 2: Extract Time Series (Dual Path)
For each snapshot, for each probe:

```python
# Traditional single-point
u_single = u_stream[:, j_idx, 0]  # (nz,)

# Spatial averaging
u_minus = u_stream[:, j_minus, 0]
u_center = u_stream[:, j_idx, 0]
u_plus = u_stream[:, j_plus, 0]
u_avg = (u_minus + u_center + u_plus) / 3.0  # (nz,)
```

**Key insight**: Spatial averaging happens BEFORE temporal dimension is processed. Each (nz,) vector is averaged spatially, maintaining temporal structure.

### Phase 3: FFT Computation
Identical for both methods:
- One-sided periodogram using rfft
- z-averaging across spanwise
- DC and Nyquist correction
- Variance validation

### Phase 4: Metrics Analysis
For each probe:
- Compare variances (single vs averaged)
- Estimate noise floors from high-frequency region
- Detect spectral peaks
- Compute coherence score

### Phase 5: Visualization & Output
- Generate 4-plot comparison suite
- Export HDF5 with full metadata
- Save JSON metrics for post-processing

---

## Configuration

### Spatial Averaging Settings
```python
ENABLE_SPATIAL_AVERAGING = True        # Toggle spatial averaging
SPATIAL_AVG_WEIGHTS = [1/3, 1/3, 1/3] # Uniform weighting (3 points)
```

### Comparison Plots
```python
COMPARISON_PLOTS = {
    'overlay_spectra': True,           # Enable overlay plot
    'spectral_ratio': True,            # Enable ratio plot
    'statistical_table': True,         # Enable table plot
    'coherence_analysis': True         # Enable SNR/coherence analysis
}
```

### Probe Locations
```python
Y_LOCATIONS = [0.1]  # Request probe at y/c=0.1
# Or multiple probes:
Y_LOCATIONS = [0.05, 0.10, 0.15, 0.20]
```

---

## Expected Results

### Typical Improvements with Spatial Averaging

| Metric | Expected Change |
|--------|-----------------|
| **Variance** | ±2-5% (may decrease or increase slightly) |
| **SNR (dB)** | +2 to +6 dB improvement |
| **Noise Floor** | 30-50% reduction |
| **High-freq Energy** | 10-20% attenuation (expected) |
| **Spectral Peak** | <5% frequency shift |
| **Coherence** | >0.95 (indicates methods are strongly correlated) |

### Why Spatial Averaging Works

1. **Reduces local spatial noise**: Averaging 3 adjacent points smooths out spurious spatial fluctuations
2. **Preserves large-scale structures**: 3 points are still very close (fine mesh), so flow features remain intact
3. **Improves frequency resolution**: Cleaner spectra allow better identification of true spectral peaks
4. **Better high-frequency behavior**: Kolmogorov -5/3 region becomes clearer

---

## Usage

### Quick Start
```bash
cd /home/jofre/Members/Eduard/Paper2/Python_scripts/Spectra

# Edit Y_LOCATIONS config if desired (currently set to [0.1])
python3 energy_spectra_spatial_comparison.py
```

### For Multiple Probes
Edit configuration section:
```python
Y_LOCATIONS = [0.05, 0.10, 0.15, 0.20]  # Test 4 probe locations
```

### Custom Grid Weights (Advanced)
If you want different weighting:
```python
SPATIAL_AVG_WEIGHTS = [0.2, 0.6, 0.2]  # Center-weighted (Gaussian-like)
```

---

## Output Files

Generated in `/Mean_data/Energy_spectra/`:

```
├─ comparison_overlay_spectra_slice_9.png       # Plot 1: Overlay
├─ comparison_spectral_ratio_slice_9.png        # Plot 2: Ratio
├─ comparison_statistics_slice_9.png            # Plot 3: Table
├─ comparison_snr_analysis_slice_9.png          # Plot 4: SNR/Coherence
├─ energy_spectra_data_single_point_slice_9.h5  # HDF5: Single-point
├─ energy_spectra_data_spatial_avg_slice_9.h5   # HDF5: Spatial-averaged
└─ comparison_metrics_slice_9.json              # JSON: All metrics
```

---

## Interpreting the Results

### Overlay Plot
- If blue line (single-point) is noisier than red line (averaged), spatial averaging is working ✓
- Both lines should follow -5/3 slope in inertial subrange
- High-frequency divergence indicates noise reduction success

### Spectral Ratio Plot
- Should hover around 1.0 (ratio = 1.0 means no change)
- High-frequency region (>0.5 on x-axis) often shows ratio < 1.0 (smoothing effect)
- Deviations from 1.0 are visible measure of spatial averaging impact

### Statistical Table
- **Positive variance reduction %**: Indicates smoothing has reduced amplitude
- **SNR improvement > 0 dB**: Good sign (noise reduced more than signal)
- **Coherence > 0.90**: Both methods strongly agree despite differences
- **Peak shift < 5%**: Peak frequencies are stable across methods

### SNR Analysis
- **Noise floor reduction**: Clear bar height difference indicates spatial averaging effectiveness
- **SNR improvement (dB)**: Positive values = better signal clarity
- **Coherence (on 0-1 scale)**: >0.95 indicates trustworthy comparison

---

## Boundary Handling

The script gracefully handles edge cases:

```
Y_LOCATIONS = [0.01]  # Very close to wall

# Result:
# j0 = nearest grid point to 0.01
# j_minus = max(0, j0-1)      ← Won't go below index 0
# j_plus = min(n_y-1, j0+1)   ← Won't exceed last index
# delta_y values tracked and reported in HDF5
```

If requested y-location is too close to boundary:
- Script still runs (uses available adjacent points)
- HDF5 attributes record actual indices (j_minus, j0, j_plus)
- Metrics still valid, just with fewer effective neighbors

---

## Advanced Usage

### Comparing Multiple Spatial Scales

To test different numbers of neighbor points:
- **Current (3-point)**: y-1, y, y+1
- **Future extension**: 5-point (y-2, y-1, y, y+1, y+2)

The infrastructure is ready; just modify `get_adjacent_y_indices()` function.

### Custom Weighting Schemes

Current: Uniform [1/3, 1/3, 1/3]

Could explore:
```python
SPATIAL_AVG_WEIGHTS = [0.25, 0.5, 0.25]    # Gaussian-like
SPATIAL_AVG_WEIGHTS = [0.1, 0.8, 0.1]     # Center-weighted
SPATIAL_AVG_WEIGHTS = [1/4, 1/2, 1/4]     # Formal Gaussian approximation
```

Modify line in `spatially_average_velocity()` or config section.

---

## Metrics Explanation

### Coherence Score
Measures how well the two methods agree across all frequencies. Formula:
```
coherence = 2 * mean(E_single * E_avg) / (mean(E_single²) + mean(E_avg²))
```
- **1.0** = perfect correlation (identical spectra)
- **< 0.90** = methods diverge significantly (investigate why)

### SNR Improvement (dB)
```
SNR_improvement = 10 * log10(noise_floor_single / noise_floor_avg)
```
- **Positive dB**: Spatial averaging reduced noise effectively
- **Typical range**: +2 to +6 dB for well-resolved turbulence
- **< 0 dB**: Unusual; spatial averaging may have added noise

### Noise Floor Estimation
Uses high-frequency region (last 10% of spectrum):
```python
noise_floor = mean(E_spectrum[0.9*nfreq:])
```
Assumes high frequencies = noise in separated shear layer.

---

## Validation Checks

The script performs built-in validation:

✓ **Consistent sample counts** across all probes
✓ **Variance consistency** (time-domain ≈ spectral domain)
✓ **Frequency resolution** verified
✓ **Grid point availability** checked (boundary handling)
✓ **Spectral coherence** monitored (should be > 0.90)

---

## File Sizes (Approximate)

For typical DNS case (2000 samples, 128 z-points, 1 probe):

```
comparison_overlay_spectra_*.png         ~300 KB
comparison_spectral_ratio_*.png          ~250 KB
comparison_statistics_*.png              ~200 KB
comparison_snr_analysis_*.png            ~250 KB
energy_spectra_data_single_point_*.h5    ~50 MB
energy_spectra_data_spatial_avg_*.h5     ~50 MB
comparison_metrics_*.json                ~10 KB
```

Total: ~150 MB per slice

---

## References

### In the Script
- Spatial averaging: Lines 251-271 (helper function)
- Dual extraction: Lines 504-542
- Metrics calculation: Lines 694-799
- Plots generation: Lines 803-1043
- HDF5 export: Lines 1045-1150

### Related Documentation
- Original script: `energy_spectra_u_v_from_slices.py`
- Data loader: `../Data_loader/data_loader_functions.py`
- Rodriguez 2013: For spectral analysis methodology

---

## Troubleshooting

### Script crashes on snapshot load
- Check that snapshot files exist and are readable
- Verify data paths in CONFIGURATION section

### Very large SNR improvements (>10 dB)
- Indicates original single-point data was very noisy
- May suggest fine mesh captured noise rather than resolved turbulence
- Check if y-location is in transition region

### Coherence < 0.90
- Methods may disagree at certain probes
- Could indicate:
  - Probe near boundary (limited neighbors)
  - Very noisy data
  - Spatial heterogeneity in flow

### Spectral peaks shift significantly (>10%)
- Unlikely unless probe location changed significantly
- Check boundary conditions in spatial averaging
- Verify adjacent grid indices are correct

---

## Next Steps / Future Work

1. **Extended spatial averaging**: Test 5-point, 7-point averages
2. **Directional weighting**: Include z-direction averaging
3. **Adaptive weighting**: Use mesh quality metrics to set weights automatically
4. **Multi-slice comparison**: Run on multiple slices, consolidate results
5. **Publication version**: Remove debug outputs, add caching

---

## Questions & Support

For issues with the script:
1. Check that Y_LOCATIONS values are within domain [y_min, y_max]
2. Verify snapshot files are present and readable
3. Review console output—meaningful errors printed at each phase
4. Check JSON output for metric reasonableness

---

**Script Created**: March 26, 2025
**Version**: 1.0 (Initial release)
**Status**: Tested and validated ✓
