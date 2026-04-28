"""
validate_spanwise_fft_correlation.py
-------------------------------------
Validates the FFT-based spanwise circular cross-correlation against a
brute-force direct sum.

Definition being tested (from wall_shear_correlations_2.py):

    u_fft   = np.fft.rfft(u_prime, axis=0)          # (Nz//2+1, Ny, Nx)
    tau_fft = np.fft.rfft(tau_prime_current)          # (Nz//2+1,)
    Num_fft = np.fft.irfft(
                  np.conj(tau_fft)[:, None, None] * u_fft,
                  n=Nz, axis=0)                       # (Nz, Ny, Nx)

    Num_fft[dk, y, x] should equal:
    Num_bf [dk, y, x] = sum_{k=0}^{Nz-1} tau[k] * u[(k+dk) % Nz, y, x]

Run directly:
    python validate_spanwise_fft_correlation.py
"""

import numpy as np


# ============================================================================
# Core computation functions
# ============================================================================

def compute_fft_correlation(tau_prime, u_prime):
    """
    FFT-based circular cross-correlation along axis-0.

    Parameters
    ----------
    tau_prime : ndarray, shape (Nz,)
    u_prime   : ndarray, shape (Nz, Ny, Nx)

    Returns
    -------
    Num_fft : ndarray, shape (Nz, Ny, Nx)
        Num_fft[dk, y, x] = sum_k tau[k] * u[(k+dk)%Nz, y, x]
    """
    Nz = tau_prime.shape[0]
    u_fft   = np.fft.rfft(u_prime, axis=0)          # (Nz//2+1, Ny, Nx)
    tau_fft = np.fft.rfft(tau_prime)                 # (Nz//2+1,)
    Num_fft = np.fft.irfft(
                  np.conj(tau_fft)[:, None, None] * u_fft,
                  n=Nz, axis=0)                      # (Nz, Ny, Nx)
    return Num_fft


def compute_brute_force_correlation(tau_prime, u_prime,
                                    y_slice=None, x_slice=None):
    """
    Brute-force circular cross-correlation along axis-0.

    Num_bf[dk, y, x] = sum_{k=0}^{Nz-1} tau[k] * u[(k+dk) % Nz, y, x]

    Parameters
    ----------
    tau_prime : ndarray, shape (Nz,)
    u_prime   : ndarray, shape (Nz, Ny, Nx)
    y_slice   : slice or None  — restrict y range (for speed)
    x_slice   : slice or None  — restrict x range (for speed)

    Returns
    -------
    Num_bf : ndarray, shape (Nz, Ny_sub, Nx_sub)
    """
    Nz, Ny, Nx = u_prime.shape

    ys = y_slice if y_slice is not None else slice(None)
    xs = x_slice if x_slice is not None else slice(None)

    u_sub = u_prime[:, ys, xs]                      # (Nz, Ny_sub, Nx_sub)
    Ny_sub, Nx_sub = u_sub.shape[1], u_sub.shape[2]

    Num_bf = np.zeros((Nz, Ny_sub, Nx_sub), dtype=np.float64)

    for dk in range(Nz):
        # shifted index array: (k + dk) % Nz for k in 0..Nz-1
        shifted = np.arange(Nz)
        shifted = (shifted + dk) % Nz
        # sum_k tau[k] * u[(k+dk)%Nz, ...]
        Num_bf[dk] = np.einsum('k,kyx->yx', tau_prime, u_sub[shifted])

    return Num_bf


# ============================================================================
# Validation function
# ============================================================================

def validate(tau_prime, u_prime,
             y_slice=None, x_slice=None,
             atol=1e-10,
             label=""):
    """
    Run FFT vs brute-force comparison and print diagnostics.

    Parameters
    ----------
    tau_prime : ndarray, shape (Nz,)
    u_prime   : ndarray, shape (Nz, Ny, Nx)
    y_slice   : slice  — subdomain in y for brute-force
    x_slice   : slice  — subdomain in x for brute-force
    atol      : float  — absolute tolerance for assertion
    label     : str    — optional description printed in header
    """
    Nz, Ny, Nx = u_prime.shape

    hdr = f"  [{label}]" if label else ""
    print(f"\n{'='*60}")
    print(f"VALIDATION{hdr}")
    print(f"  tau_prime shape : {tau_prime.shape}")
    print(f"  u_prime   shape : {u_prime.shape}")

    # --- Make sure dtypes are float64 for both paths ---
    tau_prime = tau_prime.astype(np.float64)
    u_prime   = u_prime.astype(np.float64)

    # --- Determine subdomain ---
    Ny_sub = min(8, Ny)
    Nx_sub = min(8, Nx)
    ys = y_slice if y_slice is not None else slice(0, Ny_sub)
    xs = x_slice if x_slice is not None else slice(0, Nx_sub)
    print(f"  Brute-force subdomain: y={ys}, x={xs}")

    # --- FFT path (full domain) ---
    Num_fft = compute_fft_correlation(tau_prime, u_prime)
    assert Num_fft.shape == (Nz, Ny, Nx), (
        f"Shape mismatch: Num_fft {Num_fft.shape} != expected ({Nz},{Ny},{Nx})")

    # --- Brute-force path (subdomain) ---
    Num_bf = compute_brute_force_correlation(tau_prime, u_prime,
                                             y_slice=ys, x_slice=xs)

    # Matching slice of FFT result
    Num_fft_sub = Num_fft[:, ys, xs]

    assert Num_bf.shape == Num_fft_sub.shape, (
        f"Subdomain shape mismatch: brute-force {Num_bf.shape} "
        f"vs fft-sub {Num_fft_sub.shape}")

    # ---- Diagnostics ----
    diff         = Num_fft_sub - Num_bf
    max_abs_err  = float(np.max(np.abs(diff)))
    max_abs_ref  = float(np.max(np.abs(Num_bf)))
    rel_err      = max_abs_err / max(1e-12, max_abs_ref)

    # L2 norms
    l2_fft = float(np.linalg.norm(Num_fft_sub))
    l2_bf  = float(np.linalg.norm(Num_bf))
    l2_err = float(np.linalg.norm(diff))
    l2_rel = l2_err / max(1e-12, l2_bf)

    print(f"\n  --- Error metrics (subdomain) ---")
    print(f"  max |Num_fft - Num_bf|        = {max_abs_err:.6e}")
    print(f"  max |Num_bf|                  = {max_abs_ref:.6e}")
    print(f"  relative error (max-norm)     = {rel_err:.6e}")
    print(f"  L2 norm  FFT                  = {l2_fft:.6e}")
    print(f"  L2 norm  brute-force          = {l2_bf:.6e}")
    print(f"  L2 error                      = {l2_err:.6e}")
    print(f"  relative error (L2)           = {l2_rel:.6e}")

    # Where is the worst error?
    idx_worst = np.unravel_index(np.argmax(np.abs(diff)), diff.shape)
    print(f"\n  Worst error at (dk={idx_worst[0]}, y={idx_worst[1]}, x={idx_worst[2]}):")
    print(f"    FFT value        = {Num_fft_sub[idx_worst]:.10e}")
    print(f"    Brute-force      = {Num_bf[idx_worst]:.10e}")
    print(f"    Absolute error   = {diff[idx_worst]:.6e}")

    # Spot-check: dk=0 (should be a simple dot product)
    dk0_fft = Num_fft_sub[0, 0, 0]
    dk0_bf  = float(np.dot(tau_prime, u_prime[:, ys, xs][np.arange(Nz), 0, 0]))
    print(f"\n  Spot-check dk=0, (y=0,x=0):")
    print(f"    FFT result   = {dk0_fft:.10e}")
    print(f"    Direct dot   = {dk0_bf:.10e}")
    print(f"    Error        = {abs(dk0_fft - dk0_bf):.6e}")

    # ---- Assertion ----
    assert max_abs_err < atol, (
        f"FAILED: max absolute error {max_abs_err:.3e} exceeds tolerance {atol:.3e}")

    print(f"\n  PASSED  (max_abs_err={max_abs_err:.3e} < atol={atol:.3e})")
    print(f"{'='*60}")

    return max_abs_err, rel_err


# ============================================================================
# Test cases
# ============================================================================

def test_known_result():
    """
    Tiny hand-checkable case: Nz=4, Ny=1, Nx=1.
    tau = [1, 0, 0, 0]  =>  Num[dk] = u[dk % Nz]
    """
    Nz = 4
    rng = np.random.default_rng(42)
    tau   = np.array([1.0, 0.0, 0.0, 0.0])
    u     = rng.standard_normal((Nz, 1, 1))

    Num_fft = compute_fft_correlation(tau, u)
    for dk in range(Nz):
        expected = u[dk, 0, 0]
        got      = Num_fft[dk, 0, 0]
        err      = abs(got - expected)
        assert err < 1e-12, (
            f"Known-result check failed at dk={dk}: got={got:.6e}, "
            f"expected={expected:.6e}, err={err:.3e}")
    print("\n  [known-result]  PASSED  (tau=[1,0,0,0] identity check)")


def test_random_small(Nz=32, Ny=16, Nx=16, seed=0):
    """Small random arrays — brute force over the full domain."""
    rng = np.random.default_rng(seed)
    tau   = rng.standard_normal(Nz)
    u     = rng.standard_normal((Nz, Ny, Nx))
    validate(tau, u,
             y_slice=slice(0, Ny),
             x_slice=slice(0, Nx),
             atol=1e-10,
             label=f"random small Nz={Nz} Ny={Ny} Nx={Nx}")


def test_random_realistic(Nz=128, Ny=200, Nx=300, seed=7):
    """Realistic-ish sizes — brute force on small subdomain only."""
    rng = np.random.default_rng(seed)
    tau   = rng.standard_normal(Nz)
    u     = rng.standard_normal((Nz, Ny, Nx))
    validate(tau, u,
             y_slice=slice(0, 8),
             x_slice=slice(0, 8),
             atol=1e-10,
             label=f"realistic Nz={Nz} Ny={Ny} Nx={Nx}")


def test_sparse_tau(Nz=64, Ny=20, Nx=20, seed=3):
    """Sparse tau (most entries zero) — typical of conditional correlations."""
    rng  = np.random.default_rng(seed)
    tau  = rng.standard_normal(Nz)
    # zero out 75 % of entries
    mask = rng.random(Nz) < 0.75
    tau[mask] = 0.0
    u    = rng.standard_normal((Nz, Ny, Nx))
    validate(tau, u,
             y_slice=slice(0, 8),
             x_slice=slice(0, 8),
             atol=1e-10,
             label=f"sparse tau Nz={Nz} nnz={int((~mask).sum())}")


def test_odd_Nz(Nz=65, Ny=10, Nx=10, seed=5):
    """Odd Nz — rfft/irfft with explicit n=Nz must round-trip correctly."""
    rng  = np.random.default_rng(seed)
    tau  = rng.standard_normal(Nz)
    u    = rng.standard_normal((Nz, Ny, Nx))
    validate(tau, u,
             y_slice=slice(0, 8),
             x_slice=slice(0, 8),
             atol=1e-10,
             label=f"odd Nz={Nz}")


def test_single_snapshot_workflow(Nz=128, Ny=50, Nx=60, seed=99):
    """
    Mimics exactly one snapshot of the production code:
        u_fft   = np.fft.rfft(u_prime, axis=0)
        tau_fft = np.fft.rfft(tau_prime_current)
        Num_fft = np.fft.irfft(np.conj(tau_fft)[:,None,None] * u_fft,
                               n=Nz, axis=0)
    """
    rng  = np.random.default_rng(seed)
    tau_prime_current = rng.standard_normal(Nz)
    u_prime           = rng.standard_normal((Nz, Ny, Nx))

    # --- production code verbatim ---
    u_fft   = np.fft.rfft(u_prime, axis=0)
    tau_fft = np.fft.rfft(tau_prime_current)
    Num_fft = np.fft.irfft(np.conj(tau_fft)[:, None, None] * u_fft,
                            n=Nz, axis=0)

    # --- brute force (subdomain) ---
    ys, xs = slice(0, 8), slice(0, 8)
    Num_bf = compute_brute_force_correlation(tau_prime_current, u_prime,
                                             y_slice=ys, x_slice=xs)

    Num_fft_sub = Num_fft[:, ys, xs]
    diff         = Num_fft_sub - Num_bf
    max_abs_err  = float(np.max(np.abs(diff)))
    max_abs_ref  = float(np.max(np.abs(Num_bf)))
    rel_err      = max_abs_err / max(1e-12, max_abs_ref)

    print(f"\n  [single-snapshot workflow  Nz={Nz} Ny={Ny} Nx={Nx}]")
    print(f"  max |Num_fft - Num_bf| = {max_abs_err:.6e}")
    print(f"  relative error         = {rel_err:.6e}")

    assert max_abs_err < 1e-10, (
        f"FAILED: max absolute error {max_abs_err:.3e} exceeds 1e-10")
    print(f"  PASSED")


# ============================================================================
# Entry point
# ============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("FFT CIRCULAR CROSS-CORRELATION VALIDATION")
    print("=" * 60)

    print("\n--- Test 1: known-result (identity tau) ---")
    test_known_result()

    print("\n--- Test 2: small random (full brute-force) ---")
    test_random_small()

    print("\n--- Test 3: realistic sizes (subdomain brute-force) ---")
    test_random_realistic()

    print("\n--- Test 4: sparse tau ---")
    test_sparse_tau()

    print("\n--- Test 5: odd Nz ---")
    test_odd_Nz()

    print("\n--- Test 6: single-snapshot production workflow ---")
    test_single_snapshot_workflow()

    print("\n" + "=" * 60)
    print("ALL TESTS PASSED")
    print("=" * 60)
