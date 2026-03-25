"""
physics_model.py — CPC-Ready Anharmonic Otto Cycle Physics
==========================================================
Implements Eqs. (8–10) from the reference paper with:
  - Numerically stable coth (no overflow for large arguments)
  - All 5 parameters free: (beta_c, beta_h, omega_c, omega_h, lambda)
  - Paper-anchored ratio caps (from Fig. 2, image2 of reference):
        omega_h/omega_c <= R_OMEGA_MAX  (2 × paper's 1.5)
        beta_c/beta_h   <= R_BETA_MAX   (2 × paper's 2.0)
  - Perturbation validity: lambda <= LAM_MAX (anharmonic correction < 10%)
  - Minimum work output W_min (set from pilot scan)

References:
  Eqs. 8–10 for Q_h, Q_c, W_ext, eta.
  Fig. 2 caption: beta_h = 0.5*beta_c, omega_c=2, omega_h=3 → ratio anchors.
"""

import numpy as np
import warnings
warnings.filterwarnings("ignore")

# ─── Paper-anchored constraint caps (cite Fig. 2 of reference) ─────────────
R_OMEGA_MAX = 7.0   # NV-centre maximum — Klatzow et al., PRL 2019
R_BETA_MAX  = 25.0  # NV-centre spin-temperature range — Klatzow et al., PRL 2019
LAM_MAX     = 0.2   # perturbation validity: anharmonic correction ≤ ~10%

# Pilot-scan-derived minimum work (computed below in pilot_scan())
W_MIN = None   # set by call to calibrate_w_min() once


# ─── Stable coth implementation ─────────────────────────────────────────────

def stable_coth(x: float) -> float:
    """
    Numerically stable hyperbolic cotangent.
    - For |x| > 20: coth(x) ≈ sign(x) * (1 + 2*exp(-2|x|))
    - For |x| < 1e-6: coth(x) ≈ 1/x + x/3  (Laurent series)
    - Otherwise: 1/tanh(x)
    """
    ax = abs(x)
    if ax < 1e-10:
        return np.sign(x) * 1e10   # diverges; treat as very large
    elif ax < 1e-3:
        # Laurent series: coth(x) = 1/x + x/3 - x^3/45 + ...
        return 1.0/x + x/3.0 - x**3/45.0
    elif ax > 20.0:
        # Asymptotic: coth(x) ≈ sign(x)*(1 + 2*exp(-2|x|))
        return float(np.sign(x)) * (1.0 + 2.0*np.exp(-2.0*ax))
    else:
        return 1.0 / np.tanh(x)

def stable_coth_vec(x):
    """Vectorized stable_coth."""
    return np.vectorize(stable_coth)(x)


# ─── Core physics (Eqs. 8–10 from the paper) ────────────────────────────────

def compute_eta(bc, bh, wc, wh, lam):
    """
    Compute efficiency eta = W_ext / Q_h using Eqs. (8–10) of the paper.

    All arguments must satisfy the constraint checks (see is_feasible).
    Returns (eta, Q_h, Q_c, W_ext) or (0,0,0,0) if engine mode violated.
    """
    # Eq. (8): Q_h = (wh/2)*(X - Y) + (3*lam/(8*wh^2))*(X^2 - Y^2)
    # Eq. (9): Q_c = (wc/2)*(Y - X) + (3*lam/(8*wc^2))*(Y^2 - X^2)
    # where X = coth(bh*wh/2), Y = coth(bc*wc/2)
    X = stable_coth(bh * wh / 2.0)
    Y = stable_coth(bc * wc / 2.0)

    Q_h = (wh/2.0)*(X - Y) + (3.0*lam/(8.0*wh**2))*(X**2 - Y**2)
    Q_c = (wc/2.0)*(Y - X) + (3.0*lam/(8.0*wc**2))*(Y**2 - X**2)
    W_ext = Q_h + Q_c    # Eq. (10): W_ext = Q_h + Q_c

    if Q_h <= 0 or W_ext <= 0:
        return 0.0, 0.0, 0.0, 0.0

    eta = W_ext / Q_h
    eta = float(np.clip(eta, 0.0, 1.0))
    return eta, Q_h, Q_c, W_ext


def is_feasible(bc, bh, wc, wh, lam, w_min=None):
    """
    Check all physical + benchmark constraints.
    Returns (bool, reason_string).
    """
    # Engine-mode ordering constraints
    if wc >= wh:         return False, "wc >= wh"
    if bh >= bc:         return False, "bh >= bc"
    if lam < 0:          return False, "lam < 0"
    if bc*wc <= bh*wh:   return False, "engine mode violated"

    # Paper-anchored ratio caps
    if wh/wc > R_OMEGA_MAX:   return False, f"omega_ratio > {R_OMEGA_MAX}"
    if bc/bh > R_BETA_MAX:    return False, f"beta_ratio  > {R_BETA_MAX}"

    # Perturbation validity: anharmonic correction / harmonic term < 10%
    # At point (omega_c, beta_c): anharmonic ~ 3*lam/(8*wc^2)*coth^2,
    # harmonic ~ wc/2 * coth. Ratio must be < 0.1
    if lam > LAM_MAX:
        return False, f"lam > {LAM_MAX}"
    for w, b in [(wc, bc), (wh, bh)]:
        coth_val  = stable_coth(b*w/2.0)
        harmonic  = w/2.0 * abs(coth_val)
        anharmonic = 3.0*lam/(8.0*w**2) * coth_val**2
        if harmonic > 0 and anharmonic/harmonic > 0.1:
            return False, "perturbation not valid (anharmonic/harmonic > 10%)"

    # Minimum work output
    if w_min is not None:
        _, _, _, W = compute_eta(bc, bh, wc, wh, lam)
        if W < w_min:
            return False, f"W_ext < W_min ({w_min:.4f})"

    return True, "ok"


def eta_fn(bc, bh, wc, wh, lam, w_min=None):
    """Single-call function: check feasibility + return eta (0 if infeasible)."""
    ok, _ = is_feasible(bc, bh, wc, wh, lam, w_min=w_min)
    if not ok:
        return 0.0
    eta, *_ = compute_eta(bc, bh, wc, wh, lam)
    return eta


# ─── Box bounds (for optimizers) ────────────────────────────────────────────
# These are loose box bounds; ratio caps + other constraints narrow the feasible set
BOUNDS = {
    'beta_c':  (0.5,  20.0),
    'beta_h':  (0.1,  10.0),
    'omega_c': (1.0,   8.0),
    'omega_h': (2.0,  15.0),
    'lam':     (0.0,   LAM_MAX),
}
BOUNDS_LIST  = [BOUNDS[k] for k in ['beta_c','beta_h','omega_c','omega_h','lam']]
PARAM_NAMES  = ['beta_c','beta_h','omega_c','omega_h','lam']


def sample_feasible(rng, w_min=None, max_tries=10000):
    """Draw one uniformly random feasible parameter vector."""
    lo = np.array([b[0] for b in BOUNDS_LIST])
    hi = np.array([b[1] for b in BOUNDS_LIST])
    for _ in range(max_tries):
        x = rng.uniform(lo, hi)
        bc, bh, wc, wh, lam = x
        ok, _ = is_feasible(bc, bh, wc, wh, lam, w_min=w_min)
        if ok:
            return x
    return None


def calibrate_w_min(n_samples=10000, percentile=25, seed=0):
    """
    Compute W_min from a pilot scan of feasible samples.
    Sets W_MIN globally.
    """
    global W_MIN
    rng = np.random.default_rng(seed)
    works = []
    tries = 0
    while len(works) < n_samples and tries < n_samples * 50:
        x = sample_feasible(rng, w_min=None)
        tries += 1
        if x is not None:
            _, _, _, W = compute_eta(*x)
            if W > 0:
                works.append(W)
    W_MIN = float(np.percentile(works, percentile)) if works else 0.0
    return W_MIN


# ─── Run calibration at import time ─────────────────────────────────────────
if __name__ == "__main__":
    w_min = calibrate_w_min(n_samples=5000)
    print(f"Calibrated W_min = {w_min:.6f}")
    print(f"Constraints: omega_ratio ≤ {R_OMEGA_MAX}, beta_ratio ≤ {R_BETA_MAX}, lambda ≤ {LAM_MAX}")

    # Quick sanity check
    rng = np.random.default_rng(42)
    best = 0.0
    for _ in range(1000):
        x = sample_feasible(rng, w_min=w_min)
        if x is not None:
            e = eta_fn(*x, w_min=w_min)
            best = max(best, e)
    print(f"Best eta in 1000 feasible samples: {best:.4f}")
    print(f"(Max possible with caps: < {1 - 1/R_OMEGA_MAX:.2f} for harmonic limit)")
