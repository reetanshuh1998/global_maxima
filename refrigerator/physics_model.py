import numpy as np
import warnings
warnings.filterwarnings("ignore")

# ─── Physical Constants and Constraints ──────────────────────────────────────
# We keep caps looser for from-scratch exploration
LAM_MAX = 0.5  

def stable_coth(x: float) -> float:
    ax = abs(x)
    if ax < 1e-10: return np.sign(x) * 1e10
    if ax < 1e-3:  return 1.0/x + x/3.0 - x**3/45.0
    if ax > 20.0:  return float(np.sign(x)) * (1.0 + 2.0*np.exp(-2.0*ax))
    return 1.0 / np.tanh(x)

def compute_metrics(bc, bh, wc, wh, lam):
    """
    Returns (COP, Qc, Qh, W, COP_Carnot, COP_Harmonic, is_refrigerator)
    Definition: COP = -Qc / (Qc + Qh)
    """
    X = stable_coth(bh * wh / 2.0)
    Y = stable_coth(bc * wc / 2.0)

    # Q_h and Q_c from reference Eqs 8, 9
    Q_h = (wh/2.0)*(X - Y) + (3.0*lam/(8.0*wh**2))*(X**2 - Y**2)
    Q_c = (wc/2.0)*(Y - X) + (3.0*lam/(8.0*wc**2))*(Y**2 - X**2)
    W = Q_h + Q_c

    # Carnot COP = Tc / (Th - Tc) = 1 / (bc/bh - 1)
    # bc > bh for Th > Tc
    cop_carnot = 1.0 / (bc/bh - 1.0) if bc > bh else 0.0
    
    # Harmonic Refrigerator COP = 1 / (wh/wc - 1)
    cop_harmonic = 1.0 / (wh/wc - 1.0) if wh > wc else 0.0

    is_refrigerator = (Q_c > 0 and Q_h < 0 and W < 0)
    
    if not is_refrigerator:
        return 0.0, Q_c, Q_h, W, cop_carnot, cop_harmonic, False

    cop = -Q_c / W
    return cop, Q_c, Q_h, W, cop_carnot, cop_harmonic, True

def is_valid_physics(bc, bh, wc, wh, lam):
    """Perturbation validity check (10% threshold) and basic ordering."""
    if wc >= wh: return False
    if bh >= bc: return False # Th > Tc
    if lam < 0 or lam > LAM_MAX: return False
    
    # Refrigerator condition (approximate harmonic limit): wh/wc > bc/bh
    # We use the actual Qc > 0 from compute_metrics for strict checks
    
    for w, b in [(wc, bc), (wh, bh)]:
        coth_val  = stable_coth(b*w/2.0)
        harmonic  = w/2.0 * abs(coth_val)
        anharmonic = 3.0*lam/(8.0*w**2) * coth_val**2
        if harmonic > 0 and (anharmonic/harmonic) > 0.1:
            return False
            
    return True

# Broad bounds for from-scratch exploration
# We focus on the thermal regime: beta*omega ~ 1-5
BOUNDS = [
    (0.1, 20.0),  # beta_c
    (0.01, 10.0), # beta_h
    (0.1, 10.0),  # omega_c
    (0.5, 30.0),  # omega_h
    (0.0, LAM_MAX) # lambda
]
PARAM_NAMES = ['beta_c', 'beta_h', 'omega_c', 'omega_h', 'lambda']
