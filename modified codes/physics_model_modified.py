import numpy as np
import warnings
warnings.filterwarnings("ignore")

# ─── Paper-anchored constraint caps (from codes/physics_model.py) ───────────
R_OMEGA_MAX = 7.0
R_BETA_MAX  = 25.0
LAM_MAX     = 0.2

def lamba_ab(wc, wh):
    return (wc**2 + wh**2) / (2 * wc * wh)
    
lamba_cd = 1

def stable_coth(x: float) -> float:
    ax = abs(x)
    if ax < 1e-10: return np.sign(x) * 1e10
    if ax < 1e-3:  return 1.0/x + x/3.0 - x**3/45.0
    if ax > 20.0:  return float(np.sign(x)) * (1.0 + 2.0*np.exp(-2.0*ax))
    return 1.0 / np.tanh(x)

def compute_metrics(bc, bh, wc, wh, lam, custom_lamba_ab=None, custom_lamba_cd=None):
    """
    Returns (eta, Qh, Qc, W_ext, eta_carnot, is_engine)
    Definition: eta = W_ext / Qh
    """
    X = stable_coth(bh * wh / 2.0)
    Y = stable_coth(bc * wc / 2.0)

    # Resolve lambda values (evaluate function or use custom values/defaults)
    l_ab = custom_lamba_ab if custom_lamba_ab is not None else lamba_ab(wc, wh)
    l_cd = custom_lamba_cd if custom_lamba_cd is not None else lamba_cd

    # Q_h and Q_c from reference Eqs 8, 9
    Q_h = (wh/2.0)*(X - l_ab*Y) + (3.0*lam/(8.0*wh**2))*(X**2 - l_ab*Y**2)
    Q_c = (wc/2.0)*(Y - l_cd*X) + (3.0*lam/(8.0*wc**2))*(Y**2 - l_cd*X**2)
    W_ext = Q_h + Q_c

    if Q_h <= 0 or W_ext <= 0:
        return 0.0, Q_h, Q_c, W_ext, 0.0, False

    eta = W_ext / Q_h
    eta = float(np.clip(eta, 0.0, 1.0))
    eta_carnot = 1.0 - bh/bc if bc > 0 else 0.0

    return eta, Q_h, Q_c, W_ext, eta_carnot, True

def is_valid_physics(bc, bh, wc, wh, lam):
    """Perturbation validity check (10% threshold) and basic ordering."""
    if wc >= wh:         return False
    if bh >= bc:         return False
    if lam < 0 or lam > LAM_MAX: return False
    if bc*wc <= bh*wh:   return False

    if wh/wc > R_OMEGA_MAX:   return False
    if bc/bh > R_BETA_MAX:    return False
    
    for w, b in [(wc, bc), (wh, bh)]:
        coth_val  = stable_coth(b*w/2.0)
        harmonic  = w/2.0 * abs(coth_val)
        anharmonic = 3.0*lam/(8.0*w**2) * coth_val**2
        if harmonic > 0 and (anharmonic/harmonic) > 0.1:
            return False
            
    return True

BOUNDS = [
    (0.5, 20.0),  # beta_c
    (0.1, 10.0),  # beta_h
    (1.0, 8.0),   # omega_c
    (2.0, 15.0),  # omega_h
    (0.0, LAM_MAX) # lambda
]
PARAM_NAMES = ['beta_c', 'beta_h', 'omega_c', 'omega_h', 'lambda']
