import numpy as np
from scipy.optimize import minimize
from physics_model_modified import stable_coth, BOUNDS, R_OMEGA_MAX, R_BETA_MAX

# Bounds for beta_c, beta_h, omega_c, omega_h
BOUNDS_4D = BOUNDS[:4]

def get_lambdas(wc, wh, case):
    l_fn = (wc**2 + wh**2) / (2.0 * wc * wh)
    if case == 1:
        return 1.0, 1.0
    elif case == 2:
        return l_fn, 1.0
    elif case == 3:
        return 1.0, l_fn
    else:
        return l_fn, l_fn

def compute_eta_alpha0(bc, bh, wc, wh, case):
    X = stable_coth(bh * wh / 2.0)
    Y = stable_coth(bc * wc / 2.0)
    l_ab, l_cd = get_lambdas(wc, wh, case)

    Q_h = (wh / 2.0) * (X - l_ab * Y)
    Q_c = (wc / 2.0) * (Y - l_cd * X)
    W_ext = Q_h + Q_c

    if Q_h <= 0 or W_ext <= 0:
        return 0.0

    return W_ext / Q_h

def is_valid(bc, bh, wc, wh):
    if wc >= wh: return False
    if bh >= bc: return False
    if bc*wc <= bh*wh: return False
    if wh/wc > R_OMEGA_MAX + 1e-5: return False
    if bc/bh > R_BETA_MAX + 1e-5: return False
    return True

def find_global_max(case):
    # We will do a random grid sample with 200,000 points first to find candidates,
    # then run SLSQP minimizations starting from the top 50 candidates to find the absolute global max.
    
    np.random.seed(42)
    lo = np.array([b[0] for b in BOUNDS_4D])
    hi = np.array([b[1] for b in BOUNDS_4D])
    
    # 1. Random Sampling
    N = 250000
    samples = np.random.uniform(lo, hi, size=(N, 4))
    
    best_eta = 0.0
    best_pt = None
    
    candidates = []
    
    for x in samples:
        bc, bh, wc, wh = x
        if is_valid(bc, bh, wc, wh):
            eta = compute_eta_alpha0(bc, bh, wc, wh, case)
            if eta > 0:
                candidates.append((eta, x))
                
    # Sort candidates by eta descending
    candidates.sort(key=lambda item: item[0], reverse=True)
    
    # 2. Local SLSQP from top candidates
    cons = [
        {'type': 'ineq', 'fun': lambda x: x[3] - x[2] - 0.01},
        {'type': 'ineq', 'fun': lambda x: x[0] - x[1] - 0.01},
        {'type': 'ineq', 'fun': lambda x: x[0]*x[2] - x[1]*x[3] - 0.01},
        {'type': 'ineq', 'fun': lambda x: R_OMEGA_MAX - x[3]/x[2]},
        {'type': 'ineq', 'fun': lambda x: R_BETA_MAX - x[0]/x[1]},
    ]
    
    best_overall_eta = 0.0
    best_overall_pt = None
    
    # Run minimizer from top 100 random feasible candidates
    for eta_init, x0 in candidates[:100]:
        def obj(x):
            return -compute_eta_alpha0(x[0], x[1], x[2], x[3], case)
            
        res = minimize(obj, x0, bounds=BOUNDS_4D, method='SLSQP', constraints=cons)
        if res.success:
            eta_res = -res.fun
            if eta_res > best_overall_eta and is_valid(*res.x):
                best_overall_eta = eta_res
                best_overall_pt = res.x
                
    return best_overall_eta, best_overall_pt

if __name__ == '__main__':
    print("Starting absolute global verification of eta_max at alpha=0...")
    for c in [1, 2, 3, 4]:
        eta_max, pt = find_global_max(c)
        if pt is not None:
            bc, bh, wc, wh = pt
            print(f"Case {c}:")
            print(f"  eta_max = {eta_max:.9f}")
            print(f"  Parameters: beta_c={bc:.6f}, beta_h={bh:.6f}, omega_c={wc:.6f}, omega_h={wh:.6f}")
            print(f"  R_omega = {wh/wc:.6f}, R_beta = {bc/bh:.6f}")
        else:
            print(f"Case {c}: No feasible point found.")
