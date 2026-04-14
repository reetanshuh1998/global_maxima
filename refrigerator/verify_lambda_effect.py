import numpy as np
from scipy.optimize import minimize
from physics_model import compute_cop, is_feasible, BOUNDS, PARAM_NAMES

def optimize_qc_at_lambda(lam_val, n_restarts=20):
    def obj(x_subset):
        bc, bh, wc, wh = x_subset
        _, qc, _, _ = compute_cop(bc, bh, wc, wh, lam_val)
        return -qc if qc > 0 else 1e6
        
    subset_bounds = BOUNDS[:4]
    best_qc = 0
    best_cop = 0
    
    lo = np.array([b[0] for b in subset_bounds])
    hi = np.array([b[1] for b in subset_bounds])
    
    for _ in range(n_restarts):
        x0 = np.random.uniform(lo, hi)
        res = minimize(obj, x0, bounds=subset_bounds, method='SLSQP')
        if res.success and -res.fun > best_qc:
            if is_feasible(*res.x, lam_val):
                best_qc = -res.fun
                cop, _, _, _ = compute_cop(*res.x, lam_val)
                best_cop = cop
    return best_qc, best_cop

lams = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
print("Lambda | Max Qc | COP at Max Qc")
print("-------|--------|--------------")
for l in lams:
    qc, cop = optimize_qc_at_lambda(l)
    print(f"{l:.1f}    | {qc:.6e} | {cop:.4f}")
