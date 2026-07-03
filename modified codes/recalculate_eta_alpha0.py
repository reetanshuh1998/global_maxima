import numpy as np
from scipy.optimize import minimize
from physics_model_modified import compute_metrics, is_valid_physics, BOUNDS, R_OMEGA_MAX, R_BETA_MAX, stable_coth

BOUNDS_4D = BOUNDS[:4]

def get_lambdas(wc, wh, case):
    l_fn = (wc**2 + wh**2) / (2.0 * wc * wh)
    if case == 1:
        return 1.0, 1.0
    elif case == 2:
        return l_fn, 1.0
    elif case == 3:
        return 1.0, l_fn
    else:  # case 4: both asymmetric
        return l_fn, l_fn

def sample_feasible(case, max_tries=50000):
    lo = np.array([b[0] for b in BOUNDS_4D])
    hi = np.array([b[1] for b in BOUNDS_4D])
    for _ in range(max_tries):
        x = np.random.uniform(lo, hi)
        bc, bh, wc, wh = x
        if is_valid_physics(bc, bh, wc, wh, 0.0):
            l_ab, l_cd = get_lambdas(wc, wh, case)
            _, _, _, _, _, ok = compute_metrics(bc, bh, wc, wh, 0.0, l_ab, l_cd)
            if ok:
                return x
    return None

def optimize_case(case):
    def obj(x):
        bc, bh, wc, wh = x
        l_ab, l_cd = get_lambdas(wc, wh, case)
        eta, _, _, _, _, ok = compute_metrics(bc, bh, wc, wh, 0.0, l_ab, l_cd)
        return -eta if ok else 1e6

    cons = [
        {'type': 'ineq', 'fun': lambda x: x[3] - x[2] - 0.01},
        {'type': 'ineq', 'fun': lambda x: x[0] - x[1] - 0.01},
        {'type': 'ineq', 'fun': lambda x: x[0]*x[2] - x[1]*x[3] - 0.01},
        {'type': 'ineq', 'fun': lambda x: R_OMEGA_MAX - x[3]/x[2]},
        {'type': 'ineq', 'fun': lambda x: R_BETA_MAX - x[0]/x[1]},
    ]

    best_val = 1e10
    best_x = None

    # Use 120 restarts for ultra-high probability of finding absolute global maximum
    for _ in range(120):
        x0 = sample_feasible(case)
        if x0 is None:
            continue
        res = minimize(obj, x0, bounds=BOUNDS_4D, method='SLSQP', constraints=cons)
        if res.success and res.fun < best_val:
            if is_valid_physics(*res.x, 0.0):
                bc, bh, wc, wh = res.x
                l_ab, l_cd = get_lambdas(wc, wh, case)
                _, _, _, _, _, ok = compute_metrics(bc, bh, wc, wh, 0.0, l_ab, l_cd)
                if ok:
                    best_val = res.fun
                    best_x = res.x.copy()

    return -best_val if best_x is not None else np.nan, best_x

if __name__ == '__main__':
    print("=== DYNAMIC RECALCULATION OF ETA_MAX AT ALPHA = 0 ===")
    for c in [1, 2, 3, 4]:
        eta_val, params = optimize_case(c)
        if params is not None:
            bc, bh, wc, wh = params
            print(f"Case {c}:")
            print(f"  Recalculated eta_max = {eta_val:.9f}")
            print(f"  Optimal parameters   = bc={bc:.6f}, bh={bh:.6f}, wc={wc:.6f}, wh={wh:.6f}")
            print(f"  Compression ratio Rw = {wh/wc:.6f}")
        else:
            print(f"Case {c}: Failed to optimize.")
