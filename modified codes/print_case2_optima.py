import os
import json
import numpy as np
from scipy.optimize import minimize
from physics_model_modified import compute_metrics, is_valid_physics, BOUNDS, R_OMEGA_MAX, R_BETA_MAX, stable_coth

BOUNDS_4D = BOUNDS[:4]

def sample_feasible(alpha, l_ab, l_cd, max_tries=20000):
    lo = np.array([b[0] for b in BOUNDS_4D])
    hi = np.array([b[1] for b in BOUNDS_4D])
    for _ in range(max_tries):
        x = np.random.uniform(lo, hi)
        bc, bh, wc, wh = x
        if is_valid_physics(bc, bh, wc, wh, alpha):
            _, _, _, _, _, ok = compute_metrics(bc, bh, wc, wh, alpha, l_ab, l_cd)
            if ok:
                return x
    return None

def optimize_for_alpha(alpha):
    l_ab = None  # Case 2 asymmetric
    l_cd = 1.0
    
    def obj(x):
        bc, bh, wc, wh = x
        eta, _, _, _, _, ok = compute_metrics(bc, bh, wc, wh, alpha, l_ab, l_cd)
        return -eta if ok else 1e6

    cons = [
        {'type': 'ineq', 'fun': lambda x: x[3] - x[2] - 0.01}, # wh > wc
        {'type': 'ineq', 'fun': lambda x: x[0] - x[1] - 0.01}, # bc > bh
        {'type': 'ineq', 'fun': lambda x: x[0]*x[2] - x[1]*x[3] - 0.01}, # engine mode
        {'type': 'ineq', 'fun': lambda x: R_OMEGA_MAX - x[3]/x[2]},
        {'type': 'ineq', 'fun': lambda x: R_BETA_MAX - x[0]/x[1]},
        {'type': 'ineq', 'fun': lambda x: 0.1 - (3.0 * alpha * stable_coth(x[0] * x[2] / 2.0)) / (4.0 * x[2]**3)},
        {'type': 'ineq', 'fun': lambda x: 0.1 - (3.0 * alpha * stable_coth(x[1] * x[3] / 2.0)) / (4.0 * x[3]**3)},
    ]

    best_val = 1e10
    best_x = None
    
    for _ in range(60):  # 60 restarts for extra precision
        x0 = sample_feasible(alpha, l_ab, l_cd)
        if x0 is None:
            continue
        res = minimize(obj, x0, bounds=BOUNDS_4D, method='SLSQP', constraints=cons)
        if res.success and res.fun < best_val:
            if is_valid_physics(*res.x, alpha):
                bc, bh, wc, wh = res.x
                _, _, _, _, _, ok = compute_metrics(bc, bh, wc, wh, alpha, l_ab, l_cd)
                if ok:
                    best_val = res.fun
                    best_x = res.x.copy()
                
    return best_x, -best_val

if __name__ == "__main__":
    np.random.seed(42)
    alphas = np.linspace(0.0, 0.2, 11)  # 11 points for a clean display table
    
    print("\n| Alpha (alpha) | Beta_c | Beta_h | Omega_c | Omega_h | Efficiency (eta) | Carnot Limit |")
    print("| :--- | :--- | :--- | :--- | :--- | :--- | :--- |")
    
    results = []
    for a in alphas:
        x, eta = optimize_for_alpha(a)
        if x is not None:
            bc, bh, wc, wh = x
            eta_c = 1.0 - bh/bc
            print(f"| {a:.3f} | {bc:.6f} | {bh:.6f} | {wc:.6f} | {wh:.6f} | {eta:.6f} | {eta_c:.6f} |")
            results.append({
                "alpha": float(a),
                "beta_c": float(bc),
                "beta_h": float(bh),
                "omega_c": float(wc),
                "omega_h": float(wh),
                "eta": float(eta),
                "eta_carnot": float(eta_c)
            })
        else:
            print(f"| {a:.3f} | FAILED | FAILED | FAILED | FAILED | FAILED | FAILED |")
            
    with open("case2_optimal_parameters.json", "w") as f:
        json.dump(results, f, indent=4)
