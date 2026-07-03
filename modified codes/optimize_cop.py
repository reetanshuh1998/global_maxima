import numpy as np
import json
import os
from scipy.optimize import minimize
from physics_model_modified import compute_metrics, is_valid_physics, BOUNDS, PARAM_NAMES

# Bounds and restart parameters
BOUNDS_LIST = BOUNDS
N_RESTARTS = 200

# Objective functions (we minimize -eta)
def objective_case1(x):
    bc, bh, wc, wh, lam = x
    eta, _, _, _, _, ok = compute_metrics(bc, bh, wc, wh, lam, custom_lamba_ab=1.0, custom_lamba_cd=1.0)
    return -eta if ok else 1e6

def objective_case2(x):
    bc, bh, wc, wh, lam = x
    eta, _, _, _, _, ok = compute_metrics(bc, bh, wc, wh, lam, custom_lamba_ab=None, custom_lamba_cd=None)
    return -eta if ok else 1e6

def optimize_case(objective_func):
    best_val = 1e10
    best_x = None
    lo = np.array([b[0] for b in BOUNDS_LIST])
    hi = np.array([b[1] for b in BOUNDS_LIST])
    
    for _ in range(N_RESTARTS):
        x0 = np.random.uniform(lo, hi)
        res = minimize(objective_func, x0, bounds=BOUNDS_LIST, method='SLSQP')
        if res.success and res.fun < best_val:
            if is_valid_physics(*res.x):
                # Verify engine mode and physical validity
                bc, bh, wc, wh, lam = res.x
                if objective_func == objective_case1:
                    eta, _, _, _, _, ok = compute_metrics(bc, bh, wc, wh, lam, custom_lamba_ab=1.0, custom_lamba_cd=1.0)
                else:
                    eta, _, _, _, _, ok = compute_metrics(bc, bh, wc, wh, lam, custom_lamba_ab=None, custom_lamba_cd=None)
                
                if ok:
                    best_val = res.fun
                    best_x = res.x.copy()
                    
    return best_x, -best_val

if __name__ == "__main__":
    np.random.seed(42)
    
    print("Running optimization for Case 1 (lambda_ab = lambda_cd = 1.0)...")
    x_opt1, eta_opt1 = optimize_case(objective_case1)
    
    print("\nRunning optimization for Case 2 (lambda_ab = (wc^2+wh^2)/2wcwh, lambda_cd = 1.0)...")
    x_opt2, eta_opt2 = optimize_case(objective_case2)
    
    results = {}
    if x_opt1 is not None:
        bc, bh, wc, wh, lam = x_opt1
        m = compute_metrics(bc, bh, wc, wh, lam, custom_lamba_ab=1.0, custom_lamba_cd=1.0)
        results["case1"] = {
            "params": {n: float(v) for n, v in zip(PARAM_NAMES, x_opt1)},
            "eta": m[0], "qh": m[1], "qc": m[2], "w": m[3], "eta_carnot": m[4]
        }
        print("\nCase 1 Optimal Parameters:")
        for n, v in zip(PARAM_NAMES, x_opt1):
            print(f"  {n}: {v:.6f}")
        print(f"  Efficiency (eta): {m[0]:.6f} (Carnot limit: {m[4]:.6f})")
        
    if x_opt2 is not None:
        bc, bh, wc, wh, lam = x_opt2
        m = compute_metrics(bc, bh, wc, wh, lam, custom_lamba_ab=None, custom_lamba_cd=None)
        results["case2"] = {
            "params": {n: float(v) for n, v in zip(PARAM_NAMES, x_opt2)},
            "eta": m[0], "qh": m[1], "qc": m[2], "w": m[3], "eta_carnot": m[4]
        }
        print("\nCase 2 Optimal Parameters:")
        for n, v in zip(PARAM_NAMES, x_opt2):
            print(f"  {n}: {v:.6f}")
        print(f"  Efficiency (eta): {m[0]:.6f} (Carnot limit: {m[4]:.6f})")

    # Save to json file
    with open("optimal_parameters_comparison.json", "w") as f:
        json.dump(results, f, indent=4)
        
    print("\nResults saved to optimal_parameters_comparison.json")
