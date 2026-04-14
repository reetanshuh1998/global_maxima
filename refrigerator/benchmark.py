import numpy as np
import json
import os
import time
from scipy.optimize import minimize
from physics_model import compute_metrics, is_valid_physics, BOUNDS, PARAM_NAMES

# ─── Optimization Objectives ────────────────────────────────────────────────
def objective_qc(x):
    _, qc, _, _, _, _, ok = compute_metrics(*x)
    return -qc if ok else 1e6

def objective_chi(x):
    # Chi = Qc * COP
    cop, qc, _, _, _, _, ok = compute_metrics(*x)
    return -(qc * cop) if ok else 1e6

def objective_cop(x):
    cop, _, _, _, _, _, ok = compute_metrics(*x)
    return -cop if ok else 1e6

# ─── Simple SLSQP Multi-start ───────────────────────────────────────────────
def find_optimal_slsqp(obj_func, n_restarts=50):
    best_val = 1e10
    best_x = None
    lo = np.array([b[0] for b in BOUNDS])
    hi = np.array([b[1] for b in BOUNDS])
    
    for _ in range(n_restarts):
        x0 = np.random.uniform(lo, hi)
        res = minimize(obj_func, x0, bounds=BOUNDS, method='SLSQP')
        if res.success and res.fun < best_val:
            if is_valid_physics(*res.x):
                # Verify it's actually in refrigerator mode
                *_, ok = compute_metrics(*res.x)
                if ok:
                    best_val = res.fun
                    best_x = res.x
    return best_x, -best_val

# ─── Main Benchmark ────────────────────────────────────────────────────────
def run_refrigerator_benchmark():
    results = {}
    
    print("Optimization Phase: Starting 'From-Scratch' Search...")
    
    # 1. Max Cooling Power (Qc)
    print("Target: Maximum Cooling Power (Qc)...")
    x_qc, val_qc = find_optimal_slsqp(objective_qc, 100)
    if x_qc is not None:
        metrics = compute_metrics(*x_qc)
        results['max_qc'] = {
            'params': {n: float(v) for n, v in zip(PARAM_NAMES, x_qc)},
            'metrics': {
                'cop': metrics[0], 'qc': metrics[1], 'qh': metrics[2], 
                'w': metrics[3], 'cop_carnot': metrics[4], 'cop_harmonic': metrics[5]
            }
        }
        print(f"  Max Qc found: {val_qc:.6e} at COP={metrics[0]:.4f}")

    # 2. Max Chi (Qc * COP)
    print("Target: Maximum Chi (Qc * COP)...")
    x_chi, val_chi = find_optimal_slsqp(objective_chi, 100)
    if x_chi is not None:
        metrics = compute_metrics(*x_chi)
        results['max_chi'] = {
            'params': {n: float(v) for n, v in zip(PARAM_NAMES, x_chi)},
            'metrics': {
                'cop': metrics[0], 'qc': metrics[1], 'qh': metrics[2], 
                'w': metrics[3], 'cop_carnot': metrics[4], 'cop_harmonic': metrics[5]
            }
        }
        print(f"  Max Chi found: {val_chi:.6e} at COP={metrics[0]:.4f}, Qc={metrics[1]:.6e}")

    # 3. Max COP (Pure Optimization)
    print("Target: Maximum COP (Note: Reversible limit)...")
    x_cop, val_cop = find_optimal_slsqp(objective_cop, 100)
    if x_cop is not None:
        metrics = compute_metrics(*x_cop)
        results['max_cop_pure'] = {
            'params': {n: float(v) for n, v in zip(PARAM_NAMES, x_cop)},
            'metrics': {
                'cop': metrics[0], 'qc': metrics[1], 'qh': metrics[2], 
                'w': metrics[3], 'cop_carnot': metrics[4], 'cop_harmonic': metrics[5]
            }
        }
        print(f"  Max COP found: {val_cop:.4f} (Qc={metrics[1]:.2e})")

    # Save Results
    os.makedirs('results', exist_ok=True)
    with open('results/refrigerator_optima.json', 'w') as f:
        json.dump(results, f, indent=4)
    print("\nResults saved to refrigerator/results/refrigerator_optima.json")

if __name__ == "__main__":
    np.random.seed(42)
    run_refrigerator_benchmark()
