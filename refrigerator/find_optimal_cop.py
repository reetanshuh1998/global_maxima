import numpy as np
from scipy.optimize import minimize
from physics_model import compute_cop, is_feasible, BOUNDS, PARAM_NAMES

def objective_cop(x):
    cop, _, _, _ = compute_cop(*x)
    return -cop if cop > 0 else 1e6

def objective_qc(x):
    _, qc, _, _ = compute_cop(*x)
    return -qc if qc > 0 else 1e6

def find_optimal(objective_func, n_restarts=100):
    best_val = 1e6
    best_x = None
    
    lo = np.array([b[0] for b in BOUNDS])
    hi = np.array([b[1] for b in BOUNDS])
    
    for _ in range(n_restarts):
        x0 = np.random.uniform(lo, hi)
        res = minimize(
            objective_func, x0, 
            bounds=BOUNDS, 
            method='SLSQP',
            options={'ftol': 1e-9}
        )
        if res.success and res.fun < best_val:
            # Check feasibility after optimization
            if is_feasible(*res.x):
                best_val = res.fun
                best_x = res.x
                
    return best_x, -best_val

if __name__ == "__main__":
    print("Finding Maximum COP (from scratch)...")
    x_cop, val_cop = find_optimal(objective_cop, 200)
    if x_cop is not None:
        print(f"Max COP: {val_cop:.6f}")
        for name, val in zip(PARAM_NAMES, x_cop):
            print(f"  {name}: {val:.4f}")
        _, qc, qh, w = compute_cop(*x_cop)
        print(f"  Qc: {qc:.6f}, Qh: {qh:.6f}, W: {w:.6f}")
    else:
        print("Failed to find valid COP optimum.")

    print("\nFinding Maximum Cooling Power Qc...")
    x_qc, val_qc = find_optimal(objective_qc, 200)
    if x_qc is not None:
        cop, qc, qh, w = compute_cop(*x_qc)
        print(f"Max Qc: {qc:.6f}")
        for name, val in zip(PARAM_NAMES, x_qc):
            print(f"  {name}: {val:.4f}")
        print(f"  COP at Max Qc: {cop:.6f}")
        print(f"  Qh: {qh:.6f}, W: {w:.6f}")
