import numpy as np
from scipy.optimize import minimize

def efficiency(params):
    bc, bh, wc, wh, lam = params
    
    # physical constraints to return 0 or negative efficiency if violated
    if wc >= wh or bh >= bc or lam < 0:
        return 0.0
    if bc * wc <= bh * wh: # Positive work condition
        return 0.0
        
    X = np.cosh(bh * wh / 2) / np.sinh(bh * wh / 2) # coth
    Y = np.cosh(bc * wc / 2) / np.sinh(bc * wc / 2) # coth
    
    Q_h = (wh / 2) * (X - Y) + (3 * lam / (8 * wh**2)) * (X**2 - Y**2)
    Q_c = (wc / 2) * (Y - X) + (3 * lam / (8 * wc**2)) * (Y**2 - X**2)
    
    W_ext = Q_h + Q_c
    
    # Engine must extract positive work and have positive input heat
    if Q_h <= 0 or W_ext <= 0:
        return 0.0
        
    eta = W_ext / Q_h
    return eta if eta <= 1 else 0.0

def objective(params):
    return -efficiency(params)

# Constraints
# wc/wh < 1 -> wc - wh < 0
# bh/bc < 1 -> bh - bc < 0
# lam >= 0
# we also need bc * wc > bh * wh for it to be a heat engine

bounds = [
    (0.1, 100),   # beta_c
    (0.01, 10),  # beta_h
    (0.1, 10),   # omega_c
    (1, 100),    # omega_h
    (0.0, 1.0)   # lambda (perturbation must be small typically)
]

def constraint_work(x):
    return x[0]*x[2] - x[1]*x[3] - 0.01

constraints = [
    {'type': 'ineq', 'fun': lambda x: x[3] - x[2] - 0.1}, # wh > wc
    {'type': 'ineq', 'fun': lambda x: x[0] - x[1] - 0.1}, # bc > bh
    {'type': 'ineq', 'fun': constraint_work}              # bc*wc > bh*wh
]

# Multiple random starts to find global maximum
best_eta = -1
best_params = None

for _ in range(50):
    x0 = [
        np.random.uniform(5, 10),
        np.random.uniform(0.1, 1),
        np.random.uniform(0.1, 2),
        np.random.uniform(5, 15),
        np.random.uniform(0, 0.1)
    ]
    res = minimize(objective, x0, bounds=bounds, constraints=constraints, method='SLSQP')
    if -res.fun > best_eta:
        best_eta = -res.fun
        best_params = res.x

print("Optimal Parameters Found:")
print(f"beta_c  = {best_params[0]:.6f}")
print(f"beta_h  = {best_params[1]:.6f}")
print(f"omega_c = {best_params[2]:.6f}")
print(f"omega_h = {best_params[3]:.6f}")
print(f"lambda  = {best_params[4]:.6f}")
print(f"Max Efficiency (eta) = {best_eta:.6f}")
