import os
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize
from physics_model_modified import compute_metrics, is_valid_physics, BOUNDS, R_OMEGA_MAX, R_BETA_MAX, stable_coth

# Setup directory for saving plots
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PLOTS_DIR = os.path.join(SCRIPT_DIR, 'plots')
os.makedirs(PLOTS_DIR, exist_ok=True)

# 1. Remove all old files in plots directory to avoid wrong plots
print("Cleaning old plots...")
for filename in os.listdir(PLOTS_DIR):
    file_path = os.path.join(PLOTS_DIR, filename)
    try:
        if os.path.isfile(file_path):
            os.remove(file_path)
            print(f"Removed old plot: {file_path}")
    except Exception as e:
        print(f"Error removing {file_path}: {e}")

# Bounds
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

def optimize_for_alpha(alpha, is_case1=True):
    l_ab = 1.0 if is_case1 else None
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
    
    # 50 restarts for high accuracy
    for _ in range(50):
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
                
    return -best_val if best_x is not None else np.nan

# Define alpha sweep range (15 points for solid resolution and quick run time)
alphas = np.linspace(0.0, 0.2, 15)

print("\nStarting envelope optimization for Case 1 (Symmetric)...")
etas_case1 = []
for a in alphas:
    val = optimize_for_alpha(a, is_case1=True)
    etas_case1.append(val)
    print(f"  alpha = {a:.3f} -> max eta = {val:.6f}")

print("\nStarting envelope optimization for Case 2 (Asymmetric)...")
etas_case2 = []
for a in alphas:
    val = optimize_for_alpha(a, is_case1=False)
    etas_case2.append(val)
    print(f"  alpha = {a:.3f} -> max eta = {val:.6f}")

etas_case1 = np.array(etas_case1)
etas_case2 = np.array(etas_case2)

# Save the optimized results
results_dict = {
    "alphas": alphas.tolist(),
    "case1_etas": etas_case1.tolist(),
    "case2_etas": etas_case2.tolist()
}
with open(os.path.join(SCRIPT_DIR, "optimized_envelope_data.json"), "w") as f:
    json.dump(results_dict, f, indent=4)

# ─── Plot Styling ───────────────────────────────────────────────────────────
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'Times', 'DejaVu Serif'],
    'mathtext.fontset': 'stix',
    'font.size': 10,
    'axes.labelsize': 11,
    'axes.titlesize': 12,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'axes.grid': True,
    'grid.alpha': 0.25,
    'grid.linestyle': '--',
    'figure.dpi': 300,
    'savefig.dpi': 300,
})

# ─── FIGURE: Combined Overlay Comparison ────────────────────────────────────
fig, ax = plt.subplots(figsize=(7, 5))

ax.plot(alphas, etas_case1, 'o-', color="#2980b9", lw=2.0, ms=4, label=r"Case 1 ($\lambda_{ab}=\lambda_{cd}=1.0$)")
ax.plot(alphas, etas_case2, 's-', color="#8e44ad", lw=2.0, ms=4, label=r"Case 2 ($\lambda_{ab}(\omega_c,\omega_h), \lambda_{cd}=1.0$)")

ax.axhline(1 - 1/R_OMEGA_MAX, color='#e74c3c', linestyle=':', alpha=0.8, label=r"Harmonic bound $1 - 1/R_\omega \approx 0.8571$")

ax.set_xlabel(r"Anharmonicity Parameter $\alpha$ (lambda)")
ax.set_ylabel(r"Maximum Achievable Efficiency $\eta_{\mathrm{max}}$")
ax.set_title("Otto Cycle Maximum Efficiency Envelope Comparison", pad=10)
ax.set_xlim(0.0, 0.2)
ax.set_ylim(0.48, 0.88)
ax.legend(loc="lower left", framealpha=0.9, fontsize=9)

plt.tight_layout()
overlay_path = os.path.join(PLOTS_DIR, "cop_vs_alpha_optimized_overlay.png")
plt.savefig(overlay_path, dpi=300)
plt.close()
print(f"\nSaved direct overlay comparison to: {overlay_path}")
