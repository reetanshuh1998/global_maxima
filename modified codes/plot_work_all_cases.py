import os
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize
from physics_model_modified import compute_metrics, is_valid_physics, BOUNDS, R_OMEGA_MAX, R_BETA_MAX, stable_coth

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PLOTS_DIR = os.path.join(SCRIPT_DIR, 'plots')
os.makedirs(PLOTS_DIR, exist_ok=True)

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

def sample_feasible(alpha, case, max_tries=50000):
    lo = np.array([b[0] for b in BOUNDS_4D])
    hi = np.array([b[1] for b in BOUNDS_4D])
    for _ in range(max_tries):
        x = np.random.uniform(lo, hi)
        bc, bh, wc, wh = x
        if is_valid_physics(bc, bh, wc, wh, alpha):
            l_ab, l_cd = get_lambdas(wc, wh, case)
            _, _, _, _, _, ok = compute_metrics(bc, bh, wc, wh, alpha, l_ab, l_cd)
            if ok:
                return x
    return None

def optimize_for_alpha(alpha, case):
    """Returns (eta_at_max_work, W_ext_max)"""
    def obj(x):
        bc, bh, wc, wh = x
        l_ab, l_cd = get_lambdas(wc, wh, case)
        eta, qh, qc, w_ext, _, ok = compute_metrics(bc, bh, wc, wh, alpha, l_ab, l_cd)
        return -w_ext if ok else 1e6

    cons = [
        {'type': 'ineq', 'fun': lambda x: x[3] - x[2] - 0.01},
        {'type': 'ineq', 'fun': lambda x: x[0] - x[1] - 0.01},
        {'type': 'ineq', 'fun': lambda x: x[0]*x[2] - x[1]*x[3] - 0.01},
        {'type': 'ineq', 'fun': lambda x: R_OMEGA_MAX - x[3]/x[2]},
        {'type': 'ineq', 'fun': lambda x: R_BETA_MAX - x[0]/x[1]},
        {'type': 'ineq', 'fun': lambda x: 0.1 - (3.0*alpha*stable_coth(x[0]*x[2]/2.0))/(4.0*x[2]**3)},
        {'type': 'ineq', 'fun': lambda x: 0.1 - (3.0*alpha*stable_coth(x[1]*x[3]/2.0))/(4.0*x[3]**3)},
    ]

    best_val = 1e10
    best_x   = None

    for _ in range(80):
        x0 = sample_feasible(alpha, case)
        if x0 is None:
            continue
        res = minimize(obj, x0, bounds=BOUNDS_4D, method='SLSQP', constraints=cons)
        if res.success and res.fun < best_val:
            if is_valid_physics(*res.x, alpha):
                bc, bh, wc, wh = res.x
                l_ab, l_cd = get_lambdas(wc, wh, case)
                _, _, _, _, _, ok = compute_metrics(bc, bh, wc, wh, alpha, l_ab, l_cd)
                if ok:
                    best_val = res.fun
                    best_x = res.x.copy()

    if best_x is not None:
        bc, bh, wc, wh = best_x
        l_ab, l_cd = get_lambdas(wc, wh, case)
        eta, qh, qc, w_ext, _, _ = compute_metrics(bc, bh, wc, wh, alpha, l_ab, l_cd)
        return eta, w_ext
    else:
        return np.nan, np.nan

# Discrete alphas requested by the user
alphas = [0.0, 0.05, 0.1, 0.15, 0.2]
json_path = os.path.join(SCRIPT_DIR, "optimized_work_all_cases.json")
data = {}

if os.path.exists(json_path):
    print("Loading optimized work results from JSON...")
    try:
        with open(json_path, "r") as f:
            data = json.load(f)
    except Exception as e:
        print(f"Error loading JSON: {e}")

case_configs = {
    1: (r"Case 1 ($\lambda_{ab}=\lambda_{cd}=1$)",             "#2980b9"),
    2: (r"Case 2 ($\lambda_{ab}=\frac{\omega_c^2+\omega_h^2}{2\omega_c\omega_h},\,\lambda_{cd}=1$)",
                                                                 "#8e44ad"),
    3: (r"Case 3 switched ($\lambda_{ab}=1,\,\lambda_{cd}=\frac{\omega_c^2+\omega_h^2}{2\omega_c\omega_h}$)",
                                                                 "#e67e22"),
    4: (r"Case 4 ($\lambda_{ab}=\lambda_{cd}=\frac{\omega_c^2+\omega_h^2}{2\omega_c\omega_h}$)",
                                                                 "#27ae60"),
}

results = {}
dirty = False

for case_id, (label, color) in case_configs.items():
    key = f"case{case_id}"
    if key in data and len(data[key]) == len(alphas):
        print(f"Case {case_id} work loaded from cache.")
        results[case_id] = {"label": label, "color": color, "works": data[key]}
    else:
        print(f"\n=== Work Optimization Case {case_id} ===")
        works = []
        for a in alphas:
            eta, w_ext = optimize_for_alpha(a, case_id)
            works.append(w_ext)
            print(f"  alpha={a:.2f}  eta={eta:.6f}  W_ext={w_ext:.6f}")
        results[case_id] = {"label": label, "color": color, "works": works}
        data[key] = works
        dirty = True

if dirty:
    data["alphas"] = alphas
    with open(json_path, "w") as f:
        json.dump(data, f, indent=4)

# ─── Plot Styling ────────────────────────────────────────────────────────────
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

fig, ax = plt.subplots(figsize=(9.5, 5.5))

x = np.arange(len(alphas))
width = 0.18

# Plot bars for each case
for i, case_id in enumerate([1, 2, 3, 4]):
    cfg = results[case_id]
    offset = (i - 1.5) * width
    ax.bar(x + offset, cfg["works"], width, label=cfg["label"], color=cfg["color"],
           alpha=0.9, edgecolor='black', linewidth=0.6)

# Set labels
ax.set_xlabel(r"Anharmonicity Parameter $\alpha$")
ax.set_ylabel(r"Work Output $W_{\mathrm{ext}} = Q_h + Q_c$")
# ax.set_title("Otto Cycle Work Output: Comparison of All Parameter Configurations", pad=12)
ax.set_xticks(x)
ax.set_xticklabels([f"{a:.2f}" for a in alphas])

# Move legend outside to the right of the plot
ax.legend(bbox_to_anchor=(1.02, 1.0), loc="upper left", framealpha=0.9, fontsize=8.5)

out_path = os.path.join(PLOTS_DIR, "work_vs_alpha_all_cases.png")
plt.savefig(out_path, dpi=300, bbox_inches='tight')
plt.savefig(out_path.replace(".png", ".pdf"), dpi=300, bbox_inches='tight')
plt.close()
print(f"\nSaved: {out_path} and PDF version.")
