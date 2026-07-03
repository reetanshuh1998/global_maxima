import os
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize
from physics_model_modified import compute_metrics, is_valid_physics, BOUNDS, R_OMEGA_MAX, R_BETA_MAX, stable_coth, lamba_ab as lamba_ab_fn

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PLOTS_DIR = os.path.join(SCRIPT_DIR, 'plots')
os.makedirs(PLOTS_DIR, exist_ok=True)

# Clean old plots
print("Cleaning old plots...")
for filename in os.listdir(PLOTS_DIR):
    file_path = os.path.join(PLOTS_DIR, filename)
    if os.path.isfile(file_path):
        os.remove(file_path)
        print(f"  Removed: {filename}")

BOUNDS_4D = BOUNDS[:4]

def get_lambdas(wc, wh, case):
    """Compute (l_ab, l_cd) for the given case from actual wc, wh values."""
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
    """Sample a random feasible starting point using the correct lambdas for the case."""
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
    """
    case = 1: lambda_ab=1,    lambda_cd=1    (Symmetric)
    case = 2: lambda_ab=f(w), lambda_cd=1    (Asymmetric, hot stroke)
    case = 3: lambda_ab=1,    lambda_cd=f(w) (Switched, cold stroke)
    """
    def obj(x):
        bc, bh, wc, wh = x
        l_ab, l_cd = get_lambdas(wc, wh, case)
        eta, _, _, _, _, ok = compute_metrics(bc, bh, wc, wh, alpha, l_ab, l_cd)
        return -eta if ok else 1e6

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

    for _ in range(80):   # 80 restarts for reliability
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

    return -best_val if best_x is not None else np.nan

# Alpha sweep (higher resolution)
alphas = np.linspace(0.0, 0.2, 21)
json_path = os.path.join(SCRIPT_DIR, "optimized_envelope_all_cases.json")
data = {}

if os.path.exists(json_path):
    print("Loading optimized results from JSON...")
    try:
        with open(json_path, "r") as f:
            data = json.load(f)
    except Exception as e:
        print(f"Error loading JSON: {e}")

case_configs = {
    1: (r"Case 1 ($\lambda_{ab}=\lambda_{cd}=1$)",             "#2980b9", "o-", 2.0),
    2: (r"Case 2 ($\lambda_{ab}=\frac{\omega_c^2+\omega_h^2}{2\omega_c\omega_h},\,\lambda_{cd}=1$)",
                                                                 "#8e44ad", "s-", 2.0),
    3: (r"Case 3 switched ($\lambda_{ab}=1,\,\lambda_{cd}=\frac{\omega_c^2+\omega_h^2}{2\omega_c\omega_h}$)",
                                                                 "#e67e22", "^-", 2.0),
    4: (r"Case 4 ($\lambda_{ab}=\lambda_{cd}=\frac{\omega_c^2+\omega_h^2}{2\omega_c\omega_h}$)",
                                                                 "#27ae60", "d-", 2.0),
}

results = {}
dirty = False

for case_id, (label, color, marker, lw) in case_configs.items():
    key = f"case{case_id}"
    if key in data and len(data[key]) == len(alphas):
        print(f"Case {case_id} loaded from cache.")
        results[case_id] = {"label": label, "color": color, "marker": marker,
                            "lw": lw, "etas": data[key]}
    else:
        print(f"\n=== Envelope Optimization Case {case_id} ===")
        etas = []
        for a in alphas:
            eta = optimize_for_alpha(a, case_id)
            etas.append(eta)
            print(f"  alpha={a:.3f}  eta={eta:.6f}")
        results[case_id] = {"label": label, "color": color, "marker": marker,
                            "lw": lw, "etas": etas}
        data[key] = etas
        dirty = True

if dirty:
    data["alphas"] = alphas.tolist()
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

for case_id, cfg in results.items():
    ax.plot(alphas, cfg["etas"], cfg["marker"], color=cfg["color"],
            lw=cfg["lw"], ms=4, label=cfg["label"])

ax.axhline(1 - 1/R_OMEGA_MAX, color='#e74c3c', linestyle=':', lw=1.2, alpha=0.8,
           label=r"Harmonic bound $1-1/R_\omega \approx 0.8571$")

ax.set_xlabel(r"Anharmonicity Parameter $\alpha$")
ax.set_ylabel(r"Maximum Achievable Efficiency $\eta_{\mathrm{max}}$")
ax.set_title(r"Otto Cycle Efficiency: Comparison of All Parameter Configurations", pad=12)
ax.set_xlim(0.0, 0.2)
ax.set_ylim(0.15, 0.90)

# Move legend outside to the right of the plot
ax.legend(bbox_to_anchor=(1.02, 1.0), loc="upper left", framealpha=0.9, fontsize=8.5)

# Annotate starting points
for case_id, cfg in results.items():
    eta0 = cfg["etas"][0]
    if not np.isnan(eta0):
        ax.annotate(f"$\\eta_0={eta0:.4f}$",
                    xy=(0.0, eta0),
                    xytext=(0.012, eta0 + (0.008 if case_id == 1 else -0.018 if case_id == 2 else -0.010 if case_id == 3 else -0.010)),
                    fontsize=7.5, color=cfg["color"],
                    arrowprops=dict(arrowstyle='->', color=cfg["color"], lw=0.8))

out_path = os.path.join(PLOTS_DIR, "eta_vs_alpha_all_cases.png")
plt.savefig(out_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"\nSaved: {out_path}")
