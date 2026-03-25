"""
Comparison of 3 optimization strategies for finding global max of eta:
  1. Monte Carlo / EVT  — brute force random sampling (1M points)
  2. SLSQP              — multi-start gradient-based optimizer
  3. Bayesian Optimization — surrogate-model (Gaussian Process) based

Final plot: "best eta found" vs "number of eta evaluations" for all 3 methods.
This is the Gold Standard convergence comparison.
"""

import numpy as np
import matplotlib.pyplot as plt
import warnings
from scipy.optimize import minimize
from skopt import gp_minimize
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# Shared eta function
# ─────────────────────────────────────────────
def eta_fn(bc, bh, wc, wh, lam):
    """Returns eta in [0,1], or 0.0 if constraints violated."""
    if wc >= wh or bh >= bc or lam < 0:
        return 0.0
    if bc * wc <= bh * wh:
        return 0.0
    arg_h = bh * wh / 2
    arg_c = bc * wc / 2
    if arg_h > 300 or arg_c > 300:
        return 0.0
    X = 1.0 / np.tanh(arg_h)
    Y = 1.0 / np.tanh(arg_c)
    Q_h = (wh / 2) * (X - Y) + (3 * lam / (8 * wh**2)) * (X**2 - Y**2)
    Q_c = (wc / 2) * (Y - X) + (3 * lam / (8 * wc**2)) * (Y**2 - X**2)
    W_ext = Q_h + Q_c
    if Q_h <= 0 or W_ext <= 0:
        return 0.0
    v = W_ext / Q_h
    return float(v) if 0 < v <= 1 else 0.0

# Parameter bounds: [beta_c, beta_h, omega_c, omega_h, lambda]
BOUNDS = [(0.5, 50), (0.01, 5), (0.1, 15), (1.0, 80), (0.0, 1.0)]

def sample_valid():
    """Draw one random valid parameter set."""
    while True:
        bc  = np.random.uniform(*BOUNDS[0])
        bh  = np.random.uniform(BOUNDS[1][0], min(BOUNDS[1][1], bc - 0.01))
        wc  = np.random.uniform(*BOUNDS[2])
        wh  = np.random.uniform(max(BOUNDS[3][0], wc + 0.01), BOUNDS[3][1])
        lam = np.random.uniform(*BOUNDS[4])
        if bc * wc > bh * wh:
            return bc, bh, wc, wh, lam

# ─────────────────────────────────────────────
# METHOD 1: Monte Carlo / EVT  (1 Million samples)
# ─────────────────────────────────────────────
print("Running Method 1: Monte Carlo with 1,000,000 samples ...")
np.random.seed(42)
N_MC = 1_000_000
mc_best = []
current_best = 0.0
RECORD_EVERY = 1000  # record running max every 1000 samples

for i in range(N_MC):
    bc, bh, wc, wh, lam = sample_valid()
    val = eta_fn(bc, bh, wc, wh, lam)
    if val > current_best:
        current_best = val
    if (i + 1) % RECORD_EVERY == 0:
        mc_best.append(current_best)

mc_calls = [(i + 1) * RECORD_EVERY for i in range(len(mc_best))]
print(f"  MC best eta = {current_best:.6f}  ({N_MC:,} evaluations)")

# ─────────────────────────────────────────────
# METHOD 2: SLSQP  (50 random restarts)
# ─────────────────────────────────────────────
print("Running Method 2: SLSQP with 50 random restarts ...")
np.random.seed(0)

slsqp_evals = [0]
slsqp_best_history = [0.0]
slsqp_global_best = 0.0
slsqp_eval_count = 0

def objective_slsqp(x):
    global slsqp_eval_count, slsqp_global_best
    slsqp_eval_count += 1
    val = eta_fn(*x)
    if val > slsqp_global_best:
        slsqp_global_best = val
    slsqp_evals.append(slsqp_eval_count)
    slsqp_best_history.append(slsqp_global_best)
    return -val  # minimize negative

constraints = [
    {'type': 'ineq', 'fun': lambda x: x[3] - x[2] - 0.1},     # wh > wc
    {'type': 'ineq', 'fun': lambda x: x[0] - x[1] - 0.01},    # bc > bh
    {'type': 'ineq', 'fun': lambda x: x[0]*x[2] - x[1]*x[3] - 0.01},  # bc*wc > bh*wh
]

for restart in range(50):
    bc, bh, wc, wh, lam = sample_valid()
    x0 = [bc, bh, wc, wh, lam]
    try:
        minimize(objective_slsqp, x0,
                 bounds=BOUNDS, constraints=constraints,
                 method='SLSQP', options={'maxiter': 200, 'ftol': 1e-9})
    except Exception:
        pass

print(f"  SLSQP best eta = {slsqp_global_best:.6f}  ({slsqp_eval_count:,} evaluations)")

# ─────────────────────────────────────────────
# METHOD 3: Bayesian Optimization (600 calls)
# ─────────────────────────────────────────────
print("Running Method 3: Bayesian Optimization (200 calls) ...")
np.random.seed(7)

bayes_best_history = []

def objective_bayes(x):
    """Skopt minimizes, so we negate eta."""
    return -eta_fn(*x)

result = gp_minimize(
    objective_bayes,
    dimensions=BOUNDS,
    n_calls=200,
    n_initial_points=20,   # first 20 are random exploration
    acq_func='EI',          # Expected Improvement acquisition function
    random_state=7,
    noise=1e-10,
    verbose=False
)

# Build running-best history from skopt results
current_best_b = 0.0
for val in result.func_vals:
    if -val > current_best_b:
        current_best_b = -val
    bayes_best_history.append(current_best_b)

bayes_calls = list(range(1, len(bayes_best_history) + 1))
print(f"  Bayesian best eta = {-result.fun:.6f}  ({len(result.func_vals):,} evaluations)")

# ─────────────────────────────────────────────
# FINAL COMPARISON PLOT
# ─────────────────────────────────────────────
print("Plotting comparison ...")

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle('Optimization Method Comparison: Finding Global Maximum of $\\eta$', fontsize=14)

# ── Panel A: Convergence curves (best eta vs evaluations) ──────────────────
ax = axes[0]
ax.plot(mc_calls, mc_best,         color='steelblue',  lw=2, label='Monte Carlo / EVT (1M samples)')
ax.plot(slsqp_evals, slsqp_best_history, color='tomato', lw=2, label='SLSQP (50 restarts)')
ax.plot(bayes_calls, bayes_best_history, color='seagreen', lw=2.5, label='Bayesian Optimization (200 calls)')
ax.set_xlabel('Number of $\\eta$ Evaluations', fontsize=12)
ax.set_ylabel('Best $\\eta$ Found (Running Maximum)', fontsize=12)
ax.set_ylim(0.5, 1.05)
ax.set_xscale('log')
ax.axhline(max(slsqp_global_best, -result.fun, current_best), ls='--',
           color='gray', lw=1, label='True Best Observed')
ax.legend(fontsize=10); ax.grid(True, alpha=0.4)
ax.set_title('Convergence: Best $\\eta$ Found vs Evaluations\n(log scale — leftward = more efficient)')

# ── Panel B: Bar chart — final best eta & total evaluations ────────────────
ax = axes[1]
methods  = ['Monte Carlo\n(EVT, 1M)', 'SLSQP\n(50 restarts)', 'Bayesian\nOptimization']
best_vals = [current_best, slsqp_global_best, -result.fun]
total_evals = [N_MC, slsqp_eval_count, len(result.func_vals)]
colors = ['steelblue', 'tomato', 'seagreen']

bars = ax.bar(methods, best_vals, color=colors, edgecolor='black', linewidth=0.8)
ax.set_ylim(0, 1.1)
ax.set_ylabel('Best $\\eta$ Found', fontsize=12)
ax.set_title('Final Best $\\eta$ and Total Evaluations Used')
ax.grid(axis='y', alpha=0.4)

for bar, bv, ev in zip(bars, best_vals, total_evals):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
            f'$\\eta$={bv:.5f}\n({ev:,} calls)', ha='center', va='bottom', fontsize=9)

plt.tight_layout()
plt.savefig('plot_optimizer_comparison.png', dpi=200, bbox_inches='tight')
print("Saved: plot_optimizer_comparison.png")

print("\n=== FINAL SUMMARY ===")
print(f"{'Method':<30} {'Best eta':>10} {'# Evaluations':>15}")
print("-" * 58)
print(f"{'Monte Carlo / EVT':<30} {current_best:>10.6f} {N_MC:>15,}")
print(f"{'SLSQP (50 restarts)':<30} {slsqp_global_best:>10.6f} {slsqp_eval_count:>15,}")
print(f"{'Bayesian Optimization':<30} {-result.fun:>10.6f} {len(result.func_vals):>15,}")
