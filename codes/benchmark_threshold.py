"""
Fast Time-to-Threshold Benchmark
==================================
Threshold: eta >= 0.99

Settings chosen for fast runtime:
  - MC: up to 200K samples per trial  (~10s per trial)
  - SLSQP: up to 100 restarts         (<1s per trial)
  - Bayesian Opt: 50 calls max        (~30s per trial, GP is O(N³))

3 trials per method (enough for error bars).
"""

import numpy as np
import time
import matplotlib.pyplot as plt
import warnings
from scipy.optimize import minimize
from skopt import gp_minimize
warnings.filterwarnings("ignore")

THRESHOLD  = 0.99
N_TRIALS   = 3

# ─────────────────────────────────────────────
def eta_fn(bc, bh, wc, wh, lam):
    if wc >= wh or bh >= bc or lam < 0:
        return 0.0
    if bc * wc <= bh * wh:
        return 0.0
    arg_h, arg_c = bh * wh / 2, bc * wc / 2
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

BOUNDS = [(0.5, 50), (0.01, 5), (0.1, 15), (1.0, 80), (0.0, 1.0)]

def sample_valid(rng):
    while True:
        bc  = rng.uniform(*BOUNDS[0])
        bh  = rng.uniform(BOUNDS[1][0], min(BOUNDS[1][1], bc - 0.01))
        wc  = rng.uniform(*BOUNDS[2])
        wh  = rng.uniform(max(BOUNDS[3][0], wc + 0.01), BOUNDS[3][1])
        lam = rng.uniform(*BOUNDS[4])
        if bc * wc > bh * wh:
            return bc, bh, wc, wh, lam

constraints = [
    {'type': 'ineq', 'fun': lambda x: x[3] - x[2] - 0.1},
    {'type': 'ineq', 'fun': lambda x: x[0] - x[1] - 0.01},
    {'type': 'ineq', 'fun': lambda x: x[0]*x[2] - x[1]*x[3] - 0.01},
]

# ─────────────────────────────────────────────
def run_mc(seed):
    rng = np.random.default_rng(seed)
    t0 = time.perf_counter()
    for count in range(1, 200_001):
        val = eta_fn(*sample_valid(rng))
        if val >= THRESHOLD:
            return count, time.perf_counter() - t0
    return None, time.perf_counter() - t0

def run_slsqp(seed):
    rng = np.random.default_rng(seed)
    count = [0]
    t0 = time.perf_counter()
    found = [None]

    def obj(x):
        count[0] += 1
        val = eta_fn(*x)
        if found[0] is None and val >= THRESHOLD:
            found[0] = (count[0], time.perf_counter() - t0)
        return -val

    for _ in range(100):
        x0 = list(sample_valid(rng))
        try:
            minimize(obj, x0, bounds=BOUNDS, constraints=constraints,
                     method='SLSQP', options={'maxiter': 150, 'ftol': 1e-9})
        except Exception:
            pass
        if found[0]:
            return found[0]
    return None, time.perf_counter() - t0

def run_bayes(seed):
    count = [0]
    t0 = time.perf_counter()
    found = [None]

    def obj(x):
        count[0] += 1
        val = eta_fn(*x)
        if found[0] is None and val >= THRESHOLD:
            found[0] = (count[0], time.perf_counter() - t0)
        return -val

    try:
        gp_minimize(obj, dimensions=BOUNDS, n_calls=50,
                    n_initial_points=10, acq_func='EI',
                    random_state=seed, noise=1e-10, verbose=False)
    except Exception:
        pass

    if found[0]:
        return found[0]
    return None, time.perf_counter() - t0

# ─────────────────────────────────────────────
results = {k: {'evals': [], 'time': []} for k in ['MC', 'SLSQP', 'Bayes']}

for trial in range(N_TRIALS):
    seed = 42 + trial * 13
    print(f"\n── Trial {trial+1}/{N_TRIALS} (seed={seed}) ──")

    print("  Monte Carlo (200K max) ...", end=' ', flush=True)
    e, t = run_mc(seed)
    print(f"evals={e}, time={t:.2f}s" if e else f"NOT FOUND in 200K, time={t:.1f}s")
    if e: results['MC']['evals'].append(e); results['MC']['time'].append(t)

    print("  SLSQP (100 restarts max)...", end=' ', flush=True)
    r = run_slsqp(seed)
    e, t = r
    print(f"evals={e}, time={t:.4f}s" if e else f"NOT FOUND, time={t:.1f}s")
    if e: results['SLSQP']['evals'].append(e); results['SLSQP']['time'].append(t)

    print("  Bayesian (50 calls max)...", end=' ', flush=True)
    r = run_bayes(seed)
    e, t = r
    print(f"evals={e}, time={t:.2f}s" if e else f"NOT FOUND in 50 calls, time={t:.1f}s")
    if e: results['Bayes']['evals'].append(e); results['Bayes']['time'].append(t)

# ─────────────────────────────────────────────
print("\nBuilding plot ...")

def stats(lst):
    if not lst: return float('nan'), float('nan')
    return float(np.mean(lst)), float(np.std(lst))

labels = ['Monte Carlo\n(EVT)', 'SLSQP', 'Bayesian\nOptimization']
keys   = ['MC', 'SLSQP', 'Bayes']
colors = ['steelblue', 'tomato', 'seagreen']

fig, axes = plt.subplots(1, 2, figsize=(13, 6))
fig.suptitle(f'Time-to-Threshold: First time each method reaches $\\eta \\geq {THRESHOLD}$\n'
             f'(mean ± std over {N_TRIALS} trials, log y-axis)', fontsize=13)

for ax, metric, ylabel, unit, fmt in zip(
        axes,
        ['evals', 'time'],
        ['# of $\\eta$ evaluations needed', 'Wall-clock time (seconds)'],
        ['calls', 's'],
        ['.0f', '.4f']):

    means = [stats(results[k][metric])[0] for k in keys]
    stds  = [stats(results[k][metric])[1] for k in keys]

    bars = ax.bar(labels, means, color=colors, edgecolor='black', linewidth=0.8,
                  yerr=stds, capsize=9, error_kw=dict(lw=2, ecolor='black', capthick=2))
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_yscale('log')
    ax.grid(axis='y', alpha=0.4)
    ax.set_title(ylabel, fontsize=11)

    for bar, m, s in zip(bars, means, stds):
        if not np.isnan(m):
            txt = f'{m:{fmt}} ± {s:{fmt}} {unit}'
            ax.text(bar.get_x() + bar.get_width()/2,
                    bar.get_height() * 1.5,
                    txt, ha='center', va='bottom', fontsize=9, fontweight='bold')

plt.tight_layout()
plt.savefig('plot_time_to_threshold.png', dpi=200, bbox_inches='tight')
print("Saved: plot_time_to_threshold.png")

print(f"\n{'Method':<22} {'Mean Evals':>12} {'Mean Time (s)':>15}")
print("-" * 52)
for k, lbl in zip(keys, ['Monte Carlo', 'SLSQP', 'Bayesian']):
    me, _ = stats(results[k]['evals'])
    mt, _ = stats(results[k]['time'])
    print(f"{lbl:<22} {me:>12.0f} {mt:>15.4f}")
