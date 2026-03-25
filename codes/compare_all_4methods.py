"""
4-Method Comparison on the Static Anharmonic Otto Cycle Problem
================================================================
Problem: maximize eta(beta_c, beta_h, omega_c, omega_h, lambda)
         subject to physical constraints.

Methods compared:
  1. Monte Carlo / EVT
  2. SLSQP (multi-start, gradient-based)
  3. Bayesian Optimization (Gaussian Process)
  4. ES-RL (Evolution Strategies policy gradient — NEW)

Metric: Time (s) and # evaluations to FIRST reach eta >= THRESHOLD = 0.96
        Averaged over N_TRIALS independent runs.
"""

import numpy as np
import time
import matplotlib.pyplot as plt
import warnings
from scipy.optimize import minimize
from skopt import gp_minimize
warnings.filterwarnings("ignore")

THRESHOLD  = 0.96
N_TRIALS   = 5


# ─────────────────────────────────────────────────────────────────────────────
def eta_fn(bc, bh, wc, wh, lam):
    """Exact efficiency from Eq.(8-11) of the paper."""
    if wc >= wh or bh >= bc or lam < 0: return 0.0
    if bc * wc <= bh * wh: return 0.0
    a_h, a_c = bh*wh/2, bc*wc/2
    if a_h > 300 or a_c > 300: return 0.0
    X = 1.0/np.tanh(a_h); Y = 1.0/np.tanh(a_c)
    Qh = (wh/2)*(X-Y) + (3*lam/(8*wh**2))*(X**2-Y**2)
    Qc = (wc/2)*(Y-X) + (3*lam/(8*wc**2))*(Y**2-X**2)
    W  = Qh + Qc
    if Qh <= 0 or W <= 0: return 0.0
    v = W / Qh
    return float(v) if 0 < v <= 1 else 0.0

BOUNDS = np.array([(0.5, 50), (0.01, 5), (0.1, 15), (1.0, 80), (0.0, 1.0)])
BOUNDS_SK = list(map(tuple, BOUNDS))

def sample_valid(rng):
    while True:
        bc  = rng.uniform(0.5, 50)
        bh  = rng.uniform(0.01, min(5, bc-0.01))
        wc  = rng.uniform(0.1, 15)
        wh  = rng.uniform(max(1.0, wc+0.01), 80)
        lam = rng.uniform(0, 1.0)
        if bc*wc > bh*wh: return np.array([bc, bh, wc, wh, lam])

constraints = [
    {'type':'ineq','fun': lambda x: x[3]-x[2]-0.1},
    {'type':'ineq','fun': lambda x: x[0]-x[1]-0.01},
    {'type':'ineq','fun': lambda x: x[0]*x[2]-x[1]*x[3]-0.01},
]


# ─────────────────────────────────────────────────────────────────────────────
# Method 1: Monte Carlo / EVT
def run_mc(seed, max_evals=500_000):
    rng = np.random.default_rng(seed)
    t0  = time.perf_counter()
    for count in range(1, max_evals+1):
        if eta_fn(*sample_valid(rng)) >= THRESHOLD:
            return count, time.perf_counter()-t0
    return None, time.perf_counter()-t0


# ─────────────────────────────────────────────────────────────────────────────
# Method 2: SLSQP multi-restart
def run_slsqp(seed):
    rng = np.random.default_rng(seed)
    count = [0]; found = [None]; t0 = time.perf_counter()
    def obj(x):
        v = eta_fn(*x); count[0] += 1
        if v >= THRESHOLD and found[0] is None:
            found[0] = (count[0], time.perf_counter()-t0)
        return -v
    for _ in range(200):
        x0 = sample_valid(rng)
        try:
            minimize(obj, x0, bounds=BOUNDS_SK, constraints=constraints,
                     method='SLSQP', options={'maxiter':200, 'ftol':1e-9})
        except Exception: pass
        if found[0]: return found[0]
    return None, time.perf_counter()-t0


# ─────────────────────────────────────────────────────────────────────────────
# Method 3: Bayesian Optimization (GP, skopt)
def run_bayes(seed):
    count = [0]; found = [None]; t0 = time.perf_counter()
    def obj(x):
        v = eta_fn(*x); count[0] += 1
        if v >= THRESHOLD and found[0] is None:
            found[0] = (count[0], time.perf_counter()-t0)
        return -v
    try:
        gp_minimize(obj, dimensions=BOUNDS_SK, n_calls=80,
                    n_initial_points=15, acq_func='EI',
                    random_state=seed, noise=1e-10, verbose=False)
    except Exception: pass
    return (found[0] if found[0] else (None, time.perf_counter()-t0))


# ─────────────────────────────────────────────────────────────────────────────
# Method 4: ES-RL (Evolution Strategies on the 5-parameter space)
#
# The "policy" here is a 5D mean vector μ ∈ R^5 (one per parameter, in [0,1]).
# We sample perturbations ε ~ N(0, σ²I), evaluate eta for μ+ε,
# and do Adam gradient ascent on μ.
# This is essentially CMA-ES / Natural ES applied to find theta → params.

def run_rl_es(seed, n_episodes=300, batch_size=16):
    rng = np.random.default_rng(seed)
    t0  = time.perf_counter()
    
    # Encode params in normalized [0,1]^5 space
    def decode(x_norm):
        """Map normalized [0,1]^5 → physical parameters."""
        bc  = BOUNDS[0,0] + x_norm[0]*(BOUNDS[0,1]-BOUNDS[0,0])
        bh  = BOUNDS[1,0] + x_norm[1]*(BOUNDS[1,1]-BOUNDS[1,0])
        # Enforce bh < bc
        bh  = min(bh, bc - 0.01)
        wc  = BOUNDS[2,0] + x_norm[2]*(BOUNDS[2,1]-BOUNDS[2,0])
        wh  = BOUNDS[3,0] + x_norm[3]*(BOUNDS[3,1]-BOUNDS[3,0])
        # Enforce wh > wc
        wh  = max(wh, wc + 0.1)
        lam = BOUNDS[4,0] + x_norm[4]*(BOUNDS[4,1]-BOUNDS[4,0])
        return np.array([bc, bh, wc, wh, lam])

    def reward(x_norm):
        params = decode(np.clip(x_norm, 0, 1))
        # Extra constraint: bc*wc > bh*wh
        if params[0]*params[2] <= params[1]*params[3]: return 0.0
        return eta_fn(*params)

    # Initialize mean in a promising area
    # (high bc/bh ratio, low wc, high wh, low lambda → should give high eta)
    mu = np.array([0.7, 0.01, 0.0, 0.98, 0.0])  # normalized starting point
    sigma = 0.2
    # Adam state
    m = np.zeros(5); v = np.zeros(5); step_t = 0

    evals = 0
    found = None
    best_r = 0.0

    eps_arr = rng.standard_normal((n_episodes, batch_size//2, 5)) * sigma

    for ep in range(n_episodes):
        half = batch_size // 2
        eps  = eps_arr[ep]
        r_pos = np.zeros(half); r_neg = np.zeros(half)

        for i in range(half):
            for sign, arr in [(+1, r_pos), (-1, r_neg)]:
                r = reward(mu + sign*eps[i])
                arr[i] = r
                evals += 1
                if r > best_r: best_r = r
                if r >= THRESHOLD and found is None:
                    found = (evals, time.perf_counter()-t0)

        if found: break

        # ES antithetic gradient
        grad = np.sum(((r_pos-r_neg)[:,None]*eps), axis=0)/(2*sigma*half)

        # Adam ascent
        step_t += 1
        b1, b2, e = 0.9, 0.999, 1e-8
        m = b1*m + (1-b1)*grad; v = b2*v + (1-b2)*grad**2
        mh = m/(1-b1**step_t); vh = v/(1-b2**step_t)
        mu = np.clip(mu + 0.05*mh/(np.sqrt(vh)+e), 0, 1)

    return found if found else (None, time.perf_counter()-t0)


# ─────────────────────────────────────────────────────────────────────────────
# Run ALL methods
print(f"Threshold η ≥ {THRESHOLD} | {N_TRIALS} independent trials each\n")
methods = ['Monte Carlo\n(EVT)', 'SLSQP', 'Bayesian\nOpt.', 'ES-RL\n(NEW)']
runners = [run_mc, run_slsqp, run_bayes, run_rl_es]
colors  = ['steelblue', 'tomato', 'seagreen', 'darkorchid']

results = {m: {'evals': [], 'time': [], 'success': 0} for m in methods}

print(f"{'Trial':>6} {'MC evals':>10} {'MC time':>9} {'SLSQP ev':>10} {'SLSQP t':>9} "
      f"{'Bayes ev':>10} {'Bayes t':>9} {'RL ev':>8} {'RL t':>8}")
print("-" * 85)

for trial in range(N_TRIALS):
    seed = 42 + trial * 17
    row  = [f"{trial+1:>6}"]
    for m, runner in zip(methods, runners):
        t0 = time.perf_counter()
        result = runner(seed)
        # runner may return (evals, time) or (None, time) tuple/pair
        if isinstance(result, tuple):
            e, t = result
        else:
            e, t = result, time.perf_counter()-t0
        if e is not None:
            results[m]['evals'].append(e); results[m]['time'].append(t); results[m]['success'] += 1
            row.append(f"{e:>10} {t:>9.3f}")
        else:
            row.append(f"{'FAIL':>10} {t:>9.1f}")
    print(' '.join(row))

# ─────────────────────────────────────────────────────────────────────────────
print(f"\n{'Method':<22} {'Success':>8} {'Mean Evals':>12} {'Mean Time(s)':>13}")
print("-" * 58)
for m in methods:
    d = results[m]
    me = np.mean(d['evals']) if d['evals'] else float('nan')
    mt = np.mean(d['time'])  if d['time']  else float('nan')
    m_clean = m.replace('\n', ' ')
    print(f"{m_clean:<22} {d['success']}/{N_TRIALS}  {me:>12.0f} {mt:>13.4f}")

# ─────────────────────────────────────────────────────────────────────────────
# Plot
fig, axes = plt.subplots(1, 2, figsize=(13, 6))
fig.suptitle(f'4-Method Comparison: Time to Reach $\\eta \\geq {THRESHOLD}$\n'
             f'({N_TRIALS} trials each, log y-axis | old static 5-parameter problem)',
             fontsize=13)

labels_clean = ['Monte Carlo\n(EVT)', 'SLSQP\n(multi-start)', 'Bayesian\nOpt. (GP)', 'ES-RL\n(NEW)']

for ax, metric, ylabel, unit in zip(
        axes,
        ['evals', 'time'],
        [r'# of $\eta$ evaluations needed', 'Wall-clock time (seconds)'],
        ['evals', 's']):

    means = [np.mean(results[m][metric]) if results[m][metric] else float('nan')
             for m in methods]
    stds  = [np.std(results[m][metric])  if len(results[m][metric]) > 1 else 0.0
             for m in methods]
    success = [results[m]['success'] for m in methods]

    bars = ax.bar(labels_clean, means, color=colors, edgecolor='black', linewidth=0.8,
                  yerr=stds, capsize=9, error_kw=dict(lw=2, ecolor='black'))
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_yscale('log')
    ax.grid(axis='y', alpha=0.4)
    ax.set_title(ylabel, fontsize=10)

    for bar, m, s, sc in zip(bars, means, stds, success):
        if not np.isnan(m):
            fmt = '.0f' if metric == 'evals' else '.3f'
            label = f'{m:{fmt}}\n({sc}/{N_TRIALS} ✓)'
            ax.text(bar.get_x()+bar.get_width()/2,
                    bar.get_height()*1.4,
                    label, ha='center', fontsize=8.5, fontweight='bold')

plt.tight_layout()
plt.savefig('plot_4method_threshold096.png', dpi=200, bbox_inches='tight')
print(f"\nSaved: plot_4method_threshold096.png")
