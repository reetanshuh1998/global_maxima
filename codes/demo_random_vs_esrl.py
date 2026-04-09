"""
demo_random_vs_esrl.py
======================
Demonstrates Random Search and ES-RL (Evolution Strategy + Adam)
on a deliberately complex single-variable function with many local maxima.

The two algorithms are run for the same evaluation budget, across
multiple random seeds, so that their behaviour can be compared
statistically rather than just on one lucky run.

Saved figures (in final_result/plots/):
  demo_function.png          – the test function annotated with local maxima
  demo_convergence.png       – best-so-far vs evaluations (median ± IQR)
  demo_search_paths.png      – where each algorithm actually looked
  demo_distribution.png      – violin of final best-f found across seeds
"""

import signal
signal.signal(signal.SIGINT, signal.SIG_IGN)

import os
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.optimize import minimize_scalar

# ── Output directory ──────────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PDIR = os.path.join(SCRIPT_DIR, '..', 'final_result', 'plots')
os.makedirs(PDIR, exist_ok=True)

# ══════════════════════════════════════════════════════════════════════════════
# 1.  Test function  — complex, multi-modal, single variable
# ══════════════════════════════════════════════════════════════════════════════
X_MIN, X_MAX = 0.0, 10.0   # search domain

def f(x):
    """
    Complex multi-modal test function.

    Composed of:
      • Two oscillatory sine terms with incommensurate frequencies → many local peaks
      • A slow cosine modulation → envelope that varies across the domain
      • A mild quadratic tilt → breaks perfect symmetry, hides the true max

    True maximum is near x ≈ 7.9 (found numerically below).
    """
    return (
        2.0 * np.sin(2.3 * x)
        + 1.5 * np.sin(4.1 * x + 0.7)
        + 1.0 * np.cos(1.3 * x)
        + 0.6 * np.sin(7.7 * x + 1.2)
        - 0.05 * (x - 5.0)**2          # slight tilt — penalises extremes
        + 3.0                           # vertical shift → keep f > 0 mostly
    )

# Find the true maximum numerically (fine grid + local polish)
x_grid   = np.linspace(X_MIN, X_MAX, 100_000)
f_grid   = f(x_grid)
x_true   = x_grid[np.argmax(f_grid)]
f_true   = f(x_true)
# Refine with scipy
res = minimize_scalar(lambda x: -f(x),
                      bounds=(max(X_MIN, x_true-0.1), min(X_MAX, x_true+0.1)),
                      method='bounded')
x_true = -res.x if res.x < 0 else res.x   # fix sign artefact
x_true = float(minimize_scalar(lambda x: -f(x),
                               bounds=(max(X_MIN, x_true-0.2), min(X_MAX, x_true+0.2)),
                               method='bounded').x)
f_true = float(f(x_true))
print(f"True maximum: f({x_true:.6f}) = {f_true:.6f}")

# ══════════════════════════════════════════════════════════════════════════════
# 2.  Algorithm implementations
# ══════════════════════════════════════════════════════════════════════════════
BUDGET  = 200    # total function evaluations per run
N_SEEDS = 30     # independent trials per algorithm

# ── 2a. Random Search ─────────────────────────────────────────────────────────
def run_random(seed):
    """
    Pure random search:
    At each step draw x ~ Uniform[X_MIN, X_MAX],
    evaluate f(x), and keep the best seen so far.

    Returns
    -------
    hist    : list of best-f after each evaluation
    best_x  : x value achieving best_f
    all_x   : every x that was evaluated (for visualisation)
    """
    rng    = np.random.default_rng(seed)
    best_f = -np.inf;  best_x = None
    hist   = [];       all_x  = []
    for _ in range(BUDGET):
        x = rng.uniform(X_MIN, X_MAX)
        v = float(f(x))
        all_x.append(x)
        if v > best_f:
            best_f = v;  best_x = x
        hist.append(best_f)
    return dict(hist=hist, best_f=best_f, best_x=best_x, all_x=all_x)

# ── 2b. ES-RL (Evolution Strategy + Adam) ────────────────────────────────────
def run_esrl(seed):
    """
    Evolution Strategy with antithetic sampling and Adam gradient update.

    State
    ─────
    μ  : current mean ("best guess" of the optimum location), in [0, 1]
         (normalised; un-normalised value = X_MIN + μ*(X_MAX-X_MIN))
    σ  : exploration spread (fixed here)
    m, v, t  : Adam first/second moment accumulators and timestep

    One step
    ─────────
    1. Sample H perturbation directions εᵢ ~ N(0,1)
    2. Evaluate antithetic pairs:
         f_plus[i]  = f(μ + σ·εᵢ)
         f_minus[i] = f(μ - σ·εᵢ)
    3. Gradient estimate (REINFORCE / ES):
         ĝ = (1 / 2Hσ) · Σᵢ (f_plus[i] - f_minus[i]) · εᵢ
    4. Adam update:
         m ← β₁·m + (1-β₁)·ĝ
         v ← β₂·v + (1-β₂)·ĝ²
         m̂ = m/(1-β₁ᵗ),  v̂ = v/(1-β₂ᵗ)
         μ ← clip(μ + α · m̂/√(v̂+ε), 0, 1)

    Returns
    -------
    hist      : list of best-f after each evaluation
    best_x    : x achieving best_f
    mu_hist   : trajectory of μ (un-normalised) across iterations
    all_x     : every x evaluated
    """
    rng = np.random.default_rng(seed)

    # Hyper-parameters
    H    = 8      # antithetic pairs per step (costs 2H evals)
    sig  = 0.12   # exploration spread (normalised units)
    lr   = 0.05   # Adam learning rate
    b1, b2, eps_adam = 0.9, 0.999, 1e-8

    # Initialise
    mu   = rng.uniform(0.05, 0.95)   # random start in (0,1)
    m_adam = 0.0;  v_adam = 0.0;  t_adam = 0

    best_f = -np.inf;  best_x = None
    hist   = [];  all_x = [];  mu_hist = []

    ev = 0
    while ev < BUDGET:
        # --- sample perturbations ---
        eps_vec = rng.standard_normal(H)     # H directions
        f_plus  = np.zeros(H)
        f_minus = np.zeros(H)

        for i in range(H):
            if ev >= BUDGET: break
            # antithetic pair
            for sign, arr in [(+1, f_plus), (-1, f_minus)]:
                if ev >= BUDGET: break
                x_norm = np.clip(mu + sign * sig * eps_vec[i], 0.0, 1.0)
                x_real = X_MIN + x_norm * (X_MAX - X_MIN)
                v = float(f(x_real));  ev += 1
                arr[i] = v;  all_x.append(x_real)
                if v > best_f:
                    best_f = v;  best_x = x_real
            hist.append(best_f)

        # --- gradient estimate ---
        g_hat = np.mean((f_plus - f_minus) * eps_vec) / (2.0 * sig)

        # --- Adam update ---
        t_adam += 1
        m_adam  = b1 * m_adam + (1 - b1) * g_hat
        v_adam  = b2 * v_adam + (1 - b2) * g_hat**2
        m_hat   = m_adam / (1 - b1**t_adam)
        v_hat   = v_adam / (1 - b2**t_adam)
        mu      = np.clip(mu + lr * m_hat / (np.sqrt(v_hat) + eps_adam), 0.0, 1.0)

        mu_hist.append(X_MIN + mu * (X_MAX - X_MIN))

    return dict(hist=hist, best_f=best_f, best_x=best_x,
                mu_hist=mu_hist, all_x=all_x)

# ══════════════════════════════════════════════════════════════════════════════
# 3.  Run both algorithms
# ══════════════════════════════════════════════════════════════════════════════
print(f"\nRunning Random Search ({N_SEEDS} seeds × {BUDGET} evals) …")
rand_results = [run_random(s) for s in range(N_SEEDS)]
rand_best    = [r['best_f'] for r in rand_results]
print(f"  Best:   {max(rand_best):.6f}")
print(f"  Median: {np.median(rand_best):.6f}")
print(f"  Std:    {np.std(rand_best):.6f}")

print(f"\nRunning ES-RL ({N_SEEDS} seeds × {BUDGET} evals) …")
esrl_results = [run_esrl(s) for s in range(N_SEEDS)]
esrl_best    = [r['best_f'] for r in esrl_results]
print(f"  Best:   {max(esrl_best):.6f}")
print(f"  Median: {np.median(esrl_best):.6f}")
print(f"  Std:    {np.std(esrl_best):.6f}")

print(f"\nTrue maximum f({x_true:.4f}) = {f_true:.6f}")

# ══════════════════════════════════════════════════════════════════════════════
# 4.  Plotting
# ══════════════════════════════════════════════════════════════════════════════
plt.rcParams.update({
    'font.family': 'serif', 'font.size': 11,
    'axes.linewidth': 1.2, 'axes.grid': True,
    'grid.alpha': 0.3, 'legend.framealpha': 0.9,
})

C_RAND = '#e74c3c'    # red for random
C_ESRL = '#2980b9'    # blue for ES-RL
C_TRUE = '#27ae60'    # green for true max

x_plot = np.linspace(X_MIN, X_MAX, 2000)
f_plot = f(x_plot)

# ── Fig 1: The function ────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(11, 5))
ax.plot(x_plot, f_plot, 'k-', lw=2.0, label='f(x)', zorder=3)
ax.axvline(x_true, ls='--', color=C_TRUE, lw=1.8,
           label=rf'True max: $x^*={x_true:.3f}$, $f^*={f_true:.3f}$')
ax.scatter([x_true], [f_true], color=C_TRUE, s=120, zorder=6, marker='*')

# annotate all local maxima (peaks above the median)
from scipy.signal import argrelextrema
peaks = argrelextrema(f_plot, np.greater, order=20)[0]
for p in peaks:
    if f_plot[p] > np.median(f_plot) + 0.3:
        ax.annotate('', xy=(x_plot[p], f_plot[p]),
                    xytext=(x_plot[p], f_plot[p] + 0.5),
                    arrowprops=dict(arrowstyle='->', color='grey', lw=1.0))

ax.fill_between(x_plot, f_plot, alpha=0.07, color='k')
ax.set_xlabel('x', fontsize=13)
ax.set_ylabel('f(x)', fontsize=13)
ax.set_title('Test Function: Complex Multi-Modal Single-Variable Landscape\n'
             r'$f(x) = 2\sin(2.3x) + 1.5\sin(4.1x+0.7) + \cos(1.3x) '
             r'+ 0.6\sin(7.7x+1.2) - 0.05(x-5)^2 + 3$',
             fontsize=11)
ax.legend(fontsize=10)
fig.tight_layout()
fig.savefig(os.path.join(PDIR, 'demo_function.png'), dpi=200, bbox_inches='tight')
plt.close(fig)
print("\nSaved: demo_function.png")

# ── Fig 2: Convergence curves ─────────────────────────────────────────────────
def pad(h, n):
    h = list(h); return (h + [h[-1]]*(n-len(h)))[:n]

fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
fig.suptitle(
    f'Convergence Comparison: Random Search vs ES-RL\n'
    f'({N_SEEDS} independent seeds × {BUDGET} evaluations each)',
    fontsize=12
)

# Left: full budget
ax = axes[0]
for res_list, color, label in [
    (rand_results, C_RAND, 'Random Search'),
    (esrl_results, C_ESRL, 'ES-RL'),
]:
    maxl = max(len(r['hist']) for r in res_list)
    H    = np.array([pad(r['hist'], maxl) for r in res_list])
    ev   = np.arange(maxl)
    med  = np.median(H, 0)
    q25  = np.percentile(H, 25, 0)
    q75  = np.percentile(H, 75, 0)
    ax.plot(ev, med, color=color, lw=2.2, label=label)
    ax.fill_between(ev, q25, q75, color=color, alpha=0.2)

ax.axhline(f_true, ls='--', color=C_TRUE, lw=1.8,
           label=f'True maximum f* = {f_true:.3f}')
ax.set_xlabel('Evaluations', fontsize=12)
ax.set_ylabel('Best f found (so far)', fontsize=12)
ax.set_title('All evaluations', fontsize=11)
ax.legend(fontsize=10)

# Right: zoom first 60 evals
ax = axes[1]
ZOOM = 60
for res_list, color, label in [
    (rand_results, C_RAND, 'Random Search'),
    (esrl_results, C_ESRL, 'ES-RL'),
]:
    H   = np.array([pad(r['hist'], ZOOM) for r in res_list])
    med = np.median(H, 0)
    q25 = np.percentile(H, 25, 0)
    q75 = np.percentile(H, 75, 0)
    ax.plot(np.arange(ZOOM), med, color=color, lw=2.2, label=label)
    ax.fill_between(np.arange(ZOOM), q25, q75, color=color, alpha=0.2)
ax.axhline(f_true, ls='--', color=C_TRUE, lw=1.8, label=f'f* = {f_true:.3f}')
ax.set_xlabel('Evaluations', fontsize=12)
ax.set_ylabel('Best f found (so far)', fontsize=12)
ax.set_title(f'Zoom: first {ZOOM} evaluations', fontsize=11)
ax.legend(fontsize=10)

fig.tight_layout()
fig.savefig(os.path.join(PDIR, 'demo_convergence.png'), dpi=200, bbox_inches='tight')
plt.close(fig)
print("Saved: demo_convergence.png")

# ── Fig 3: Search paths (one representative seed) ─────────────────────────────
SHOW_SEED = 5   # pick a visually clear seed
rand_r = rand_results[SHOW_SEED]
esrl_r = esrl_results[SHOW_SEED]

fig, axes = plt.subplots(2, 1, figsize=(13, 9), sharex=True)
fig.suptitle(
    f'Where Each Algorithm Searched  (seed={SHOW_SEED}, {BUDGET} evaluations)\n'
    'Colour = evaluation order (dark=early, bright=late)',
    fontsize=12
)

for ax, res, color, label in [
    (axes[0], rand_r, C_RAND, 'Random Search'),
    (axes[1], esrl_r, C_ESRL, 'ES-RL'),
]:
    # plot function
    ax.plot(x_plot, f_plot, 'k-', lw=1.5, alpha=0.5, zorder=1)
    ax.axvline(x_true, ls='--', color=C_TRUE, lw=1.5, alpha=0.8)

    xs   = np.array(res['all_x'])
    ys   = np.array([f(x) for x in xs])
    cmap = plt.cm.YlOrRd if label == 'Random Search' else plt.cm.YlGnBu
    sc   = ax.scatter(xs, ys, c=np.arange(len(xs)), cmap=cmap,
                      s=20, alpha=0.7, zorder=4)
    plt.colorbar(sc, ax=ax, label='Evaluation index')

    # mark best found
    ax.scatter([res['best_x']], [res['best_f']], marker='*',
               color=color, s=350, edgecolor='k', linewidth=1.2, zorder=6,
               label=f'Best found: x={res["best_x"]:.3f}, f={res["best_f"]:.3f}')
    ax.scatter([x_true], [f_true], marker='*', color=C_TRUE, s=350,
               edgecolor='k', linewidth=1.2, zorder=7,
               label=f'True max: x={x_true:.3f}, f={f_true:.3f}')

    # For ES-RL: overlay mean trajectory
    if 'mu_hist' in res:
        mu_h = np.array(res['mu_hist'])
        mu_f = np.array([f(x) for x in mu_h])
        ax.plot(mu_h, mu_f, color='navy', lw=1.5, ls='-', alpha=0.6, zorder=5,
                label='ES-RL mean (μ) trajectory')

    ax.set_ylabel('f(x)', fontsize=11)
    ax.set_title(label, fontsize=12, color=color, fontweight='bold')
    ax.legend(fontsize=9, loc='upper left')

axes[1].set_xlabel('x', fontsize=12)
fig.tight_layout()
fig.savefig(os.path.join(PDIR, 'demo_search_paths.png'), dpi=200, bbox_inches='tight')
plt.close(fig)
print("Saved: demo_search_paths.png")

# ── Fig 4: Distribution of final best (violin + strip) ───────────────────────
fig, ax = plt.subplots(figsize=(8, 6))
fig.suptitle(
    f'Distribution of Best f Found  ({N_SEEDS} seeds × {BUDGET} evals)\n'
    f'True maximum f* = {f_true:.4f}   (dashed green)',
    fontsize=12
)
data   = [rand_best, esrl_best]
labels = ['Random\nSearch', 'ES-RL']
colors = [C_RAND, C_ESRL]
parts  = ax.violinplot(data, positions=[0, 1], showmedians=True, showextrema=True,
                        widths=0.55)
for pc, col in zip(parts['bodies'], colors):
    pc.set_facecolor(col); pc.set_alpha(0.7)
for comp in ['cmedians', 'cmins', 'cmaxes', 'cbars']:
    parts[comp].set_color('black'); parts[comp].set_linewidth(1.5)
rng_j = np.random.default_rng(7)
for i, (vals, col) in enumerate(zip(data, colors)):
    jitter = rng_j.uniform(-0.08, 0.08, len(vals))
    ax.scatter(i + jitter, vals, color=col, edgecolor='k',
               linewidths=0.5, s=45, zorder=5, alpha=0.85)
ax.axhline(f_true, ls='--', color=C_TRUE, lw=2.0,
           label=f'True max f* = {f_true:.4f}')
ax.set_xticks([0, 1]); ax.set_xticklabels(labels, fontsize=13)
ax.set_ylabel('Best f found', fontsize=12)
ax.legend(fontsize=10)
for i, (vals, col) in enumerate(zip(data, colors)):
    ax.text(i, np.median(vals) + 0.05,
            f'med={np.median(vals):.3f}\nstd={np.std(vals):.3f}',
            ha='center', fontsize=9, color='black', fontweight='bold')
fig.tight_layout()
fig.savefig(os.path.join(PDIR, 'demo_distribution.png'), dpi=200, bbox_inches='tight')
plt.close(fig)
print("Saved: demo_distribution.png")

# ── Summary ───────────────────────────────────────────────────────────────────
print("\n" + "=" * 58)
print("  SUMMARY")
print("=" * 58)
print(f"  True maximum:  f({x_true:.4f}) = {f_true:.6f}")
print(f"  {'Method':<16} {'Best':>8} {'Median':>8} {'Std':>8} "
      f"{'Gap to f*':>12}")
print(f"  {'-'*54}")
for label, results, best_list in [
    ('Random Search', rand_results, rand_best),
    ('ES-RL',         esrl_results, esrl_best),
]:
    b = max(best_list)
    print(f"  {label:<16} {b:>8.5f} {np.median(best_list):>8.5f} "
          f"{np.std(best_list):>8.5f} {f_true-b:>12.5f}")
print("=" * 58)
print("\nAll figures saved to final_result/plots/")
