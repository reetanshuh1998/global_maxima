"""
final_benchmark.py
==================
Comprehensive benchmark of four optimisation algorithms on the
anharmonic quantum Otto cycle efficiency maximisation problem.

Physically grounded λ range
───────────────────────────
The efficiency expressions (Eqs. 8-10 of the reference paper) are
first-order perturbation theory results.  The dimensionless
validity ratio at each bath is:

    ε(ω, β, λ) = (3λ / 4ω³) · coth(βω/2)

The cold bath (small ωc) is always the binding constraint.
We sweep over the feasible space and compute ε at each point;
LAM_MAX is the largest λ such that, at the specific (ωc, βc) of
ANY sample, ε ≤ 0.10 (first-order valid threshold).  In practice
we keep the benchmark cap at LAM_MAX = 0.2 as a domain cap,
but the η-vs-λ plot shows the validity zones explicitly.

Four algorithms
───────────────
  1. Random  — uniform sampling over the feasible set (baseline)
  2. SLSQP   — gradient-based, multi-start
  3. CMA-ES  — covariance-matrix adaptation evolution strategy
  4. ES-RL   — evolution strategy with Adam gradient updates

Outputs (all saved under final_result/)
────────────────────────────────────────
  results/best_parameters.json     – optimal (η, bc, bh, wc, wh, λ) per method
  results/benchmark_raw.json       – full seed-by-seed trial data
  plots/fig_convergence.png        – best-so-far η vs evaluations (median ± IQR)
  plots/fig_violin.png             – final η distribution (violin + strip)
  plots/fig_ecdf.png               – ECDF of evaluations-to-threshold
  plots/fig_param_scatter.png      – scatter of best param vectors coloured by method
  plots/fig_eta_vs_lambda.png      – η(λ) with perturbation validity overlay
  plots/fig_comparison_summary.png – one-page summary panel
"""

import signal
signal.signal(signal.SIGINT, signal.SIG_IGN)

import os, json, time, warnings
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy.optimize import minimize

warnings.filterwarnings("ignore")

# ══════════════════════════════════════════════════════════════════════════════
# 0.  Paths
# ══════════════════════════════════════════════════════════════════════════════
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR   = os.path.join(SCRIPT_DIR, '..', 'final_result')
RDIR       = os.path.join(BASE_DIR, 'results')
PDIR       = os.path.join(BASE_DIR, 'plots')
os.makedirs(RDIR, exist_ok=True)
os.makedirs(PDIR, exist_ok=True)

# ══════════════════════════════════════════════════════════════════════════════
# 1.  Physics
# ══════════════════════════════════════════════════════════════════════════════
R_OMEGA = 7.0    # NV-centre platform frequency ratio cap
R_BETA  = 25.0   # NV-centre temperature ratio cap
LAM_MAX = 0.20   # benchmark domain cap (first-order valid for most of feasible space)

def coth(x):
    ax = abs(x)
    if ax < 1e-10: return np.sign(x) * 1e10
    if ax < 1e-3:  return 1.0/x + x/3.0 - x**3/45.0
    if ax > 20.0:  return float(np.sign(x)) * (1.0 + 2.0*np.exp(-2.0*ax))
    return float(np.cosh(x) / np.sinh(x))

def compute_eta(bc, bh, wc, wh, lam):
    """Eqs. 8-10 of the reference paper."""
    X = coth(bh*wh/2.0);  Y = coth(bc*wc/2.0)
    Qh = (wh/2.0)*(X - Y) + (3.0*lam/(8.0*wh**2))*(X**2 - Y**2)
    Qc = (wc/2.0)*(Y - X) + (3.0*lam/(8.0*wc**2))*(Y**2 - X**2)
    W  = Qh + Qc
    if Qh <= 0 or W <= 0: return 0.0
    return float(np.clip(W / Qh, 0.0, 1.0))

def eps_cold(bc, wc, lam):
    """Perturbation validity ratio at cold bath."""
    return (3.0*lam / (4.0*wc**3)) * coth(bc*wc/2.0)

def eps_hot(bh, wh, lam):
    """Perturbation validity ratio at hot bath."""
    return (3.0*lam / (4.0*wh**3)) * coth(bh*wh/2.0)

def feasible(bc, bh, wc, wh, lam):
    if wc >= wh or bh >= bc: return False
    if lam < 0 or lam > LAM_MAX: return False
    if bc*wc <= bh*wh: return False
    if wh/wc > R_OMEGA or bc/bh > R_BETA: return False
    return True

def eta_safe(params):
    bc, bh, wc, wh, lam = (float(p) for p in params)
    if not feasible(bc, bh, wc, wh, lam): return 0.0
    return compute_eta(bc, bh, wc, wh, lam)

# Parameter box bounds  [bc, bh, wc, wh, lam]
LO = np.array([0.5,  0.05, 0.3, 1.0, 0.0])
HI = np.array([30.0, 15.0, 8.0, 30.0, LAM_MAX])
PARAM_NAMES = [r'$\beta_c$', r'$\beta_h$', r'$\omega_c$', r'$\omega_h$', r'$\lambda$']

def sample_feasible(rng, n_try=20000):
    for _ in range(n_try):
        x = rng.uniform(LO, HI)
        if feasible(*x): return x
    return (LO + HI) / 2.0   # fallback

# ══════════════════════════════════════════════════════════════════════════════
# 2.  Algorithm implementations
# ══════════════════════════════════════════════════════════════════════════════
THRESHOLD = 0.80
BUDGET    = 1500
N_SEEDS   = 15

SLSQP_CONS = [
    {'type':'ineq','fun': lambda x: x[3] - x[2] - 0.05},
    {'type':'ineq','fun': lambda x: x[0] - x[1] - 0.05},
    {'type':'ineq','fun': lambda x: x[0]*x[2] - x[1]*x[3] - 0.01},
    {'type':'ineq','fun': lambda x: R_OMEGA - x[3]/max(x[2], 1e-6)},
    {'type':'ineq','fun': lambda x: R_BETA  - x[0]/max(x[1], 1e-6)},
    {'type':'ineq','fun': lambda x: LAM_MAX - x[4]},
]
BNDS = list(zip(LO, HI))

# ── Random ────────────────────────────────────────────────────────────────────
def run_random(seed):
    rng   = np.random.default_rng(seed)
    hist  = [0.0]; best = 0.0; best_x = None
    ev    = 0; found = None; t0 = time.perf_counter()
    while ev < BUDGET:
        x = sample_feasible(rng)
        r = eta_safe(x); ev += 1
        if r > best: best = r; best_x = x.copy()
        hist.append(best)
        if r >= THRESHOLD and found is None:
            found = (ev, time.perf_counter() - t0)
    return dict(hist=hist, best=best, best_x=best_x,
                ev=ev, t=time.perf_counter()-t0, found=found)

# ── SLSQP ─────────────────────────────────────────────────────────────────────
def run_slsqp(seed):
    rng   = np.random.default_rng(seed)
    hist  = [0.0]; best = [0.0]; best_x = [None]
    ev    = [0]; found = [None]; t0 = time.perf_counter()
    def obj(x):
        r = eta_safe(x); ev[0] += 1
        if r > best[0]: best[0] = r; best_x[0] = x.copy()
        hist.append(best[0])
        if r >= THRESHOLD and found[0] is None:
            found[0] = (ev[0], time.perf_counter() - t0)
        return -r
    for _ in range(25):
        if ev[0] >= BUDGET: break
        x0 = sample_feasible(rng)
        # bias toward high ratio regime (where η is large)
        if np.random.rand() < 0.4:
            r_w = np.random.uniform(R_OMEGA*0.8, R_OMEGA*0.99)
            r_b = np.random.uniform(R_BETA *0.7, R_BETA *0.99)
            wc0 = np.random.uniform(0.3, 2.0)
            wh0 = min(wc0 * r_w, HI[3])
            bh0 = np.random.uniform(0.1, 5.0)
            bc0 = min(bh0 * r_b, HI[0])
            if bc0*wc0 > bh0*wh0:
                x0 = np.clip([bc0, bh0, wc0, wh0, 0.0], LO, HI)
        try:
            minimize(obj, x0, method='SLSQP', bounds=BNDS,
                     constraints=SLSQP_CONS,
                     options={'maxiter': 200, 'ftol': 1e-12, 'disp': False})
        except Exception:
            pass
    return dict(hist=hist, best=best[0], best_x=best_x[0],
                ev=ev[0], t=time.perf_counter()-t0, found=found[0])

# ── CMA-ES ────────────────────────────────────────────────────────────────────
def run_cmaes(seed):
    np.random.seed(seed)
    rng = np.random.default_rng(seed)
    hist = [0.0]; best = 0.0; best_x = None
    ev = 0; found = None; t0 = time.perf_counter()
    d = 5; lam_p = 12; mu = lam_p // 2
    w = np.log(mu + 0.5) - np.log(np.arange(1, mu+1))
    w /= w.sum()
    mueff = 1.0 / (w**2).sum()
    C = np.eye(d); ps = np.zeros(d); pc = np.zeros(d); s = 0.3; g = 0
    cc   = (4 + mueff/d) / (d + 4 + 2*mueff/d)
    cs   = (mueff + 2) / (d + mueff + 5)
    c1   = 2.0 / ((d+1.3)**2 + mueff)
    cmu  = min(1-c1, 2*(mueff-2+1/mueff) / ((d+2)**2 + mueff))
    damp = 1 + 2*max(0, np.sqrt((mueff-1)/(d+1))-1) + cs
    chiN = d**0.5 * (1 - 1/(4*d) + 1/(21*d**2))
    # initialise near the high-ratio regime
    x0_raw = sample_feasible(rng)
    m = np.clip((x0_raw - LO) / (HI - LO), 0.05, 0.95)
    while ev < BUDGET:
        try:
            L = np.linalg.cholesky(C + 1e-9*np.eye(d))
        except np.linalg.LinAlgError:
            L = np.eye(d)
        z  = np.random.randn(lam_p, d)
        xs = m + s * (z @ L.T)
        fs = []
        for x_norm in xs:
            x_real = LO + np.clip(x_norm, 0, 1) * (HI - LO)
            r = eta_safe(x_real); ev += 1; fs.append(r)
            if r > best: best = r; best_x = x_real.copy()
            hist.append(best)
            if r >= THRESHOLD and found is None:
                found = (ev, time.perf_counter() - t0)
            if ev >= BUDGET: break
        fs = np.array(fs)
        idx = np.argsort(-fs)
        old = m.copy()
        m   = (w * xs[idx[:mu]].T).sum(1)
        step = (m - old) / s
        try:
            Ci = np.linalg.inv(L).T
        except np.linalg.LinAlgError:
            Ci = np.eye(d)
        g += 1
        ps = (1-cs)*ps + np.sqrt(cs*(2-cs)*mueff) * Ci @ step
        hs = (np.linalg.norm(ps) / np.sqrt(1-(1-cs)**(2*g)) / chiN) < 1.4 + 2/(d+1)
        pc = (1-cc)*pc + hs * np.sqrt(cc*(2-cc)*mueff) * step
        diff = (xs[idx[:mu]] - old) / s
        C = ((1-c1-cmu)*C
             + c1*(np.outer(pc,pc) + (1-hs)*cc*(2-cc)*C)
             + cmu*(w * diff.T @ diff))
        s *= np.exp((cs/damp) * (np.linalg.norm(ps)/chiN - 1))
        s  = np.clip(s, 1e-8, 2.0)
    return dict(hist=hist, best=best, best_x=best_x,
                ev=ev, t=time.perf_counter()-t0, found=found)

# ── ES-RL ─────────────────────────────────────────────────────────────────────
def run_esrl(seed):
    rng  = np.random.default_rng(seed)
    hist = [0.0]; best = 0.0; best_x = None
    ev   = 0; found = None; t0 = time.perf_counter()
    x0_raw = sample_feasible(rng)
    mu = np.clip((x0_raw - LO) / (HI - LO), 0.05, 0.95)
    sig = 0.15; m_adam = np.zeros(5); v_adam = np.zeros(5)
    t_adam = 0; lr = 0.03; b1, b2, eps_a = 0.9, 0.999, 1e-8
    H = 10  # antithetic pairs per step
    while ev < BUDGET:
        eps_b = rng.standard_normal((H, 5)) * sig
        rp = np.zeros(H); rn = np.zeros(H)
        for i in range(H):
            if ev >= BUDGET: break
            for sgn, arr in [(+1, rp), (-1, rn)]:
                if ev >= BUDGET: break
                x_real = LO + np.clip(mu + sgn*eps_b[i], 0, 1) * (HI - LO)
                r = eta_safe(x_real); ev += 1; arr[i] = r
                if r > best: best = r; best_x = x_real.copy()
                hist.append(best)
                if r >= THRESHOLD and found is None:
                    found = (ev, time.perf_counter() - t0)
        g_est = np.sum(((rp - rn)[:, None] * eps_b), axis=0) / (2 * sig * H)
        t_adam += 1
        m_adam = b1*m_adam + (1-b1)*g_est
        v_adam = b2*v_adam + (1-b2)*g_est**2
        mh = m_adam / (1 - b1**t_adam)
        vh = v_adam / (1 - b2**t_adam)
        mu = np.clip(mu + lr * mh / (np.sqrt(vh) + eps_a), 0.0, 1.0)
    return dict(hist=hist, best=best, best_x=best_x,
                ev=ev, t=time.perf_counter()-t0, found=found)

# ══════════════════════════════════════════════════════════════════════════════
# 3.  Run all methods
# ══════════════════════════════════════════════════════════════════════════════
METHODS = [
    ('Random', run_random, '#7f8c8d'),
    ('SLSQP',  run_slsqp, '#e74c3c'),
    ('CMA-ES', run_cmaes, '#3498db'),
    ('ES-RL',  run_esrl,  '#9b59b6'),
]
COLORS = {name: col for name, _, col in METHODS}

print("=" * 65)
print("   Anharmonic Quantum Otto Cycle — Final Benchmark")
print(f"   R_ω={R_OMEGA}, R_β={R_BETA}, λ_max={LAM_MAX}")
print(f"   Seeds={N_SEEDS}, Budget={BUDGET}, Threshold={THRESHOLD}")
print("=" * 65)

all_trials = {}   # {method: [trial_dict, ...]}
best_overall = {'eta': 0.0, 'params': None, 'method': None}

for name, runner, _ in METHODS:
    print(f"\n  Running {name} ({N_SEEDS} seeds × {BUDGET} evals) …")
    trials = []
    for s in range(N_SEEDS):
        t = runner(s)
        trials.append(t)
        if t['best'] > best_overall['eta']:
            best_overall['eta']   = t['best']
            best_overall['params'] = t['best_x']
            best_overall['method'] = name
    all_trials[name] = trials

    bests  = [t['best'] for t in trials]
    succ   = [t['found'] for t in trials if t['found']]
    sev    = [s[0] for s in succ]
    stim   = [s[1] for s in succ]
    best_p = max(trials, key=lambda t: t['best'])['best_x']
    print(f"    η_max  = {max(bests):.6f}")
    print(f"    η_med  = {np.median(bests):.6f}  ± {np.std(bests):.6f} (std)")
    print(f"    Success= {len(succ)}/{N_SEEDS}  "
          f"med_evals={int(np.median(sev)) if sev else 'N/A'}  "
          f"med_time={np.median(stim):.4f}s" if succ else "")
    if best_p is not None:
        bc,bh,wc,wh,lam = best_p
        print(f"    Best params: β_c={bc:.4f}, β_h={bh:.4f}, "
              f"ω_c={wc:.4f}, ω_h={wh:.4f}, λ={lam:.4f}")
        ec = eps_cold(bc, wc, lam)
        eh = eps_hot(bh, wh, lam)
        print(f"    Perturbation validity: ε_cold={ec:.4f}, ε_hot={eh:.4f}")

print(f"\n{'─'*65}")
print(f"  Global best: η = {best_overall['eta']:.6f}  (found by {best_overall['method']})")
if best_overall['params'] is not None:
    bc,bh,wc,wh,lam = best_overall['params']
    print(f"  Parameters:  β_c={bc:.4f}, β_h={bh:.4f}, "
          f"ω_c={wc:.4f}, ω_h={wh:.4f}, λ={lam:.4f}")
    print(f"  Analytic bound: 1-1/R_ω = {1-1/R_OMEGA:.6f}")
print('─'*65)

# ══════════════════════════════════════════════════════════════════════════════
# 4.  Save results JSON
# ══════════════════════════════════════════════════════════════════════════════
def ser(o):
    if isinstance(o, np.integer):  return int(o)
    if isinstance(o, np.floating): return float(o)
    if isinstance(o, np.ndarray):  return o.tolist()
    return str(o)

# best_parameters.json — one entry per method
best_params_out = {}
for name, _, _ in METHODS:
    trials = all_trials[name]
    best_t = max(trials, key=lambda t: t['best'])
    bests  = [t['best'] for t in trials]
    succ   = [t['found'] for t in trials if t['found']]
    bp = best_t['best_x']
    entry = {
        'eta_max':         best_t['best'],
        'eta_median':      float(np.median(bests)),
        'eta_std':         float(np.std(bests)),
        'success_rate':    len(succ) / N_SEEDS,
        'med_evals_to_threshold': int(np.median([s[0] for s in succ])) if succ else None,
        'med_time_to_threshold':  float(np.median([s[1] for s in succ])) if succ else None,
        'best_params': {
            'beta_c':  float(bp[0]) if bp is not None else None,
            'beta_h':  float(bp[1]) if bp is not None else None,
            'omega_c': float(bp[2]) if bp is not None else None,
            'omega_h': float(bp[3]) if bp is not None else None,
            'lambda':  float(bp[4]) if bp is not None else None,
        } if bp is not None else None,
        'eps_cold_at_best': eps_cold(bp[0], bp[2], bp[4]) if bp is not None else None,
        'eps_hot_at_best':  eps_hot(bp[1], bp[3], bp[4]) if bp is not None else None,
        'analytic_bound': 1.0 - 1.0/R_OMEGA,
    }
    best_params_out[name] = entry

best_params_out['_meta'] = {
    'R_omega': R_OMEGA, 'R_beta': R_BETA, 'LAM_MAX': LAM_MAX,
    'N_seeds': N_SEEDS, 'budget': BUDGET, 'threshold': THRESHOLD,
    'global_best_eta': best_overall['eta'],
    'global_best_method': best_overall['method'],
}
with open(os.path.join(RDIR, 'best_parameters.json'), 'w') as f:
    json.dump(best_params_out, f, default=ser, indent=2)
print(f"\nSaved → final_result/results/best_parameters.json")

# benchmark_raw.json
raw_out = {}
for name, _, _ in METHODS:
    raw_out[name] = []
    for t in all_trials[name]:
        raw_out[name].append({
            'best': t['best'],
            'ev':   t['ev'],
            't':    t['t'],
            'found': list(t['found']) if t['found'] else None,
            'hist': t['hist'][::10],   # sub-sampled to keep file small
        })
with open(os.path.join(RDIR, 'benchmark_raw.json'), 'w') as f:
    json.dump(raw_out, f, default=ser, indent=2)
print(f"Saved → final_result/results/benchmark_raw.json")

# ══════════════════════════════════════════════════════════════════════════════
# 5.  Plotting helpers
# ══════════════════════════════════════════════════════════════════════════════
MNAMES = [m[0] for m in METHODS]
plt.rcParams.update({
    'font.family': 'serif', 'font.size': 11,
    'axes.linewidth': 1.2, 'axes.grid': True, 'grid.alpha': 0.3,
    'legend.framealpha': 0.9,
})

def pad_hist(h, n):
    h = list(h)
    return (h + [h[-1]] * (n - len(h)))[:n]

# ── Fig 1: Convergence (best-so-far η vs evaluations) ─────────────────────────
print("\nGenerating plots …")
fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), sharey=True)
fig.suptitle(
    rf'Best-So-Far $\eta$ vs Evaluations  (Median ± IQR, {N_SEEDS} seeds)'
    f'\n(NV-centre platform: $R_\\omega={R_OMEGA}$, $R_\\beta={R_BETA}$, '
    rf'$\lambda \leq {LAM_MAX}$)',
    fontsize=12
)
for ax, track, warm_label in [(axes[0], 'cold', 'Cold Start'),
                               (axes[1], 'warm', 'Warm Start')]:
    # all methods run cold-start (warm not separately tracked here)
    maxl = max(len(t['hist']) for name in MNAMES for t in all_trials[name])
    for name in MNAMES:
        H = np.array([pad_hist(t['hist'], maxl) for t in all_trials[name]])
        med = np.median(H, 0)
        q25 = np.percentile(H, 25, 0)
        q75 = np.percentile(H, 75, 0)
        x   = np.arange(maxl)
        ax.plot(x, med, color=COLORS[name], lw=2.2, label=name)
        ax.fill_between(x, q25, q75, color=COLORS[name], alpha=0.18)
    ax.axhline(THRESHOLD, ls='--', color='k', lw=1.2, alpha=0.7,
               label=rf'Threshold $\eta={THRESHOLD}$')
    ax.axhline(1-1/R_OMEGA, ls=':', color='green', lw=1.5,
               label=rf'Analytic $\eta_{{max}}={1-1/R_OMEGA:.4f}$')
    ax.set_xlabel('Evaluations', fontsize=12)
    ax.set_ylabel(r'Best $\eta$ (so far)', fontsize=12)
    ax.set_title(track.capitalize() + ' Start', fontsize=12)
    ax.legend(fontsize=9)
    if track == 'cold': break   # single track — both panels same data

# second panel: zoom 0-200 evals
ax2 = axes[1]
maxl_z = min(200, maxl)
for name in MNAMES:
    H = np.array([pad_hist(t['hist'], maxl_z) for t in all_trials[name]])
    med = np.median(H, 0)
    q25 = np.percentile(H, 25, 0)
    q75 = np.percentile(H, 75, 0)
    x   = np.arange(maxl_z)
    ax2.plot(x, med, color=COLORS[name], lw=2.2, label=name)
    ax2.fill_between(x, q25, q75, color=COLORS[name], alpha=0.18)
ax2.axhline(THRESHOLD, ls='--', color='k', lw=1.2, alpha=0.7)
ax2.axhline(1-1/R_OMEGA, ls=':', color='green', lw=1.5)
ax2.set_xlabel('Evaluations', fontsize=12)
ax2.set_title('Zoom: first 200 evaluations', fontsize=12)
ax2.legend(fontsize=9)

fig.tight_layout()
fig.savefig(os.path.join(PDIR, 'fig_convergence.png'), dpi=200, bbox_inches='tight')
plt.close(fig)
print("  Saved: fig_convergence.png")

# ── Fig 2: Violin + strip plot of final η ─────────────────────────────────────
fig, ax = plt.subplots(figsize=(9, 6))
fig.suptitle(
    rf'Distribution of Best $\eta$ Achieved  ({N_SEEDS} seeds per method)'
    f'\nAnalytic bound: $\\eta_{{max}} = 1-1/R_\\omega = {1-1/R_OMEGA:.4f}$',
    fontsize=12
)
data   = [[t['best'] for t in all_trials[name]] for name in MNAMES]
pos    = np.arange(len(MNAMES))
parts  = ax.violinplot(data, positions=pos, showmedians=True, showextrema=True,
                        widths=0.6)
for pc, name in zip(parts['bodies'], MNAMES):
    pc.set_facecolor(COLORS[name]); pc.set_alpha(0.7)
for comp in ['cmedians', 'cmins', 'cmaxes', 'cbars']:
    parts[comp].set_color('black'); parts[comp].set_linewidth(1.5)
# strip jitter
rng_p = np.random.default_rng(0)
for i, (name, vals) in enumerate(zip(MNAMES, data)):
    jitter = rng_p.uniform(-0.08, 0.08, len(vals))
    ax.scatter(i + jitter, vals, color=COLORS[name],
               edgecolor='k', linewidth=0.5, s=45, zorder=5, alpha=0.8)
ax.axhline(THRESHOLD, ls='--', color='red', lw=1.5,
           label=rf'Threshold $\eta={THRESHOLD}$')
ax.axhline(1-1/R_OMEGA, ls=':', color='green', lw=1.8,
           label=rf'Analytic $\eta_{{max}}={1-1/R_OMEGA:.4f}$')
ax.set_xticks(pos); ax.set_xticklabels(MNAMES, fontsize=12)
ax.set_ylabel(r'Best $\eta$ achieved', fontsize=12)
ax.legend(fontsize=10)
# annotate medians
for i, (name, vals) in enumerate(zip(MNAMES, data)):
    ax.text(i, np.median(vals) + 0.005, f'med={np.median(vals):.4f}',
            ha='center', fontsize=8, color='black', fontweight='bold')
fig.tight_layout()
fig.savefig(os.path.join(PDIR, 'fig_violin.png'), dpi=200, bbox_inches='tight')
plt.close(fig)
print("  Saved: fig_violin.png")

# ── Fig 3: ECDF of evaluations-to-threshold ───────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 5.5))
fig.suptitle(
    rf'ECDF of Evaluations to Reach $\eta \geq {THRESHOLD}$  ({N_SEEDS} seeds)',
    fontsize=12
)
for name in MNAMES:
    evs = sorted([t['found'][0] for t in all_trials[name] if t['found']])
    if not evs:
        ax.step([0, BUDGET], [0, 0], color=COLORS[name], lw=2, label=f'{name} (0/{N_SEEDS})')
        continue
    n     = len(evs)
    ecdf_x = [0] + evs + [BUDGET]
    ecdf_y = [0] + [i/N_SEEDS for i in range(1, n+1)] + [n/N_SEEDS]
    ax.step(ecdf_x, ecdf_y, color=COLORS[name], lw=2.2,
            label=f'{name} ({n}/{N_SEEDS} succeed)')
ax.axhline(1.0, ls=':', color='gray', lw=1)
ax.set_xlabel('Evaluations', fontsize=12)
ax.set_ylabel(rf'Fraction reaching $\eta \geq {THRESHOLD}$', fontsize=12)
ax.set_ylim(-0.05, 1.1); ax.set_xlim(0, BUDGET)
ax.legend(fontsize=10)
fig.tight_layout()
fig.savefig(os.path.join(PDIR, 'fig_ecdf.png'), dpi=200, bbox_inches='tight')
plt.close(fig)
print("  Saved: fig_ecdf.png")

# ── Fig 4: Best parameter vectors per method ──────────────────────────────────
fig, axes = plt.subplots(2, 3, figsize=(15, 9))
fig.suptitle(
    'Best Parameter Vectors Found by Each Algorithm\n'
    r'(Scatter over all seeds — colour = method, star = global best)',
    fontsize=12
)
param_pairs = [
    (3, 2, r'$\omega_h$', r'$\omega_c$'),   # wh vs wc
    (0, 1, r'$\beta_c$',  r'$\beta_h$'),    # bc vs bh
    (4, 2, r'$\lambda$',  r'$\omega_c$'),   # lam vs wc
    (3, 0, r'$\omega_h$', r'$\beta_c$'),    # wh vs bc
    (4, 0, r'$\lambda$',  r'$\beta_c$'),    # lam vs bc
    (3, 1, r'$\omega_h$', r'$\beta_h$'),    # wh vs bh
]
for ax, (xi, yi, xl, yl) in zip(axes.flat, param_pairs):
    for name in MNAMES:
        pts = [t['best_x'] for t in all_trials[name] if t['best_x'] is not None]
        if pts:
            arr = np.array(pts)
            ax.scatter(arr[:, xi], arr[:, yi], color=COLORS[name],
                       alpha=0.6, s=30, label=name, edgecolor='none')
    # global best star
    if best_overall['params'] is not None:
        gp = best_overall['params']
        ax.scatter(gp[xi], gp[yi], marker='*', s=280, color='gold',
                   edgecolor='k', linewidth=1.2, zorder=10,
                   label=f"Global best ({best_overall['method']})")
    ax.set_xlabel(xl, fontsize=11); ax.set_ylabel(yl, fontsize=11)
    if xi == 3 and yi == 2:   # put legend only on first
        ax.legend(fontsize=8, loc='upper left')
fig.tight_layout()
fig.savefig(os.path.join(PDIR, 'fig_param_scatter.png'), dpi=200, bbox_inches='tight')
plt.close(fig)
print("  Saved: fig_param_scatter.png")

# ── Fig 5: η vs λ with perturbation validity overlay ─────────────────────────
# Use the global best parameters as the fixed reference
if best_overall['params'] is not None:
    GP = best_overall['params']
    REF_BC, REF_BH, REF_WC, REF_WH = GP[0], GP[1], GP[2], GP[3]
else:
    REF_BC, REF_BH, REF_WC, REF_WH = 30.0, 3.643, 0.594, 4.161

# Perturbation ratio per unit λ at these parameters
epl_cold = (3.0 / (4.0 * REF_WC**3)) * coth(REF_BC * REF_WC / 2.0)
epl_hot  = (3.0 / (4.0 * REF_WH**3)) * coth(REF_BH * REF_WH / 2.0)

lam_1pct  = 0.01 / epl_cold
lam_10pct = 0.10 / epl_cold
lam_30pct = 0.30 / epl_cold

# Plot up to 3× the qualitative limit or 1.0, whichever is smaller
lam_plot_max = min(3 * lam_30pct, 1.0)
lam_arr = np.linspace(0.0, lam_plot_max, 600)

# Vectorised η (fixed bc, bh, wc, wh; vary lam)
vcoth = np.vectorize(coth)
X = coth(REF_BH * REF_WH / 2.0)
Y = coth(REF_BC * REF_WC / 2.0)

def _eta_lam(lam_v):
    Qh = (REF_WH/2.0)*(X-Y) + (3.0*lam_v/(8.0*REF_WH**2))*(X**2-Y**2)
    Qc = (REF_WC/2.0)*(Y-X) + (3.0*lam_v/(8.0*REF_WC**2))*(Y**2-X**2)
    W  = Qh + Qc
    return np.where((Qh > 0) & (W > 0), np.clip(W/Qh, 0, 1), np.nan)

eta_lam_arr = _eta_lam(lam_arr)
ec_arr = epl_cold * lam_arr
eh_arr = epl_hot  * lam_arr

C_GREEN  = '#d5f5e3'; C_YELLOW = '#fef9e7'
C_ORANGE = '#fdebd0'; C_RED    = '#fadbd8'

fig, axes_lam = plt.subplots(1, 2, figsize=(15, 6))
fig.suptitle(
    rf'$\eta(\lambda)$ with Perturbation Validity Overlay'
    '\n(Other parameters fixed at global optimum: '
    rf'$\beta_c={REF_BC:.2f}$, $\beta_h={REF_BH:.3f}$, '
    rf'$\omega_c={REF_WC:.3f}$, $\omega_h={REF_WH:.3f}$)',
    fontsize=12
)
for ax_l, zoom, title_sfx in [
    (axes_lam[0], False, 'Full range'),
    (axes_lam[1], True,  rf'Zoom: $\varepsilon_{{cold}} \leq 0.30$'),
]:
    ax_r = ax_l.twinx()
    lam_plot = lam_arr if not zoom else lam_arr[lam_arr <= lam_30pct * 1.1]
    eta_plot  = eta_lam_arr if not zoom else _eta_lam(lam_plot)
    ec_plot   = epl_cold * lam_plot
    eh_plot   = epl_hot  * lam_plot

    # Background validity zones
    for xl, xr, color in [
        (0, lam_1pct,  C_GREEN),
        (lam_1pct, lam_10pct, C_YELLOW),
        (lam_10pct, lam_30pct, C_ORANGE),
        (lam_30pct, lam_plot.max(), C_RED),
    ]:
        ax_l.axvspan(xl, min(xr, lam_plot.max()), alpha=0.5, color=color, zorder=0)

    ax_l.plot(lam_plot, eta_plot, color='#2980b9', lw=2.5, zorder=5,
               label=r'$\eta(\lambda)$')
    ax_l.axhline(1-1/R_OMEGA, ls=':', color='#2980b9', lw=1.2, alpha=0.7,
                  label=rf'$\eta_0 = {1-1/R_OMEGA:.4f}$')

    ax_r.plot(lam_plot, ec_plot, color='#e74c3c', lw=1.8, ls='--',
               label=r'$\varepsilon_\mathrm{cold}$')
    ax_r.plot(lam_plot, eh_plot, color='#27ae60', lw=1.8, ls='-.',
               label=r'$\varepsilon_\mathrm{hot}$')
    for thresh, ls in [(0.01,':'), (0.10,'--'), (0.30,'-')]:
        ax_r.axhline(thresh, ls=ls, color='gray', lw=1.2, alpha=0.8)

    for lam_c, label, color in [
        (lam_1pct,  rf'$\lambda_{{1\%}}={lam_1pct:.3f}$',  '#27ae60'),
        (lam_10pct, rf'$\lambda_{{10\%}}={lam_10pct:.3f}$', '#e67e22'),
        (lam_30pct, rf'$\lambda_{{30\%}}={lam_30pct:.3f}$', '#c0392b'),
    ]:
        if lam_c <= lam_plot.max():
            ax_l.axvline(lam_c, ls=':', color=color, lw=1.4)
            ax_l.text(lam_c + 0.001, 0.01, label, color=color,
                       fontsize=8, rotation=90, va='bottom')

    ax_l.set_xlabel(r'$\lambda$ (anharmonicity)', fontsize=12)
    ax_l.set_ylabel(r'$\eta$ (efficiency)', fontsize=12, color='#2980b9')
    ax_r.set_ylabel(
        r'$\varepsilon = \frac{3\lambda}{4\omega^3}\coth\!\left(\frac{\beta\omega}{2}\right)$',
        fontsize=11, color='#7f8c8d'
    )
    ax_l.set_xlim(0, lam_plot.max()); ax_l.set_ylim(bottom=0)
    ax_r.set_ylim(bottom=0)
    ax_l.tick_params(axis='y', colors='#2980b9')
    ax_l.set_title(title_sfx, fontsize=11)

    zone_patches = [
        mpatches.Patch(color=C_GREEN,  alpha=0.7, label=r'Fully valid ($\varepsilon<0.01$)'),
        mpatches.Patch(color=C_YELLOW, alpha=0.7, label=r'1st-order valid ($\varepsilon<0.10$)'),
        mpatches.Patch(color=C_ORANGE, alpha=0.7, label=r'Qualitative ($\varepsilon<0.30$)'),
        mpatches.Patch(color=C_RED,    alpha=0.7, label=r'Broken ($\varepsilon\geq 0.30$)'),
    ]
    lines_l, labs_l = ax_l.get_legend_handles_labels()
    lines_r, labs_r = ax_r.get_legend_handles_labels()
    ax_l.legend(lines_l + lines_r + zone_patches,
                labs_l + labs_r + [p.get_label() for p in zone_patches],
                fontsize=7.5, loc='upper right', ncol=2)

fig.tight_layout()
fig.savefig(os.path.join(PDIR, 'fig_eta_vs_lambda.png'), dpi=200, bbox_inches='tight')
plt.close(fig)
print("  Saved: fig_eta_vs_lambda.png")

# ── Fig 6: One-page comparison summary ───────────────────────────────────────
fig, axes_s = plt.subplots(2, 3, figsize=(17, 10))
fig.suptitle(
    'Algorithm Comparison — Anharmonic Quantum Otto Cycle\n'
    rf'NV-centre platform: $R_\omega={R_OMEGA}$, $R_\beta={R_BETA}$, '
    rf'$\lambda\leq{LAM_MAX}$, $\eta_{{max}}^{{analytic}}={1-1/R_OMEGA:.4f}$',
    fontsize=13, y=1.01
)

# Panel (0,0): η_max bar chart
ax = axes_s[0, 0]
eta_maxes = [max(t['best'] for t in all_trials[name]) for name in MNAMES]
bars = ax.bar(MNAMES, eta_maxes, color=[COLORS[n] for n in MNAMES],
              edgecolor='k', linewidth=0.8, alpha=0.88)
ax.axhline(1-1/R_OMEGA, ls='--', color='green', lw=1.8,
           label=rf'Analytic bound')
for bar, val in zip(bars, eta_maxes):
    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.001,
            f'{val:.4f}', ha='center', fontsize=9, fontweight='bold')
ax.set_ylabel(r'$\eta_{max}$ found', fontsize=11)
ax.set_title(r'Maximum $\eta$ — Best Seed', fontsize=11)
ax.set_ylim(0, 1); ax.legend(fontsize=9)

# Panel (0,1): median η bar + std error bars
ax = axes_s[0, 1]
med_eta = [np.median([t['best'] for t in all_trials[n]]) for n in MNAMES]
std_eta = [np.std([t['best'] for t in all_trials[n]]) for n in MNAMES]
ax.bar(MNAMES, med_eta, yerr=std_eta, color=[COLORS[n] for n in MNAMES],
       edgecolor='k', linewidth=0.8, alpha=0.88, capsize=5)
ax.axhline(THRESHOLD, ls='--', color='red', lw=1.5)
ax.axhline(1-1/R_OMEGA, ls='--', color='green', lw=1.5)
ax.set_ylabel(r'Median $\eta$ ± std', fontsize=11)
ax.set_title(r'Median Efficiency (Statistical Robustness)', fontsize=11)
ax.set_ylim(0, 1)

# Panel (0,2): Success rate
ax = axes_s[0, 2]
succ_rates = [sum(1 for t in all_trials[n] if t['found'])/N_SEEDS for n in MNAMES]
bars = ax.bar(MNAMES, [r*100 for r in succ_rates],
              color=[COLORS[n] for n in MNAMES],
              edgecolor='k', linewidth=0.8, alpha=0.88)
for bar, val in zip(bars, succ_rates):
    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+1,
            f'{val*100:.0f}%', ha='center', fontsize=10, fontweight='bold')
ax.set_ylabel(f'% seeds reaching η ≥ {THRESHOLD}', fontsize=11)
ax.set_title('Success Rate', fontsize=11)
ax.set_ylim(0, 115)

# Panel (1,0): Median evaluations to threshold
ax = axes_s[1, 0]
med_ev = []
for name in MNAMES:
    evs = [t['found'][0] for t in all_trials[name] if t['found']]
    med_ev.append(np.median(evs) if evs else float('nan'))
bars = ax.bar(MNAMES, med_ev, color=[COLORS[n] for n in MNAMES],
              edgecolor='k', linewidth=0.8, alpha=0.88)
for bar, val in zip(bars, med_ev):
    if not np.isnan(val):
        ax.text(bar.get_x()+bar.get_width()/2, val+2,
                f'{int(val)}', ha='center', fontsize=10, fontweight='bold')
ax.set_ylabel(f'Median evaluations to η ≥ {THRESHOLD}', fontsize=11)
ax.set_title('Speed to Threshold (fewer = better)', fontsize=11)

# Panel (1,1): λ at best solution + ε_cold
ax = axes_s[1, 1]
lam_best   = []
ec_best    = []
eta_at_best = []
for name in MNAMES:
    bp = max(all_trials[name], key=lambda t: t['best'])['best_x']
    if bp is not None:
        lam_best.append(bp[4])
        ec_best.append(eps_cold(bp[0], bp[2], bp[4]))
        eta_at_best.append(compute_eta(*bp))
    else:
        lam_best.append(0); ec_best.append(0); eta_at_best.append(0)
x_pos = np.arange(len(MNAMES))
ax.bar(x_pos - 0.2, lam_best, 0.35, color=[COLORS[n] for n in MNAMES],
       edgecolor='k', linewidth=0.8, alpha=0.88, label=r'$\lambda^*$ (best solution)')
ax2_twin = ax.twinx()
ax2_twin.bar(x_pos + 0.2, ec_best, 0.35, color=[COLORS[n] for n in MNAMES],
             edgecolor='k', linewidth=0.8, alpha=0.45, hatch='//',
             label=r'$\varepsilon_\mathrm{cold}$')
ax2_twin.axhline(0.10, ls='--', color='orange', lw=1.5, label='ε=0.10')
ax2_twin.axhline(0.30, ls='--', color='red', lw=1.2, label='ε=0.30')
ax.set_xticks(x_pos); ax.set_xticklabels(MNAMES, fontsize=10)
ax.set_ylabel(r'$\lambda^*$ at best solution', fontsize=11)
ax2_twin.set_ylabel(r'$\varepsilon_\mathrm{cold}$ (perturbation validity)', fontsize=10)
ax.set_title('Physical Quality of Best Solution', fontsize=11)
ax.legend(fontsize=8, loc='upper left')
ax2_twin.legend(fontsize=8, loc='upper right')

# Panel (1,2): Convergence speed (IQR of evaluations)
ax = axes_s[1, 2]
maxl_s = max(len(t['hist']) for name in MNAMES for t in all_trials[name])
for name in MNAMES:
    H = np.array([pad_hist(t['hist'], maxl_s) for t in all_trials[name]])
    med = np.median(H, 0); q25 = np.percentile(H, 25, 0); q75 = np.percentile(H, 75, 0)
    x_ev = np.arange(maxl_s)
    ax.plot(x_ev, med, color=COLORS[name], lw=2, label=name)
    ax.fill_between(x_ev, q25, q75, color=COLORS[name], alpha=0.15)
ax.axhline(THRESHOLD, ls='--', color='k', lw=1.1, alpha=0.7)
ax.axhline(1-1/R_OMEGA, ls=':', color='green', lw=1.5)
ax.set_xlabel('Evaluations', fontsize=11)
ax.set_ylabel(r'Best $\eta$ (median ± IQR)', fontsize=11)
ax.set_title('Convergence', fontsize=11); ax.legend(fontsize=9)

fig.tight_layout()
fig.savefig(os.path.join(PDIR, 'fig_comparison_summary.png'), dpi=200, bbox_inches='tight')
plt.close(fig)
print("  Saved: fig_comparison_summary.png")

# ══════════════════════════════════════════════════════════════════════════════
# 6.  Print final summary table
# ══════════════════════════════════════════════════════════════════════════════
print(f"\n{'='*65}")
print(f"  FINAL SUMMARY")
print(f"{'='*65}")
print(f"  {'Method':<10} {'η_max':>8} {'η_med':>8} {'Succ':>6} "
      f"{'MedEval':>8} {'λ*':>7} {'ε_cold*':>8}")
print(f"  {'-'*63}")
for name in MNAMES:
    bests = [t['best'] for t in all_trials[name]]
    succ  = [t['found'] for t in all_trials[name] if t['found']]
    sev   = [s[0] for s in succ]
    bp    = max(all_trials[name], key=lambda t: t['best'])['best_x']
    lam_s  = bp[4] if bp is not None else float('nan')
    ec_s   = eps_cold(bp[0], bp[2], bp[4]) if bp is not None else float('nan')
    print(f"  {name:<10} {max(bests):>8.5f} {np.median(bests):>8.5f} "
          f"{len(succ):>2}/{N_SEEDS:>2}   "
          f"{int(np.median(sev)) if sev else 'N/A':>6}   "
          f"{lam_s:>7.4f} {ec_s:>8.4f}")
print(f"  {'-'*63}")
print(f"  Analytic η_max = 1 - 1/R_ω = {1-1/R_OMEGA:.6f}")
print(f"  λ_cutoff (ε=0.10, cold bath binding): {lam_10pct:.4f}")
print(f"{'='*65}")
print(f"\nAll results saved to final_result/")
