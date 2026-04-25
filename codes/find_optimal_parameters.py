"""
find_optimal_parameters.py
===========================
For each experimental platform, finds the 5-parameter set
(beta_c, beta_h, omega_c, omega_h, lambda) that maximises eta
within that platform's ratio bounds.

Platforms (from literature):
  1. Reference paper   R_omega=1.5, R_beta=2   [paper Fig.2]
  2. Superconducting   R_omega=2.0, R_beta=6   [Peterson et al., Nat.Commun. 2019]
  3. Trapped ions      R_omega=4.0, R_beta=12  [Rossnagel et al., Science 2016]
  4. NV centres        R_omega=7.0, R_beta=25  [Klatzow et al., PRL 2019]

Saves: results/optimal_parameters.json
       plots/fig6_platform_comparison.png
"""
import signal, sys, os, json
signal.signal(signal.SIGINT, signal.SIG_IGN)
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.optimize import minimize, differential_evolution

LAM_MAX = 0.2

# ── Stable coth ───────────────────────────────────────────────────────────────
def coth(x):
    ax = abs(x)
    if ax < 1e-10: return np.sign(x)*1e10
    if ax < 1e-3:  return 1/x + x/3 - x**3/45
    if ax > 20:    return float(np.sign(x))*(1 + 2*np.exp(-2*ax))
    return 1/np.tanh(x)

# ── Eqs 8-10 from reference paper ─────────────────────────────────────────────
def compute_eta_full(bc, bh, wc, wh, lam):
    """Returns (eta, Q_h, Q_c, W_ext)."""
    X = coth(bh*wh/2); Y = coth(bc*wc/2)
    Q_h = (wh/2)*(X-Y) + (3*lam/(8*wh**2))*(X**2-Y**2)
    Q_c = (wc/2)*(Y-X) + (3*lam/(8*wc**2))*(Y**2-X**2)
    W   = Q_h + Q_c
    if Q_h <= 0 or W <= 0: return 0., Q_h, Q_c, W
    return float(np.clip(W/Q_h, 0, 1)), Q_h, Q_c, W

# ── Platform definitions ───────────────────────────────────────────────────────
PLATFORMS = {
    'Reference paper\n(Fig.2)': {
        'R_omega': 1.5, 'R_beta': 2.0,
        'color': '#95a5a6',
        'cite': 'Your reference paper, Fig. 2'
    },
    'Superconducting\nqubits': {
        'R_omega': 2.0, 'R_beta': 6.0,
        'color': '#e74c3c',
        'cite': 'Peterson et al., Nat. Commun. 2019'
    },
    'Trapped\nions': {
        'R_omega': 4.0, 'R_beta': 12.0,
        'color': '#2ecc71',
        'cite': 'Roßnagel et al., Science 2016'
    },
    'NV centres\n(diamond)': {
        'R_omega': 7.0, 'R_beta': 25.0,
        'color': '#9b59b6',
        'cite': 'Klatzow et al., PRL 2019'
    },
}

# ── Optimization per platform ─────────────────────────────────────────────────
def find_optimal(R_omega, R_beta, n_restarts=40, seed=0):
    """SLSQP multi-start to maximise eta within platform bounds."""
    rng = np.random.default_rng(seed)
    best_eta, best_params = 0.0, None

    # Box bounds for SLSQP: x = [bc, bh, wc, wh, lam]
    LO = np.array([0.5,  0.05, 0.5,  1.0,  0.0])
    HI = np.array([30.,  15.,  10.,  30.,  LAM_MAX])

    # Constraints
    cons = [
        {'type':'ineq','fun': lambda x: x[3] - x[2] - 0.05},        # wh > wc
        {'type':'ineq','fun': lambda x: x[0] - x[1] - 0.05},        # bc > bh
        {'type':'ineq','fun': lambda x: x[0]*x[2] - x[1]*x[3] - 0.01},  # engine mode
        {'type':'ineq','fun': lambda x: R_omega - x[3]/max(x[2],1e-6)},  # freq ratio cap
        {'type':'ineq','fun': lambda x: R_beta  - x[0]/max(x[1],1e-6)},  # beta ratio cap
        {'type':'ineq','fun': lambda x: LAM_MAX - x[4]},
    ]

    def obj(x):
        e, *_ = compute_eta_full(*x)
        return -e

    for i in range(n_restarts):
        # Smart initialization: sample near the ratio boundaries
        r_w = rng.uniform(R_omega*0.7, R_omega*0.99)
        r_b = rng.uniform(R_beta *0.7, R_beta *0.99)
        wc  = rng.uniform(LO[2], min(HI[2], 5.0))
        wh  = min(wc * r_w, HI[3])
        bh  = rng.uniform(LO[1], min(HI[1], 5.0))
        bc  = min(bh * r_b, HI[0])
        if bc*wc <= bh*wh: bc = bh*wh/wc + 0.1
        lam = rng.uniform(0, LAM_MAX*0.5)
        x0  = np.clip([bc, bh, wc, wh, lam], LO, HI)
        try:
            res = minimize(obj, x0, method='SLSQP', bounds=list(zip(LO,HI)),
                           constraints=cons, options={'maxiter':400,'ftol':1e-12})
            if -res.fun > best_eta:
                best_eta = -res.fun
                best_params = res.x.copy()
        except Exception:
            pass

    return best_eta, best_params

# ── Run ────────────────────────────────────────────────────────────────────────
print("Finding optimal parameters for each platform...\n")
print(f"{'Platform':<30} {'R_ω':<6} {'R_β':<6} {'η_max':<8} "
      f"{'βc':>7} {'βh':>7} {'ωc':>7} {'ωh':>7} {'λ':>7}")
print("-"*85)

results = {}
for name, cfg in PLATFORMS.items():
    ro, rb = cfg['R_omega'], cfg['R_beta']
    eta_max, params = find_optimal(ro, rb, n_restarts=60)
    cfg['eta_max']  = eta_max
    cfg['params']   = params.tolist() if params is not None else None
    cfg['eta_analytical'] = 1 - 1/ro  # harmonic bound
    short = name.replace('\n',' ')
    if params is not None:
        bc,bh,wc,wh,lam = params
        print(f"{short:<30} {ro:<6} {rb:<6} {eta_max:<8.4f} "
              f"{bc:>7.3f} {bh:>7.3f} {wc:>7.3f} {wh:>7.3f} {lam:>7.4f}")
    else:
        print(f"{short:<30} {ro:<6} {rb:<6} {'FAILED'}")
    results[name] = cfg

# ── Save JSON ──────────────────────────────────────────────────────────────────
RDIR = os.path.join(os.path.dirname(__file__), '..', 'results')
os.makedirs(RDIR, exist_ok=True)
with open(f'{RDIR}/optimal_parameters.json', 'w') as f:
    json.dump(results, f, indent=2)
print(f"\nSaved → results/optimal_parameters.json")

# ── Plot: fig6 platform comparison ────────────────────────────────────────────
PDIR = os.path.join(os.path.dirname(__file__), '..', 'plots')
os.makedirs(PDIR, exist_ok=True)

platforms = list(PLATFORMS.keys())
colors    = [cfg['color'] for cfg in PLATFORMS.values()]
eta_vals  = [cfg['eta_max'] for cfg in PLATFORMS.values()]
eta_analy = [cfg['eta_analytical'] for cfg in PLATFORMS.values()]

fig, axes = plt.subplots(1, 3, figsize=(16, 6))
fig.suptitle('Optimal Efficiency by Experimental Platform\n'
             '(5-parameter SLSQP | Eqs 8–10 of reference paper)', fontsize=13)

# Panel 1: eta_max per platform
ax = axes[0]
bars = ax.bar(platforms, eta_vals, color=colors, edgecolor='k', linewidth=0.8, alpha=0.9)
ax.plot(platforms, eta_analy, 'k--o', ms=8, lw=2, label=r'Harmonic bound $1-1/R_\omega$')
for bar, ev, ea in zip(bars, eta_vals, eta_analy):
    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.005,
            f'{ev:.4f}', ha='center', fontsize=9, fontweight='bold')
ax.set_ylabel(r'Maximum $\eta$', fontsize=12)
ax.set_ylim(0, 1); ax.grid(axis='y', alpha=0.35)
ax.legend(fontsize=9)
ax.set_title('Achieved η_max per Platform', fontsize=11)
ax.tick_params(axis='x', labelsize=8)

# Panel 2: optimal parameter radar-style bar chart
ax2 = axes[1]
param_names = [r'$\beta_c$', r'$\beta_h$', r'$\omega_c$', r'$\omega_h$', r'$\alpha$']
x   = np.arange(len(param_names))
w   = 0.18
for i, (name, cfg) in enumerate(PLATFORMS.items()):
    if cfg['params'] is not None:
        # Normalise params for display
        p = np.array(cfg['params'])
        p_norm = p / np.array([30, 15, 10, 30, LAM_MAX])
        ax2.bar(x + i*w - 1.5*w, p_norm, w, label=name.replace('\n',' '),
                color=cfg['color'], edgecolor='k', linewidth=0.5, alpha=0.85)
ax2.set_xticks(x); ax2.set_xticklabels(param_names, fontsize=12)
ax2.set_ylabel('Normalised value (param / box-max)', fontsize=10)
ax2.set_ylim(0, 1.05); ax2.grid(axis='y', alpha=0.35)
ax2.legend(fontsize=7, loc='upper left')
ax2.set_title('Optimal Parameters (Normalised)', fontsize=11)

# Panel 3: ratio analysis — how close to the cap boundaries
ax3 = axes[2]
ratio_omega = []
ratio_beta  = []
labels3     = []
for name, cfg in PLATFORMS.items():
    if cfg['params'] is not None:
        bc,bh,wc,wh,lam = cfg['params']
        ratio_omega.append(wh/wc / cfg['R_omega'])   # fraction of cap used
        ratio_beta.append( bc/bh / cfg['R_beta'])
    else:
        ratio_omega.append(0); ratio_beta.append(0)
    labels3.append(name.replace('\n',' '))

x3 = np.arange(len(labels3))
ax3.bar(x3 - 0.2, ratio_omega, 0.35, color=colors, edgecolor='k',
        linewidth=0.8, alpha=0.85, label=r'$(\omega_h/\omega_c) / R_\omega$')
ax3.bar(x3 + 0.2, ratio_beta,  0.35, color=colors, edgecolor='k',
        linewidth=0.8, alpha=0.45, hatch='//', label=r'$(\beta_c/\beta_h) / R_\beta$')
ax3.axhline(1.0, ls='--', color='red', lw=1.5, label='Cap boundary')
ax3.set_xticks(x3); ax3.set_xticklabels(labels3, fontsize=7.5)
ax3.set_ylabel('Fraction of cap used', fontsize=11)
ax3.set_ylim(0, 1.15); ax3.grid(axis='y', alpha=0.35)
ax3.legend(fontsize=8)
ax3.set_title('Binding Constraint Analysis', fontsize=11)

plt.tight_layout()
fig.savefig(f'{PDIR}/fig6_platform_comparison.png', dpi=200, bbox_inches='tight')
plt.close()
print(f"Saved: plots/fig6_platform_comparison.png")
