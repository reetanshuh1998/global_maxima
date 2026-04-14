"""
eta_vs_lambda_physical.py
=========================
η vs λ over the physical region [0, 0.2].
Legend shows parameter values; placed OUTSIDE the axes to avoid any overlap.
"""

import os, json
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RDIR = os.path.join(SCRIPT_DIR, '..', 'results')
PDIR = os.path.join(SCRIPT_DIR, '..', 'plots')
os.makedirs(PDIR, exist_ok=True)

with open(os.path.join(RDIR, 'optimal_parameters.json')) as f:
    opt_data = json.load(f)

# ── stable coth ────────────────────────────────────────────────────────────────
def coth(x):
    ax = abs(x)
    if ax < 1e-10: return np.sign(x) * 1e10
    if ax < 1e-3:  return 1.0/x + x/3.0 - x**3/45.0
    if ax > 20.0:  return float(np.sign(x)) * (1.0 + 2.0*np.exp(-2.0*ax))
    return float(np.cosh(x) / np.sinh(x))

# ── Eqs 8-10 ──────────────────────────────────────────────────────────────────
def eta_vs_lam(bc, bh, wc, wh, lam_arr):
    X = coth(bh * wh / 2.0)
    Y = coth(bc * wc / 2.0)
    Qh = (wh/2.0)*(X - Y) + (3.0*lam_arr/(8.0*wh**2))*(X**2 - Y**2)
    Qc = (wc/2.0)*(Y - X) + (3.0*lam_arr/(8.0*wc**2))*(Y**2 - X**2)
    W  = Qh + Qc
    valid = (Qh > 0) & (W > 0)
    return np.where(valid, np.clip(W / Qh, 0.0, 1.0), np.nan)

LAMS   = np.linspace(0.0, 0.2, 500)
COLORS = ['#7f8c8d', '#e74c3c', '#27ae60', '#8e44ad']
LWS    = [2.0, 2.0, 2.0, 2.5]

plt.rcParams.update({
    'font.family': 'serif', 'font.size': 13,
    'axes.linewidth': 1.3,
    'axes.grid': True, 'grid.alpha': 0.25, 'grid.linestyle': '--',
    'figure.dpi': 150,
})

# ── Wider figure; right margin reserved for legend ─────────────────────────────
fig, ax = plt.subplots(figsize=(12, 5.2))

handles, labels = [], []
for (name, cfg), color, lw in zip(opt_data.items(), COLORS, LWS):
    if cfg['params'] is None:
        continue
    bc, bh, wc, wh, _ = cfg['params']
    eta  = eta_vs_lam(bc, bh, wc, wh, LAMS)
    eta0 = float(eta_vs_lam(bc, bh, wc, wh, np.array([0.0]))[0])
    lab  = (rf'$\beta_c={bc:.2f},\;\beta_h={bh:.2f},$'
            '\n'
            rf'$\omega_c={wc:.3f},\;\omega_h={wh:.3f}$')
    line, = ax.plot(LAMS, eta, color=color, lw=lw)
    ax.plot(0.0, eta0, 'o', color=color, ms=7, zorder=5)
    handles.append(line)
    labels.append(lab)

ax.axvline(0.2, color='#c0392b', lw=1.8, ls='--')
ax.text(0.197, 0.025, r'$\lambda_\mathrm{max}=0.2$',
        color='#c0392b', fontsize=10, ha='right', va='bottom')

ax.set_xlabel(r'Anharmonicity parameter $\lambda$', fontsize=14)
ax.set_ylabel(r'Efficiency $\eta$', fontsize=14)
ax.set_title(
    r'$\eta(\lambda)$: Effect of Anharmonicity on Quantum Otto Cycle Efficiency'
    '\n'
    r'(physical region $\lambda \in [0,\;0.2]$; '
    r'other parameters fixed at optimal values)',
    fontsize=12, pad=10
)
ax.set_xlim(0.0, 0.20)
ax.set_ylim(bottom=0.0)

# ── Legend: entirely outside on the right ─────────────────────────────────────
leg = ax.legend(
    handles, labels,
    fontsize=9.5,
    loc='center left',
    bbox_to_anchor=(1.02, 0.5),   # 2% to the right of axes boundary
    frameon=True,
    framealpha=0.95,
    title=r'Fixed $(\beta_c,\,\beta_h,\,\omega_c,\,\omega_h)$',
    title_fontsize=10,
    borderpad=0.9,
    labelspacing=0.9,
    handlelength=2.0,
)

# Save with bbox_inches='tight' to include the external legend
out = os.path.join(PDIR, 'eta_vs_lambda_physical.png')
fig.savefig(out, dpi=200, bbox_inches='tight', bbox_extra_artists=[leg])
plt.close(fig)
print(f"Saved → {out}")
