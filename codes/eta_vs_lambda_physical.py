"""
eta_vs_lambda_physical.py
=========================
η vs λ over the physical region [0, 0.2] for the globally optimal
parameter set (highest η_max at λ=0).
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

# ── Pick platform with highest η_max (purple curve) ────────────────────────────
best = max(opt_data.values(), key=lambda c: c['eta_max'])
bc, bh, wc, wh, _ = best['params']
eta0_harmonic      = best['eta_max']

print(f"Selected: β_c={bc:.4f}, β_h={bh:.4f}, ω_c={wc:.4f}, ω_h={wh:.4f}")
print(f"η at λ=0: {eta0_harmonic:.6f}")

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

LAMS = np.linspace(0.0, 0.2, 600)
eta  = eta_vs_lam(bc, bh, wc, wh, LAMS)

# ── Plot styling (PRD Style) ──────────────────────────────────────────────────
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times', 'Times New Roman', 'DejaVu Serif'],
    'mathtext.fontset': 'stix',
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.linewidth': 1.0,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 9,
    'axes.grid': True,
    'grid.alpha': 0.2,
    'grid.linestyle': '--',
    'figure.dpi': 300,
    'savefig.dpi': 300,
})

fig, ax = plt.subplots(figsize=(5, 3.8))

# ── Main curve ─────────────────────────────────────────────────────────────────
label = (rf'$\beta_c={bc:.2f},\;\beta_h={bh:.2f},$'
         rf'$\;\omega_c={wc:.3f},\;\omega_h={wh:.3f}$')
ax.plot(LAMS, eta, color='#8e44ad', lw=2.0, label=label)

# ── Harmonic limit dot ─────────────────────────────────────────────────────────
ax.plot(0.0, eta0_harmonic, 'o', color='#8e44ad', ms=6, zorder=5,
        label=rf'Harmonic limit $\eta_0 = {eta0_harmonic:.4f}$')
ax.axhline(eta0_harmonic, ls=':', color='#8e44ad', lw=1.0, alpha=0.5)

# ── Physical cap ──────────────────────────────────────────────────────────────
ax.axvline(0.2, color='#c0392b', lw=1.5, ls='--',
           label=r'Physical cap $\alpha_\mathrm{max}=0.2$')

# ── Annotate η drop ───────────────────────────────────────────────────────────
eta_at_02 = float(eta_vs_lam(bc, bh, wc, wh, np.array([0.2]))[0])
drop_pct   = (eta0_harmonic - eta_at_02) / eta0_harmonic * 100
# Positioned at the center of the plot
ax.text(0.5, 0.5, 
        rf'$\Delta\eta = {eta0_harmonic - eta_at_02:.4f}$ ({drop_pct:.1f}% drop)',
        transform=ax.transAxes, fontsize=11, color='#2c3e50',
        bbox=dict(facecolor='white', alpha=0.8, edgecolor='none', boxstyle='round,pad=0.3'),
        ha='center', va='center')

# ── Labels ────────────────────────────────────────────────────────────────────
ax.set_xlabel(r'$\alpha$', fontsize=13)
ax.set_ylabel(r'Efficiency', fontsize=13)

ax.set_xlim(0.0, 0.205)
ax.set_ylim(bottom=0.0)
ax.legend(fontsize=9, loc='lower left', framealpha=0.9)

fig.tight_layout()
out_png = os.path.join(PDIR, 'eta_vs_lambda_physical.png')
out_pdf = os.path.join(PDIR, 'eta_vs_lambda_physical.pdf')

fig.savefig(out_png, bbox_inches='tight')
fig.savefig(out_pdf, bbox_inches='tight')
plt.close(fig)

print(f"Saved → {out_png}")
print(f"Saved → {out_pdf}")
print(f"η at λ=0.0: {eta0_harmonic:.6f}")
print(f"η at λ=0.2: {eta_at_02:.6f}  (drop = {drop_pct:.1f}%)")

