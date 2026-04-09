"""
plot_eta_vs_lambda.py
=====================
Plots efficiency η as a function of the anharmonicity parameter λ,
with (β_c, β_h, ω_c, ω_h) fixed at the globally optimal values
(NV-centre platform, loaded from results/optimal_parameters.json).

Panel 1 — Main plot:
  • η(λ)  on the left y-axis
  • ε_cold(λ) and ε_hot(λ) on the right y-axis  [perturbation validity ratios]
  • Three horizontal threshold lines: ε = 0.01, 0.10, 0.30
  • Vertical shaded regions colour the three validity zones

Panel 2 — Zoom into the physically valid regime (ε_cold < 0.30)

The dimensionless perturbation ratio is (thermally averaged):
    ε(ω, β, λ) = (3λ / 4ω³) · coth(βω/2)
Perturbation theory is trustworthy when ε ≪ 1.

Saved figures:
  plots/eta_vs_lambda.png
"""

import signal
signal.signal(signal.SIGINT, signal.SIG_IGN)

import os, json
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ── Paths ─────────────────────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RDIR = os.path.join(SCRIPT_DIR, '..', 'results')
PDIR = os.path.join(SCRIPT_DIR, '..', 'plots')
os.makedirs(PDIR, exist_ok=True)

# ── Load globally optimal parameters ─────────────────────────────────────────
with open(os.path.join(RDIR, 'optimal_parameters.json')) as f:
    opt_data = json.load(f)

best_platform = max(opt_data, key=lambda k: opt_data[k]['eta_max'])
cfg           = opt_data[best_platform]
OPT_BC, OPT_BH, OPT_WC, OPT_WH, OPT_LAM = cfg['params']
OPT_ETA = cfg['eta_max']
R_OMEGA = cfg['R_omega']

print(f"Reference platform : {best_platform.replace(chr(10), ' ')}")
print(f"  η_max  = {OPT_ETA:.6f}")
print(f"  β_c={OPT_BC:.4f}, β_h={OPT_BH:.4f}, ω_c={OPT_WC:.4f}, ω_h={OPT_WH:.4f}")

# ── Physics ───────────────────────────────────────────────────────────────────
def coth(x):
    ax = abs(x)
    if ax < 1e-10: return np.sign(x) * 1e10
    if ax < 1e-3:  return 1.0/x + x/3.0 - x**3/45.0
    if ax > 20.0:  return float(np.sign(x)) * (1.0 + 2.0*np.exp(-2.0*ax))
    return float(np.cosh(x) / np.sinh(x))

vcoth = np.vectorize(coth)

def compute_eta_vec(lam_arr):
    """Vectorised η over an array of λ, all other params at optimal values."""
    bc, bh, wc, wh = OPT_BC, OPT_BH, OPT_WC, OPT_WH
    X = coth(bh * wh / 2.0)
    Y = coth(bc * wc / 2.0)
    # Eqs. 8-10
    Qh = (wh/2.0)*(X - Y) + (3.0*lam_arr/(8.0*wh**2))*(X**2 - Y**2)
    Qc = (wc/2.0)*(Y - X) + (3.0*lam_arr/(8.0*wc**2))*(Y**2 - X**2)
    W  = Qh + Qc
    valid = (Qh > 0) & (W > 0)
    return np.where(valid, np.clip(W / Qh, 0.0, 1.0), np.nan)

def eps_cold(lam_arr):
    """Perturbation ratio at cold bath: ε_c = (3λ/4ωc³)·coth(βc·ωc/2)."""
    return (3.0 * lam_arr / (4.0 * OPT_WC**3)) * coth(OPT_BC * OPT_WC / 2.0)

def eps_hot(lam_arr):
    """Perturbation ratio at hot bath: ε_h = (3λ/4ωh³)·coth(βh·ωh/2)."""
    return (3.0 * lam_arr / (4.0 * OPT_WH**3)) * coth(OPT_BH * OPT_WH / 2.0)

# ── λ_cutoff at each threshold ────────────────────────────────────────────────
X_cold = coth(OPT_BC * OPT_WC / 2.0)
X_hot  = coth(OPT_BH * OPT_WH / 2.0)
eps_per_lam_cold = (3.0 / (4.0 * OPT_WC**3)) * X_cold
eps_per_lam_hot  = (3.0 / (4.0 * OPT_WH**3)) * X_hot

print(f"\nPerturbation ratio (ε/λ):")
print(f"  Cold bath: {eps_per_lam_cold:.4f}  →  λ_cutoff(1%)={0.01/eps_per_lam_cold:.4f},"
      f" λ_cutoff(10%)={0.10/eps_per_lam_cold:.4f}, λ_cutoff(30%)={0.30/eps_per_lam_cold:.4f}")
print(f"  Hot  bath: {eps_per_lam_hot:.4f}   →  λ_cutoff(10%)={0.10/eps_per_lam_hot:.4f}")

lam_1pct  = 0.01  / eps_per_lam_cold   # ε_cold = 1%
lam_10pct = 0.10  / eps_per_lam_cold   # ε_cold = 10%
lam_30pct = 0.30  / eps_per_lam_cold   # ε_cold = 30%

# ── Lambda arrays ─────────────────────────────────────────────────────────────
lam_max_plot = min(3 * lam_30pct, 1.0)   # plot up to 3× the 30% cutoff or λ=1
lam_arr      = np.linspace(0.0, lam_max_plot, 500)

eta_arr  = compute_eta_vec(lam_arr)
ec_arr   = eps_cold(lam_arr)
eh_arr   = eps_hot(lam_arr)

# ── Plot styling ──────────────────────────────────────────────────────────────
plt.rcParams.update({
    'font.family':       'serif',
    'font.size':         12,
    'axes.linewidth':    1.2,
    'axes.grid':         True,
    'grid.alpha':        0.25,
    'legend.framealpha': 0.9,
    'figure.dpi':        150,
})

# Validity zone colours
C_GREEN  = '#d5f5e3'   # ε < 0.01  — fully valid
C_YELLOW = '#fef9e7'   # 0.01 < ε < 0.10 — first-order valid
C_ORANGE = '#fdebd0'   # 0.10 < ε < 0.30 — qualitative only
C_RED    = '#fadbd8'   # ε > 0.30  — perturbation broken

# ─────────────────────────────────────────────────────────────────────────────
# FIGURE — main full plot
# ─────────────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(15, 6.5))
fig.suptitle(
    r'Efficiency $\eta(\lambda)$ and Perturbation Validity $\varepsilon(\lambda)$'
    '\n'
    r'(Optimal $\beta_c, \beta_h, \omega_c, \omega_h$ fixed at NV-centre optimum, '
    rf'$\eta_\mathrm{{max}}={OPT_ETA:.4f}$)',
    fontsize=13, y=1.01
)

# ─── Panel 1: full range ─────────────────────────────────────────────────────
ax1 = axes[0]
ax1r = ax1.twinx()

# Validity zone shading (based on ε_cold, the binding side)
for xl, xr, color in [
    (0,           lam_1pct,  C_GREEN),
    (lam_1pct,    lam_10pct, C_YELLOW),
    (lam_10pct,   lam_30pct, C_ORANGE),
    (lam_30pct,   lam_max_plot, C_RED),
]:
    ax1.axvspan(xl, min(xr, lam_max_plot), alpha=0.55, color=color, zorder=0)

# η curve (left axis)
ax1.plot(lam_arr, eta_arr, color='#2980b9', lw=2.5, zorder=5,
         label=r'$\eta(\lambda)$')
ax1.axhline(OPT_ETA, ls=':', color='#2980b9', lw=1.2, alpha=0.6,
            label=rf'$\eta_0 = {OPT_ETA:.4f}$ (harmonic limit)')

# ε curves (right axis)
ax1r.plot(lam_arr, ec_arr, color='#e74c3c', lw=2.0, ls='--', zorder=6,
          label=r'$\varepsilon_\mathrm{cold}(\lambda)$')
ax1r.plot(lam_arr, eh_arr, color='#27ae60', lw=2.0, ls='-.', zorder=6,
          label=r'$\varepsilon_\mathrm{hot}(\lambda)$')

# threshold horizontal lines on ε axis
for thresh, ls, label in [
    (0.01, ':',  r'$\varepsilon = 0.01$ (weak coupling)'),
    (0.10, '--', r'$\varepsilon = 0.10$ (first-order valid)'),
    (0.30, '-',  r'$\varepsilon = 0.30$ (qualitative only)'),
]:
    ax1r.axhline(thresh, ls=ls, color='gray', lw=1.3, alpha=0.85, label=label)

# vertical cutoff lines
for lam_cut, label, color in [
    (lam_1pct,  rf'$\lambda_{{1\%}}={lam_1pct:.3f}$',  '#27ae60'),
    (lam_10pct, rf'$\lambda_{{10\%}}={lam_10pct:.3f}$','#e67e22'),
    (lam_30pct, rf'$\lambda_{{30\%}}={lam_30pct:.3f}$','#c0392b'),
]:
    ax1.axvline(lam_cut, ls=':', color=color, lw=1.5, zorder=4)
    ax1.text(lam_cut + lam_max_plot*0.012, 0.01, label,
             color=color, fontsize=8.5, va='bottom', rotation=90, zorder=7)

ax1.set_xlabel(r'$\lambda$ (anharmonicity)', fontsize=13)
ax1.set_ylabel(r'$\eta$ (efficiency)', fontsize=13, color='#2980b9')
ax1r.set_ylabel(r'$\varepsilon = \frac{3\lambda}{4\omega^3}\coth\!\left(\frac{\beta\omega}{2}\right)$',
                fontsize=12, color='#7f8c8d')
ax1.set_xlim(0, lam_max_plot)
ax1.set_ylim(bottom=0)
ax1r.set_ylim(bottom=0)
ax1.tick_params(axis='y', colors='#2980b9')
ax1.set_title('Full range', fontsize=11)

# Combined legend
lines1, labs1 = ax1.get_legend_handles_labels()
lines2, labs2 = ax1r.get_legend_handles_labels()
zone_patches = [
    mpatches.Patch(color=C_GREEN,  alpha=0.7, label=r'Fully valid ($\varepsilon < 0.01$)'),
    mpatches.Patch(color=C_YELLOW, alpha=0.7, label=r'1st-order valid ($\varepsilon < 0.10$)'),
    mpatches.Patch(color=C_ORANGE, alpha=0.7, label=r'Qualitative only ($\varepsilon < 0.30$)'),
    mpatches.Patch(color=C_RED,    alpha=0.7, label=r'Perturbation broken ($\varepsilon > 0.30$)'),
]
ax1.legend(lines1 + lines2 + zone_patches, labs1 + labs2 + [p.get_label() for p in zone_patches],
           fontsize=7.5, loc='upper right', ncol=2)

# Fixed-param annotation
ax1.text(0.02, 0.96,
         rf'Fixed: $\beta_c={OPT_BC:.2f}$, $\beta_h={OPT_BH:.3f}$, '
         rf'$\omega_c={OPT_WC:.3f}$, $\omega_h={OPT_WH:.3f}$'
         '\n(NV-centre optimal point)',
         transform=ax1.transAxes, fontsize=8.5, va='top',
         bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9))

# ─── Panel 2: zoom into valid regime ─────────────────────────────────────────
ax2 = axes[1]
ax2r = ax2.twinx()

lam_zoom = lam_arr[lam_arr <= lam_30pct * 1.1]

# zone shading
for xl, xr, color in [
    (0,        lam_1pct,  C_GREEN),
    (lam_1pct, lam_10pct, C_YELLOW),
    (lam_10pct,lam_30pct, C_ORANGE),
]:
    ax2.axvspan(xl, min(xr, lam_zoom.max()), alpha=0.55, color=color, zorder=0)

ax2.plot(lam_zoom, compute_eta_vec(lam_zoom),
         color='#2980b9', lw=2.5, zorder=5, label=r'$\eta(\lambda)$')
ax2.axhline(OPT_ETA, ls=':', color='#2980b9', lw=1.2, alpha=0.6)

ax2r.plot(lam_zoom, eps_cold(lam_zoom), color='#e74c3c', lw=2.0, ls='--',
          label=r'$\varepsilon_\mathrm{cold}$')
ax2r.plot(lam_zoom, eps_hot(lam_zoom),  color='#27ae60', lw=2.0, ls='-.',
          label=r'$\varepsilon_\mathrm{hot}$')

for thresh, ls in [(0.01, ':'), (0.10, '--'), (0.30, '-')]:
    ax2r.axhline(thresh, ls=ls, color='gray', lw=1.3, alpha=0.8)

for lam_cut, label, color in [
    (lam_1pct,  rf'$\lambda_{{1\%}}={lam_1pct:.3f}$',  '#27ae60'),
    (lam_10pct, rf'$\lambda_{{10\%}}={lam_10pct:.3f}$','#e67e22'),
]:
    ax2.axvline(lam_cut, ls=':', color=color, lw=1.5)
    ax2.text(lam_cut + lam_zoom.max()*0.015, 0.005, label,
             color=color, fontsize=8.5, va='bottom', rotation=90)

ax2.set_xlabel(r'$\lambda$ (anharmonicity)', fontsize=13)
ax2.set_ylabel(r'$\eta$ (efficiency)', fontsize=13, color='#2980b9')
ax2r.set_ylabel(r'$\varepsilon(\lambda)$', fontsize=12, color='#7f8c8d')
ax2.set_xlim(0, lam_zoom.max())
ax2.set_ylim(bottom=0)
ax2r.set_ylim(bottom=0)
ax2.tick_params(axis='y', colors='#2980b9')
ax2.set_title(r'Zoom: perturbatively valid regime ($\varepsilon_\mathrm{cold} \leq 0.30$)',
              fontsize=11)

lines2a, labs2a = ax2.get_legend_handles_labels()
lines2b, labs2b = ax2r.get_legend_handles_labels()
ax2.legend(lines2a + lines2b, labs2a + labs2b, fontsize=9, loc='upper right')

fig.tight_layout()
out = os.path.join(PDIR, 'eta_vs_lambda.png')
fig.savefig(out, dpi=200, bbox_inches='tight')
plt.close(fig)
print(f"\nSaved: {out}")
print(f"\nKey λ cutoffs (cold bath is the binding constraint):")
print(f"  λ_cutoff(ε=0.01) = {lam_1pct:.4f}   [fully valid, ~1% correction]")
print(f"  λ_cutoff(ε=0.10) = {lam_10pct:.4f}   [first-order valid, ~10% correction]")
print(f"  λ_cutoff(ε=0.30) = {lam_30pct:.4f}   [qualitative only, ~30% correction]")
print(f"  Hot bath barely constrained: λ_cutoff(ε=0.10, hot) = {0.10/eps_per_lam_hot:.3f}")
