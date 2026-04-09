"""
plot_eta_vs_params.py
=====================
Plots efficiency η as a function of each control parameter:
  1. η vs β_c   (cold inverse temperature)
  2. η vs β_h   (hot  inverse temperature)
  3. η vs ω_c   (cold frequency)
  4. η vs ω_h   (hot  frequency)

The reference / base-point used for the fixed parameters is the
globally optimal parameter set (η_max) as computed by
find_optimal_parameters.py and stored in results/optimal_parameters.json.

For each sweep one parameter is varied over its physical range while
the other four are held at the optimal values.  Multiple curves show
several λ values.

Plots saved to:  anharmonic_otto_study/plots/
"""

import os, json
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# ── Paths ─────────────────────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RDIR = os.path.join(SCRIPT_DIR, '..', 'results')
PDIR = os.path.join(SCRIPT_DIR, '..', 'plots')
os.makedirs(PDIR, exist_ok=True)

# ── Load optimal point from find_optimal_parameters results ───────────────────
OPT_JSON = os.path.join(RDIR, 'optimal_parameters.json')
with open(OPT_JSON) as f:
    opt_data = json.load(f)

# Pick the platform with the highest η_max as the global reference
best_platform = max(opt_data, key=lambda k: opt_data[k]['eta_max'])
best_cfg      = opt_data[best_platform]
opt_params    = best_cfg['params']          # [bc, bh, wc, wh, lam]
OPT_ETA       = best_cfg['eta_max']
OPT_BC, OPT_BH, OPT_WC, OPT_WH, OPT_LAM = opt_params

print(f"Reference platform : {best_platform.replace(chr(10), ' ')}")
print(f"  η_max  = {OPT_ETA:.6f}")
print(f"  β_c={OPT_BC:.4f},  β_h={OPT_BH:.4f},  ω_c={OPT_WC:.4f},  ω_h={OPT_WH:.4f},  λ={OPT_LAM:.4f}")
print()

# ── Physics ───────────────────────────────────────────────────────────────────
def coth(x):
    ax = abs(x)
    if ax < 1e-10:  return np.sign(x) * 1e10
    if ax < 1e-3:   return 1.0/x + x/3.0 - x**3/45.0
    if ax > 20.0:   return float(np.sign(x)) * (1.0 + 2.0 * np.exp(-2.0*ax))
    return float(np.cosh(x) / np.sinh(x))

vcoth = np.vectorize(coth)

def compute_eta(bc, bh, wc, wh, lam):
    """Vectorised η; returns NaN where engine conditions are violated."""
    X = vcoth(bh * wh / 2.0)
    Y = vcoth(bc * wc / 2.0)
    Qh = (wh / 2.0)*(X - Y) + (3.0*lam / (8.0*wh**2))*(X**2 - Y**2)
    Qc = (wc / 2.0)*(Y - X) + (3.0*lam / (8.0*wc**2))*(Y**2 - X**2)
    W  = Qh + Qc
    valid = (np.asarray(Qh) > 0) & (np.asarray(W) > 0)
    eta = np.where(valid, np.clip(W / Qh, 0.0, 1.0), np.nan)
    return eta

# ── λ curves ──────────────────────────────────────────────────────────────────
LAM_MAX    = 0.20
LAM_VALUES = [0.00, 0.05, 0.10, 0.15, 0.20]
LAM_COLORS = ['#2c3e50', '#2980b9', '#27ae60', '#e67e22', '#c0392b']
LAM_LABELS = [rf'$\lambda={v}$' for v in LAM_VALUES]

N_POINTS = 500

# ── Plot styling ──────────────────────────────────────────────────────────────
plt.rcParams.update({
    'font.family':      'serif',
    'font.size':        12,
    'axes.linewidth':   1.2,
    'axes.grid':        True,
    'grid.alpha':       0.3,
    'legend.framealpha': 0.9,
    'figure.dpi':       150,
})

# ── Sweep helper ──────────────────────────────────────────────────────────────
def sweep(param, x_arr, lam):
    """Compute η while sweeping one param, all others at their optimal values."""
    bc  = x_arr if param == 'bc' else OPT_BC
    bh  = x_arr if param == 'bh' else OPT_BH
    wc  = x_arr if param == 'wc' else OPT_WC
    wh  = x_arr if param == 'wh' else OPT_WH
    return compute_eta(bc, bh, wc, wh, lam)

# ── Sweep definitions ─────────────────────────────────────────────────────────
# Range chosen so physical constraints can be satisfied with the optimal
# values of the other parameters.
SWEEPS = [
    {
        'param':    'bc',
        'range':    (OPT_BH * 1.01, min(OPT_BH * 25.0, 35.0)),  # bc > bh, ratio ≤ 25
        'opt_val':  OPT_BC,
        'xlabel':   r'$\beta_c$  (cold inverse temperature)',
        'title':    r'$\eta$ vs $\beta_c$',
        'filename': 'eta_vs_beta_c.png',
    },
    {
        'param':    'bh',
        'range':    (0.05, OPT_BC / 1.01),                        # bh < bc
        'opt_val':  OPT_BH,
        'xlabel':   r'$\beta_h$  (hot inverse temperature)',
        'title':    r'$\eta$ vs $\beta_h$',
        'filename': 'eta_vs_beta_h.png',
    },
    {
        'param':    'wc',
        'range':    (0.3, OPT_WH / 1.01),                         # wc < wh
        'opt_val':  OPT_WC,
        'xlabel':   r'$\omega_c$  (cold oscillator frequency)',
        'title':    r'$\eta$ vs $\omega_c$',
        'filename': 'eta_vs_omega_c.png',
    },
    {
        'param':    'wh',
        'range':    (OPT_WC * 1.01, OPT_WC * 7.0),               # wh > wc, ratio ≤ 7
        'opt_val':  OPT_WH,
        'xlabel':   r'$\omega_h$  (hot oscillator frequency)',
        'title':    r'$\eta$ vs $\omega_h$',
        'filename': 'eta_vs_omega_h.png',
    },
]

# annotation showing which values are fixed (for individual plots)
FIX_STRS = {
    'bc': rf'Fixed: $\beta_h={OPT_BH:.3f}$,  $\omega_c={OPT_WC:.3f}$,  $\omega_h={OPT_WH:.3f}$',
    'bh': rf'Fixed: $\beta_c={OPT_BC:.3f}$,  $\omega_c={OPT_WC:.3f}$,  $\omega_h={OPT_WH:.3f}$',
    'wc': rf'Fixed: $\beta_c={OPT_BC:.3f}$,  $\beta_h={OPT_BH:.3f}$,  $\omega_h={OPT_WH:.3f}$',
    'wh': rf'Fixed: $\beta_c={OPT_BC:.3f}$,  $\beta_h={OPT_BH:.3f}$,  $\omega_c={OPT_WC:.3f}$',
}

# ── Individual plots ──────────────────────────────────────────────────────────
print("Generating individual η-vs-parameter plots …")
for sw in SWEEPS:
    x = np.linspace(*sw['range'], N_POINTS)
    fig, ax = plt.subplots(figsize=(7.5, 5.2))

    for lam, color, label in zip(LAM_VALUES, LAM_COLORS, LAM_LABELS):
        eta = sweep(sw['param'], x, lam)
        ax.plot(x, eta, color=color, lw=2.0, label=label)

    # Mark the optimal value of the swept variable
    ax.axvline(sw['opt_val'], ls='--', color='black', lw=1.6,
               label=f'Optimal {sw["xlabel"].split("(")[0].strip()} = {sw["opt_val"]:.3f}')

    # Mark the η_max value as a horizontal line
    ax.axhline(OPT_ETA, ls=':', color='gray', lw=1.3,
               label=rf'$\eta_\mathrm{{max}}$ = {OPT_ETA:.4f}')

    ax.set_xlabel(sw['xlabel'], fontsize=13)
    ax.set_ylabel(r'$\eta$  (efficiency)', fontsize=13)
    ax.set_title(sw['title'] + '\n'
                 r'(other params at globally optimal point, $\eta_\mathrm{max}$='
                 f'{OPT_ETA:.4f})',
                 fontsize=12, pad=8)
    ax.set_ylim(bottom=0)
    ax.legend(fontsize=9.5, loc='best')

    # Fixed-parameter annotation
    ax.text(0.02, 0.97, FIX_STRS[sw['param']],
            transform=ax.transAxes, fontsize=8.5, va='top',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.85))

    fig.tight_layout()
    out = os.path.join(PDIR, sw['filename'])
    fig.savefig(out, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {out}")

# ── Combined 2×2 figure ───────────────────────────────────────────────────────
print("\nGenerating combined 2×2 figure …")
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle(
    r'Efficiency $\eta$ as a Function of Each System Parameter'
    '\n(Other parameters fixed at globally optimal point, '
    + best_platform.replace('\n', ' ')
    + rf', $\eta_\mathrm{{max}}$ = {OPT_ETA:.4f})',
    fontsize=13, y=1.01
)

for ax, sw in zip(axes.flat, SWEEPS):
    x = np.linspace(*sw['range'], N_POINTS)
    for lam, color, label in zip(LAM_VALUES, LAM_COLORS, LAM_LABELS):
        ax.plot(x, sweep(sw['param'], x, lam), color=color, lw=1.8)

    ax.axvline(sw['opt_val'], ls='--', color='black', lw=1.4,
               label=f'opt = {sw["opt_val"]:.3f}')
    ax.axhline(OPT_ETA, ls=':', color='gray', lw=1.1)
    ax.set_xlabel(sw['xlabel'], fontsize=11)
    ax.set_ylabel(r'$\eta$', fontsize=11)
    ax.set_title(sw['title'], fontsize=12)
    ax.set_ylim(bottom=0)
    ax.legend(fontsize=8)
    ax.text(0.02, 0.97, FIX_STRS[sw['param']],
            transform=ax.transAxes, fontsize=7.5, va='top',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

# Shared λ legend
legend_handles = [
    Line2D([0], [0], color=c, lw=2, label=l)
    for c, l in zip(LAM_COLORS, LAM_LABELS)
]
legend_handles += [
    Line2D([0], [0], color='black', lw=1.4, ls='--', label='optimal value'),
    Line2D([0], [0], color='gray',  lw=1.1, ls=':',  label=rf'$\eta_\mathrm{{max}}$ = {OPT_ETA:.4f}'),
]
fig.legend(handles=legend_handles, loc='lower center',
           ncol=len(LAM_VALUES) + 2, fontsize=9.5,
           bbox_to_anchor=(0.5, -0.04), framealpha=0.9)

fig.tight_layout()
out_combined = os.path.join(PDIR, 'eta_vs_all_params.png')
fig.savefig(out_combined, dpi=200, bbox_inches='tight')
plt.close(fig)
print(f"  Saved: {out_combined}")

print(f"\nDone. 5 plots saved to plots/")
print(f"Reference: {best_platform.replace(chr(10), ' ')} — η_max = {OPT_ETA:.6f}")
print(f"  β_c={OPT_BC:.4f}, β_h={OPT_BH:.4f}, ω_c={OPT_WC:.4f}, ω_h={OPT_WH:.4f}, λ={OPT_LAM:.4f}")
