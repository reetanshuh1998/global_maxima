"""
plot_constraint_justification.py  (v2 — updated to match platform data)
=========================================================================
3-panel figure:
  Panel 1: eta_max = 1 - 1/R_omega  vs R_omega  (range 1–8, platforms marked)
  Panel 2: eta_Carnot = 1 - 1/R_beta vs R_beta   (range 1–28, platforms marked)
  Panel 3: 2D heatmap eta_max(R_omega, R_beta)   (R_omega 1–8, R_beta 1–28)

Platform points exactly match find_optimal_parameters.py:
  Reference paper   : R_omega=1.5, R_beta=2
  Superconducting   : R_omega=2.0, R_beta=6
  Trapped ions      : R_omega=4.0, R_beta=12
  NV centres        : R_omega=7.0, R_beta=25
"""
import signal, os
signal.signal(signal.SIGINT, signal.SIG_IGN)

import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

PDIR = os.path.join(os.path.dirname(__file__), '..', 'plots')
os.makedirs(PDIR, exist_ok=True)

# ── Platform data (matches find_optimal_parameters.py exactly) ────────────────
PLATFORMS = {
    'Reference\npaper (Fig.2)': {'R_omega':1.5, 'R_beta':2.0,  'color':'#95a5a6', 'marker':'o'},
    'Superconducting\nqubits':   {'R_omega':2.0, 'R_beta':6.0,  'color':'#e74c3c', 'marker':'s'},
    'Trapped\nions':             {'R_omega':4.0, 'R_beta':12.0, 'color':'#2ecc71', 'marker':'^'},
    'NV centres\n(diamond)':     {'R_omega':7.0, 'R_beta':25.0, 'color':'#9b59b6', 'marker':'D'},
}

# ── Our benchmark choice ──────────────────────────────────────────────────────
OUR_R_OMEGA = 7.0  # mesh maximum
OUR_R_BETA  = 25.0  # NV centres

# ── Axis ranges — chosen to span all platform points with headroom ─────────────
R_omega_max_plot = 8.5   # covers NV R_omega=7 with headroom
R_beta_max_plot  = 28.0  # covers NV R_beta=25 with headroom

# Fine x-axes
ro_line = np.linspace(1.01, R_omega_max_plot, 400)
rb_line = np.linspace(1.01, R_beta_max_plot,  400)

# 2D grids
Ro = np.linspace(1.01, R_omega_max_plot, 150)
Rb = np.linspace(1.01, R_beta_max_plot,  150)
Ro_g, Rb_g = np.meshgrid(Ro, Rb)
# eta_max is the smaller of the two bounds
eta_grid = np.where(Ro_g <= Rb_g, 1 - 1/Ro_g, 1 - 1/Rb_g)

# ── Figure ────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(17, 5.5))
fig.suptitle(r'Maximum Efficiency as a Function of Constraint Caps $R_\omega$ and $R_\beta$'
             '\n(Platform points from literature — see References)',
             fontsize=13, y=1.02)

# ── Panel 1: eta_max vs R_omega ───────────────────────────────────────────────
ax = axes[0]
ax.plot(ro_line, 1 - 1/ro_line, 'b-', lw=2.5, label=r'$\eta_\mathrm{max}=1-1/R_\omega$ (harmonic)')
# Carnot ceiling for each platform's R_beta
for name, cfg in PLATFORMS.items():
    ax.scatter(cfg['R_omega'], 1-1/cfg['R_omega'],
               color=cfg['color'], marker=cfg['marker'], s=90, zorder=6,
               label=name.replace('\n',' '))
# Our benchmark
ax.axvline(OUR_R_OMEGA, ls='--', color='purple', lw=1.8, alpha=0.8)
ax.axhline(1-1/OUR_R_OMEGA, ls='--', color='purple', lw=1.2, alpha=0.5)
ax.annotate(f'Our benchmark\n$R_\\omega$={OUR_R_OMEGA:.0f}, η={1-1/OUR_R_OMEGA:.3f}',
            xy=(OUR_R_OMEGA, 1-1/OUR_R_OMEGA),
            xytext=(OUR_R_OMEGA+0.4, 0.68), fontsize=8.5, color='purple',
            arrowprops=dict(arrowstyle='->', color='purple'))
ax.set_xlabel(r'$R_\omega = \omega_h / \omega_c$ cap', fontsize=12)
ax.set_ylabel(r'Maximum $\eta$', fontsize=12)
ax.set_xlim(1, R_omega_max_plot)
ax.set_ylim(0, 1)
ax.legend(fontsize=7.5, loc='lower right')
ax.grid(alpha=0.35)
ax.set_title(r'$\eta_\mathrm{max}$ vs Frequency Ratio Cap', fontsize=11)

# ── Panel 2: eta_Carnot vs R_beta ─────────────────────────────────────────────
ax2 = axes[1]
ax2.plot(rb_line, 1 - 1/rb_line, 'g-', lw=2.5, label=r'$\eta_\mathrm{Carnot}=1-1/R_\beta$')
for name, cfg in PLATFORMS.items():
    ax2.scatter(cfg['R_beta'], 1-1/cfg['R_beta'],
                color=cfg['color'], marker=cfg['marker'], s=90, zorder=6,
                label=name.replace('\n',' '))
# Our benchmark
ax2.axvline(OUR_R_BETA, ls='--', color='darkgreen', lw=1.8, alpha=0.8)
ax2.axhline(1-1/OUR_R_BETA, ls='--', color='darkgreen', lw=1.2, alpha=0.5)
ax2.axhline(1-1/OUR_R_OMEGA, ls='-.', color='purple', lw=1.5, alpha=0.7,
            label=r'$\eta_\mathrm{max}$ at $R_\omega$=6 (binding)')
ax2.annotate(f'Our benchmark\n$R_\\beta$={OUR_R_BETA:.0f}, η_C={1-1/OUR_R_BETA:.3f}',
             xy=(OUR_R_BETA, 1-1/OUR_R_BETA),
             xytext=(OUR_R_BETA+1, 0.74), fontsize=8.5, color='darkgreen',
             arrowprops=dict(arrowstyle='->', color='darkgreen'))
ax2.set_xlabel(r'$R_\beta = \beta_c / \beta_h$ cap', fontsize=12)
ax2.set_ylabel(r'Carnot Bound $\eta_\mathrm{Carnot}$', fontsize=12)
ax2.set_xlim(1, R_beta_max_plot)
ax2.set_ylim(0, 1)
ax2.legend(fontsize=7.5, loc='lower right')
ax2.grid(alpha=0.35)
ax2.set_title(r'$\eta_\mathrm{Carnot}$ vs Temperature Ratio Cap', fontsize=11)

# ── Panel 3: 2D heatmap ────────────────────────────────────────────────────────
ax3 = axes[2]
im = ax3.contourf(Ro_g, Rb_g, eta_grid, levels=40, cmap='plasma', vmin=0, vmax=1)
# Diagonal line R_omega = R_beta (below: freq binds | above: temp binds)
rmin = 1.01; rmax = min(R_omega_max_plot, R_beta_max_plot)
ax3.plot([rmin, rmax], [rmin, rmax], 'w--', lw=1.5, alpha=0.8,
         label=r'$R_\omega = R_\beta$ (regime boundary)')
# Platform points
for name, cfg in PLATFORMS.items():
    ax3.scatter(cfg['R_omega'], cfg['R_beta'],
                color=cfg['color'], marker=cfg['marker'],
                s=120, edgecolors='white', linewidths=1.2, zorder=8,
                label=name.replace('\n',' '))
    eta_pt = 1 - 1/min(cfg['R_omega'], cfg['R_beta'])
    ax3.annotate(f"η={eta_pt:.3f}",
                 xy=(cfg['R_omega'], cfg['R_beta']),
                 xytext=(cfg['R_omega']+0.2, cfg['R_beta']+1.0),
                 fontsize=7.5, color='white', fontweight='bold')
# Our benchmark
ax3.scatter(OUR_R_OMEGA, OUR_R_BETA, marker='*', s=280, color='white',
            edgecolors='purple', linewidths=1.5, zorder=10,
            label=f'Our benchmark ({OUR_R_OMEGA:.0f},{OUR_R_BETA:.0f})')
ax3.annotate(f'η={1-1/OUR_R_OMEGA:.3f}\n(our bench)',
             xy=(OUR_R_OMEGA, OUR_R_BETA),
             xytext=(OUR_R_OMEGA+0.5, OUR_R_BETA-4), fontsize=8, color='white',
             arrowprops=dict(arrowstyle='->', color='white'))
ax3.set_xlabel(r'$R_\omega = \omega_h/\omega_c$ cap', fontsize=12)
ax3.set_ylabel(r'$R_\beta = \beta_c/\beta_h$ cap', fontsize=12)
ax3.set_xlim(Ro.min(), Ro.max())
ax3.set_ylim(Rb.min(), Rb.max())
ax3.set_title(r'$\eta_\mathrm{max}(R_\omega, R_\beta)$ — 2D Map', fontsize=11)
ax3.legend(fontsize=7, loc='upper left')
plt.colorbar(im, ax=ax3, label=r'$\eta_\mathrm{max}$')

plt.tight_layout()
out = f'{PDIR}/fig5_constraint_justification.png'
fig.savefig(out, dpi=200, bbox_inches='tight')
plt.close()
print(f"Saved: {out}")

# Print summary table
print("\n=== Platform Summary ===")
print(f"{'Platform':<30} {'R_ω':>5} {'R_β':>5} {'η_max':>8} {'η_Carnot':>10}")
print("-"*62)
for name, cfg in PLATFORMS.items():
    ro, rb = cfg['R_omega'], cfg['R_beta']
    em = 1-1/ro; ec = 1-1/rb
    print(f"{name.replace(chr(10),' '):<30} {ro:>5} {rb:>5} {em:>8.4f} {ec:>10.4f}")
print(f"\nOur benchmark: R_omega={OUR_R_OMEGA}, R_beta={OUR_R_BETA} → η_max={1-1/OUR_R_OMEGA:.4f}")
