"""
verify_eta_max.py
==================
INDEPENDENT verification of the optimal parameters found by find_optimal_parameters.py.

This script:
  1. Loads the saved optimal_parameters.json
  2. Re-implements the eta formula from SCRATCH (no imports from other modules)
  3. Substitutes the optimal (bc, bh, wc, wh, lam) directly into the formula
  4. Prints a verification table and generates a verification figure

If the numbers match find_optimal_parameters.py → the result is confirmed. ✓
"""
import json, os, sys, signal
signal.signal(signal.SIGINT, signal.SIG_IGN)

import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# ══════════════════════════════════════════════════════════════════════════════
# STANDALONE PHYSICS — copied directly from Eqs 3-10 of the reference paper
# No imports from physics_model.py or any other project file.
# ══════════════════════════════════════════════════════════════════════════════

def coth_stable(x):
    """Numerical stable coth — Eq. surrounding Eqs 4-7 of paper."""
    ax = abs(x)
    if ax < 1e-10:         return np.sign(x) * 1e10
    elif ax < 1e-3:        return 1.0/x + x/3.0 - (x**3)/45.0   # Laurent series
    elif ax > 20.0:        return float(np.sign(x)) * (1 + 2*np.exp(-2*ax))  # asymptotic
    else:                  return np.cosh(x) / np.sinh(x)        # direct

def heat_and_efficiency(bc, bh, wc, wh, lam):
    """
    Implements Eqs 8-10 of the reference paper.

    X = coth(bh*wh/2)  — hot bath cotangent argument
    Y = coth(bc*wc/2)  — cold bath cotangent argument

    Eq 8: Q_h = (wh/2)(X-Y) + (3*lam)/(8*wh^2) * (X^2-Y^2)
    Eq 9: Q_c = (wc/2)(Y-X) + (3*lam)/(8*wc^2) * (Y^2-X^2)
    Eq10: W_ext = Q_h + Q_c
          eta   = W_ext / Q_h  (if engine mode: Q_h>0, W_ext>0)
    """
    X = coth_stable(bh * wh / 2.0)     # hot bath
    Y = coth_stable(bc * wc / 2.0)     # cold bath

    # Eq 8: heat absorbed from hot reservoir
    Q_h = (wh / 2.0) * (X - Y) + (3.0 * lam / (8.0 * wh**2)) * (X**2 - Y**2)

    # Eq 9: heat released to cold reservoir
    Q_c = (wc / 2.0) * (Y - X) + (3.0 * lam / (8.0 * wc**2)) * (Y**2 - X**2)

    # Eq 10: work extracted
    W_ext = Q_h + Q_c

    # Efficiency
    if Q_h > 1e-12 and W_ext > 1e-12:
        eta = W_ext / Q_h
    else:
        eta = 0.0

    return dict(Q_h=Q_h, Q_c=Q_c, W_ext=W_ext, eta=eta, X=X, Y=Y)

def carnot_efficiency(bc, bh):
    """eta_Carnot = 1 - T_h/T_c = 1 - beta_c/beta_h (since beta = 1/kT)."""
    return 1.0 - bh/bc

def otto_harmonic_bound(wc, wh):
    """eta_Otto,harmonic = 1 - omega_c/omega_h (lam=0 limit)."""
    return 1.0 - wc/wh

# ══════════════════════════════════════════════════════════════════════════════
# Load and verify
# ══════════════════════════════════════════════════════════════════════════════
RDIR  = os.path.join(os.path.dirname(__file__), '..', 'results')
PDIR  = os.path.join(os.path.dirname(__file__), '..', 'plots')
os.makedirs(PDIR, exist_ok=True)

json_path = os.path.join(RDIR, 'optimal_parameters.json')
with open(json_path) as f:
    data = json.load(f)

print("=" * 90)
print("  INDEPENDENT VERIFICATION OF OPTIMAL PARAMETERS")
print("  Formula: Eqs 8-10 of reference paper (standalone implementation)")
print("=" * 90)

hdr = (f"\n{'Platform':<28} {'η (optimizer)':>14} {'η (verify)':>12} "
       f"{'η_Carnot':>10} {'η_Otto,harm':>13} {'Match?':>8}")
print(hdr)
print("-" * 90)

verification_data = []

for name, cfg in data.items():
    params = cfg.get('params')
    if params is None:
        print(f"{name.replace(chr(10),' '):<28} {'---':>14} {'NO PARAMS':>12}")
        continue

    bc, bh, wc, wh, lam = params
    result  = heat_and_efficiency(bc, bh, wc, wh, lam)
    eta_ver = result['eta']
    eta_opt = cfg['eta_max']
    eta_car = carnot_efficiency(bc, bh)
    eta_har = otto_harmonic_bound(wc, wh)
    match   = abs(eta_ver - eta_opt) < 1e-4

    short = name.replace('\n',' ')
    print(f"{short:<28} {eta_opt:>14.6f} {eta_ver:>12.6f} "
          f"{eta_car:>10.6f} {eta_har:>13.6f} {'✓' if match else 'MISMATCH':>8}")

    verification_data.append({
        'name': short,
        'params': params,
        'eta_verify': eta_ver,
        'eta_optimizer': eta_opt,
        'eta_carnot': eta_car,
        'eta_harmonic_bound': eta_har,
        'Q_h': result['Q_h'], 'Q_c': result['Q_c'], 'W_ext': result['W_ext'],
        'R_omega_used': wh/wc, 'R_beta_used': bc/bh,
        'R_omega_cap': cfg['R_omega'], 'R_beta_cap': cfg['R_beta'],
        'match': match
    })

print("\nDetailed thermodynamic breakdown:")
print("-" * 90)
for d in verification_data:
    bc,bh,wc,wh,lam = d['params']
    print(f"\n  {d['name']}")
    print(f"    Parameters : βc={bc:.4f}  βh={bh:.4f}  ωc={wc:.4f}  "
          f"ωh={wh:.4f}  λ={lam:.4f}")
    print(f"    Ratios used: ωh/ωc = {d['R_omega_used']:.4f} / cap {d['R_omega_cap']}  |  "
          f"βc/βh = {d['R_beta_used']:.4f} / cap {d['R_beta_cap']}")
    print(f"    Q_h = {d['Q_h']:.6f}  Q_c = {d['Q_c']:.6f}  W_ext = {d['W_ext']:.6f}")
    print(f"    η (verified)  = {d['eta_verify']:.6f}")
    print(f"    η (Carnot)    = {d['eta_carnot']:.6f}   → η < η_Carnot: {d['eta_verify'] < d['eta_carnot']}")
    print(f"    η_harmonic_bd = {d['eta_harmonic_bound']:.6f}")

# ── Verification figure ────────────────────────────────────────────────────────
fig = plt.figure(figsize=(16, 6))
gs  = gridspec.GridSpec(1, 3, figure=fig)
fig.suptitle('Independent Verification of Optimal Parameters\n'
             '(standalone Eqs 8–10 implementation — no shared code)', fontsize=13)

colors = ['#95a5a6','#e74c3c','#2ecc71','#9b59b6']
names  = [d['name'] for d in verification_data]
x      = np.arange(len(names))

# Panel 1: optimizer vs verification
ax1 = fig.add_subplot(gs[0])
ax1.bar(x - 0.18, [d['eta_optimizer'] for d in verification_data],
        0.32, label='Optimizer result', color=colors[:len(x)], edgecolor='k',
        linewidth=0.8, alpha=0.9)
ax1.bar(x + 0.18, [d['eta_verify'] for d in verification_data],
        0.32, label='Standalone verify', color=colors[:len(x)], edgecolor='k',
        linewidth=0.8, alpha=0.4, hatch='//')
ax1.plot(x, [d['eta_carnot'] for d in verification_data],
         'r^--', ms=9, lw=1.5, label=r'$\eta_\mathrm{Carnot}$')
for i, d in enumerate(verification_data):
    ax1.text(i, d['eta_optimizer'] + 0.01, '✓' if d['match'] else 'X',
             ha='center', fontsize=14, color='green' if d['match'] else 'red')
ax1.set_xticks(x); ax1.set_xticklabels(names, fontsize=7.5)
ax1.set_ylabel(r'$\eta$', fontsize=12); ax1.set_ylim(0, 1)
ax1.legend(fontsize=8); ax1.grid(axis='y', alpha=0.35)
ax1.set_title('Optimizer vs. Verified η', fontsize=11)

# Panel 2: heat budget (Q_h, Q_c, W_ext)
ax2 = fig.add_subplot(gs[1])
q_h  = [d['Q_h']   for d in verification_data]
q_c  = [abs(d['Q_c']) for d in verification_data]
w    = [d['W_ext'] for d in verification_data]
ax2.bar(x - 0.25, q_h, 0.22, color='orange', edgecolor='k', linewidth=0.6, label=r'$Q_h$ (absorbed)')
ax2.bar(x,        q_c, 0.22, color='skyblue', edgecolor='k', linewidth=0.6, label=r'$|Q_c|$ (released)')
ax2.bar(x + 0.25, w,   0.22, color='limegreen', edgecolor='k', linewidth=0.6, label=r'$W_\mathrm{ext}$')
ax2.set_xticks(x); ax2.set_xticklabels(names, fontsize=7.5)
ax2.set_ylabel('Energy (arb. units)', fontsize=11)
ax2.legend(fontsize=8); ax2.grid(axis='y', alpha=0.35)
ax2.set_title(r'Heat Budget: $Q_h$, $|Q_c|$, $W_\mathrm{ext}$', fontsize=11)

# Panel 3: how close to cap boundaries
ax3 = fig.add_subplot(gs[2])
frac_w = [d['R_omega_used']/d['R_omega_cap'] for d in verification_data]
frac_b = [d['R_beta_used'] /d['R_beta_cap']  for d in verification_data]
ax3.bar(x - 0.2, frac_w, 0.35, color=colors[:len(x)], edgecolor='k',
        linewidth=0.8, alpha=0.9, label=r'$(\omega_h/\omega_c) / R_\omega$')
ax3.bar(x + 0.2, frac_b, 0.35, color=colors[:len(x)], edgecolor='k',
        linewidth=0.8, alpha=0.45, hatch='//', label=r'$(\beta_c/\beta_h) / R_\beta$')
ax3.axhline(1.0, ls='--', color='red', lw=1.5, label='Cap boundary (1.0)')
ax3.set_xticks(x); ax3.set_xticklabels(names, fontsize=7.5)
ax3.set_ylabel('Fraction of cap used', fontsize=11)
ax3.set_ylim(0, 1.15); ax3.legend(fontsize=8); ax3.grid(axis='y', alpha=0.35)
ax3.set_title('Binding Constraint Fractions', fontsize=11)

plt.tight_layout()
out = f'{PDIR}/fig7_verification.png'
fig.savefig(out, dpi=200, bbox_inches='tight')
plt.close()
print(f"\nSaved: {out}")
