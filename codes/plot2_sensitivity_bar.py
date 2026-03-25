"""
Plot 2: Sensitivity (gradient) bar chart — |d(eta)/dxi| for each parameter.
Shows which parameter controls the efficiency most strongly.
"""
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

def eta(bc, bh, wc, wh, lam):
    if wc >= wh or bh >= bc or lam < 0:
        return np.nan
    if bc * wc <= bh * wh:
        return np.nan
    arg_h = bh * wh / 2
    arg_c = bc * wc / 2
    if arg_h > 300 or arg_c > 300:
        return np.nan
    X = 1.0 / np.tanh(arg_h)
    Y = 1.0 / np.tanh(arg_c)
    Q_h = (wh / 2) * (X - Y) + (3 * lam / (8 * wh**2)) * (X**2 - Y**2)
    Q_c = (wc / 2) * (Y - X) + (3 * lam / (8 * wc**2)) * (Y**2 - X**2)
    W_ext = Q_h + Q_c
    if Q_h <= 0 or W_ext <= 0:
        return np.nan
    v = W_ext / Q_h
    return v if 0 < v <= 1 else np.nan

BASE = dict(bc=10.0, bh=0.5, wc=2.0, wh=10.0, lam=0.1)
params = ['bc', 'bh', 'wc', 'wh', 'lam']
labels = [r'$\beta_c$', r'$\beta_h$', r'$\omega_c$', r'$\omega_h$', r'$\lambda$']
# Percent step for finite difference
eps_frac = 0.01

gradients = []
for p in params:
    p_val = BASE[p]
    eps = max(abs(p_val) * eps_frac, 1e-5)
    up = dict(BASE)
    dn = dict(BASE)
    up[p] = p_val + eps
    dn[p] = p_val - eps
    eta_up = eta(**up)
    eta_dn = eta(**dn)
    if eta_up is None or eta_dn is None or np.isnan(eta_up) or np.isnan(eta_dn):
        gradients.append(0.0)
    else:
        gradients.append(abs((eta_up - eta_dn) / (2 * eps)))

# Normalize gradients for comparison
max_g = max(gradients) if max(gradients) > 0 else 1
normalized = [g / max_g for g in gradients]

colors = ['royalblue', 'tomato', 'seagreen', 'darkorchid', 'darkorange']
fig, ax = plt.subplots(figsize=(8, 5))
bars = ax.bar(labels, normalized, color=colors, edgecolor='black', linewidth=0.7)
ax.set_ylabel('Normalized sensitivity $|d\\eta/dx_i|$')
ax.set_title('Parameter Sensitivity: which parameter controls $\\eta$ most?\n(at base values: ' +
             f"$\\beta_c$={BASE['bc']}, $\\beta_h$={BASE['bh']}, $\\omega_c$={BASE['wc']}, $\\omega_h$={BASE['wh']}, $\\lambda$={BASE['lam']})")
ax.set_ylim(0, 1.15)
for bar, val in zip(bars, gradients):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, f"{val:.4f}", ha='center', fontsize=10)
ax.grid(axis='y', alpha=0.4)
plt.tight_layout()
plt.savefig('plot2_sensitivity_bar.png', dpi=200, bbox_inches='tight')
print("Saved: plot2_sensitivity_bar.png")
