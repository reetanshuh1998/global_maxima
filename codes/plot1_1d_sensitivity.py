"""
Plot 1: 5x 1D plots — eta as a function of each free parameter (other 4 fixed).
Shows individual sensitivity of eta to each parameter.
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

# Base (fixed) parameter values for all 1D sweeps
BASE = dict(bc=10.0, bh=0.5, wc=2.0, wh=10.0, lam=0.1)
N = 300

fig, axes = plt.subplots(1, 5, figsize=(20, 4))
fig.suptitle(r'Sensitivity: $\eta$ vs each free parameter (others fixed at base values)', fontsize=13)

# 1. eta vs beta_c
ax = axes[0]
vals = np.linspace(BASE['bh'] + 0.1, 30, N)
etas = [eta(v, BASE['bh'], BASE['wc'], BASE['wh'], BASE['lam']) for v in vals]
ax.plot(vals, etas, color='royalblue', lw=2)
ax.axvline(BASE['bc'], ls='--', color='gray', label=f"base={BASE['bc']}")
ax.set_xlabel(r'$\beta_c$'); ax.set_ylabel(r'$\eta$')
ax.set_title(r'$\eta$ vs $\beta_c$'); ax.legend(); ax.grid(True, alpha=0.4)
ax.set_ylim(0, 1); ax.ticklabel_format(useOffset=False)

# 2. eta vs beta_h
ax = axes[1]
vals = np.linspace(0.01, BASE['bc'] - 0.1, N)
etas = [eta(BASE['bc'], v, BASE['wc'], BASE['wh'], BASE['lam']) for v in vals]
ax.plot(vals, etas, color='tomato', lw=2)
ax.axvline(BASE['bh'], ls='--', color='gray', label=f"base={BASE['bh']}")
ax.set_xlabel(r'$\beta_h$'); ax.set_ylabel(r'$\eta$')
ax.set_title(r'$\eta$ vs $\beta_h$'); ax.legend(); ax.grid(True, alpha=0.4)
ax.set_ylim(0, 1); ax.ticklabel_format(useOffset=False)

# 3. eta vs omega_c
ax = axes[2]
vals = np.linspace(0.1, BASE['wh'] - 0.1, N)
etas = [eta(BASE['bc'], BASE['bh'], v, BASE['wh'], BASE['lam']) for v in vals]
ax.plot(vals, etas, color='seagreen', lw=2)
ax.axvline(BASE['wc'], ls='--', color='gray', label=f"base={BASE['wc']}")
ax.set_xlabel(r'$\omega_c$'); ax.set_ylabel(r'$\eta$')
ax.set_title(r'$\eta$ vs $\omega_c$'); ax.legend(); ax.grid(True, alpha=0.4)
ax.set_ylim(0, 1); ax.ticklabel_format(useOffset=False)

# 4. eta vs omega_h
ax = axes[3]
vals = np.linspace(BASE['wc'] + 0.1, 60, N)
etas = [eta(BASE['bc'], BASE['bh'], BASE['wc'], v, BASE['lam']) for v in vals]
ax.plot(vals, etas, color='darkorchid', lw=2)
ax.axvline(BASE['wh'], ls='--', color='gray', label=f"base={BASE['wh']}")
ax.set_xlabel(r'$\omega_h$'); ax.set_ylabel(r'$\eta$')
ax.set_title(r'$\eta$ vs $\omega_h$'); ax.legend(); ax.grid(True, alpha=0.4)
ax.set_ylim(0, 1); ax.ticklabel_format(useOffset=False)

# 5. eta vs lambda
ax = axes[4]
vals = np.linspace(0.0, 1.0, N)
etas = [eta(BASE['bc'], BASE['bh'], BASE['wc'], BASE['wh'], v) for v in vals]
ax.plot(vals, etas, color='darkorange', lw=2)
ax.axvline(BASE['lam'], ls='--', color='gray', label=f"base={BASE['lam']}")
ax.set_xlabel(r'$\lambda$'); ax.set_ylabel(r'$\eta$')
ax.set_title(r'$\eta$ vs $\lambda$'); ax.legend(); ax.grid(True, alpha=0.4)
ax.set_ylim(0, 1); ax.ticklabel_format(useOffset=False)

plt.tight_layout()
plt.savefig('plot1_1d_sensitivity.png', dpi=200, bbox_inches='tight')
print("Saved: plot1_1d_sensitivity.png")
