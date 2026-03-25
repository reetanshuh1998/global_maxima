"""
Plot 4: Carnot comparison scatter — eta_anharmonic vs eta_Carnot.
Physics check: eta must always be <= eta_Carnot = 1 - beta_h/beta_c.
"""
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")
np.random.seed(0)

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

N_samples = 10000
eta_vals, carnot_vals = [], []

for _ in range(N_samples):
    wc = np.random.uniform(0.5, 10)
    wh = np.random.uniform(wc + 0.5, 30)
    bh = np.random.uniform(0.05, 3)
    bc = np.random.uniform(bh + 0.1, 15)
    lam = np.random.uniform(0, 0.5)
    val = eta(bc, bh, wc, wh, lam)
    if val is not None and not np.isnan(val):
        carnot = 1 - bh / bc
        eta_vals.append(val)
        carnot_vals.append(carnot)

eta_vals = np.array(eta_vals)
carnot_vals = np.array(carnot_vals)

fig, ax = plt.subplots(figsize=(7, 7))
sc = ax.scatter(carnot_vals, eta_vals, alpha=0.2, s=5, c=eta_vals, cmap='plasma', label='Samples')
ax.plot([0, 1], [0, 1], 'r--', lw=2, label=r'$\eta = \eta_{Carnot}$ (physical limit)')
ax.set_xlabel(r'Carnot Efficiency $\eta_{Carnot} = 1 - \beta_h/\beta_c$', fontsize=12)
ax.set_ylabel(r'Anharmonic Otto Efficiency $\eta$', fontsize=12)
ax.set_title('Otto Efficiency vs Carnot Limit\n(all points must lie below the red line)', fontsize=12)
ax.legend(fontsize=11)
fig.colorbar(sc, label=r'$\eta$')
ax.grid(alpha=0.4)
plt.tight_layout()
plt.savefig('plot4_carnot_comparison.png', dpi=200, bbox_inches='tight')
print(f"Saved: plot4_carnot_comparison.png  |  Valid samples: {len(eta_vals)}")
