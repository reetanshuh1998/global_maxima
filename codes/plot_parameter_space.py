import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

def calculate_efficiency(bc, bh, wc, wh, lam):
    if wc >= wh or bh >= bc or lam < 0:
        return np.nan
    if bc * wc <= bh * wh:
        return np.nan
    arg_h = bh * wh / 2
    arg_c = bc * wc / 2
    # Use stable computation for large arguments
    if arg_h > 300 or arg_c > 300:
        return np.nan
    X = 1.0 / np.tanh(arg_h)  # coth(beta_h * omega_h / 2)
    Y = 1.0 / np.tanh(arg_c)  # coth(beta_c * omega_c / 2)
    Q_h = (wh / 2) * (X - Y) + (3 * lam / (8 * wh**2)) * (X**2 - Y**2)
    Q_c = (wc / 2) * (Y - X) + (3 * lam / (8 * wc**2)) * (Y**2 - X**2)
    W_ext = Q_h + Q_c
    if Q_h <= 0 or W_ext <= 0:
        return np.nan
    eta = W_ext / Q_h
    return eta if 0 < eta <= 1 else np.nan

N = 80  # resolution per panel

fig, axes = plt.subplots(2, 3, figsize=(16, 10))
fig.suptitle(r'Efficiency $\eta$ across all free parameter pairs — each bright region shows $\eta \to 1$', fontsize=13)
cmap = 'inferno'

# --- Panel 1: eta(omega_c, omega_h) --- fixed: bc=100, bh=0.01, lam=0
ax = axes[0, 0]
wc_v = np.linspace(0.1, 10, N)
wh_v = np.linspace(0.2, 100, N)
WC, WH = np.meshgrid(wc_v, wh_v)
ETA = np.vectorize(calculate_efficiency)(100.0, 0.01, WC, WH, 0.0)
im = ax.pcolormesh(WC, WH, ETA, cmap=cmap, vmin=0, vmax=1)
ax.set_xlabel(r'$\omega_c$'); ax.set_ylabel(r'$\omega_h$')
ax.set_title(r'$\eta(\omega_c, \omega_h)$' + '\n' + r'[$\beta_c=100, \beta_h=0.01, \lambda=0$]')
fig.colorbar(im, ax=ax)

# --- Panel 2: eta(beta_c, beta_h) --- fixed: wc=2, wh=20, lam=0
ax = axes[0, 1]
bc_v = np.linspace(0.5, 20, N)
bh_v = np.linspace(0.01, 5, N)
BC, BH = np.meshgrid(bc_v, bh_v)
ETA = np.vectorize(calculate_efficiency)(BC, BH, 2.0, 20.0, 0.0)
im = ax.pcolormesh(BC, BH, ETA, cmap=cmap, vmin=0, vmax=1)
ax.set_xlabel(r'$\beta_c$'); ax.set_ylabel(r'$\beta_h$')
ax.set_title(r'$\eta(\beta_c, \beta_h)$' + '\n' + r'[$\omega_c=2, \omega_h=20, \lambda=0$]')
fig.colorbar(im, ax=ax)

# --- Panel 3: eta(omega_c, lambda) --- fixed: bc=50, bh=0.05, wh=20
ax = axes[0, 2]
wc_v = np.linspace(0.5, 10, N)
lam_v = np.linspace(0.0, 1.0, N)
WC, LAM = np.meshgrid(wc_v, lam_v)
ETA = np.vectorize(calculate_efficiency)(50.0, 0.05, WC, 20.0, LAM)
im = ax.pcolormesh(WC, LAM, ETA, cmap=cmap, vmin=0, vmax=1)
ax.set_xlabel(r'$\omega_c$'); ax.set_ylabel(r'$\lambda$')
ax.set_title(r'$\eta(\omega_c, \lambda)$' + '\n' + r'[$\beta_c=50, \beta_h=0.05, \omega_h=20$]')
fig.colorbar(im, ax=ax)

# --- Panel 4: eta(omega_h, lambda) --- fixed: bc=50, bh=0.05, wc=2
ax = axes[1, 0]
wh_v = np.linspace(2.1, 100, N)
lam_v = np.linspace(0.0, 1.0, N)
WH, LAM = np.meshgrid(wh_v, lam_v)
ETA = np.vectorize(calculate_efficiency)(50.0, 0.05, 2.0, WH, LAM)
im = ax.pcolormesh(WH, LAM, ETA, cmap=cmap, vmin=0, vmax=1)
ax.set_xlabel(r'$\omega_h$'); ax.set_ylabel(r'$\lambda$')
ax.set_title(r'$\eta(\omega_h, \lambda)$' + '\n' + r'[$\beta_c=50, \beta_h=0.05, \omega_c=2$]')
fig.colorbar(im, ax=ax)

# --- Panel 5: eta(beta_c, lambda) --- fixed: bh=0.05, wc=2, wh=20
ax = axes[1, 1]
bc_v = np.linspace(1.0, 20, N)
lam_v = np.linspace(0.0, 1.0, N)
BC, LAM = np.meshgrid(bc_v, lam_v)
ETA = np.vectorize(calculate_efficiency)(BC, 0.05, 2.0, 20.0, LAM)
im = ax.pcolormesh(BC, LAM, ETA, cmap=cmap, vmin=0, vmax=1)
ax.set_xlabel(r'$\beta_c$'); ax.set_ylabel(r'$\lambda$')
ax.set_title(r'$\eta(\beta_c, \lambda)$' + '\n' + r'[$\beta_h=0.05, \omega_c=2, \omega_h=20$]')
fig.colorbar(im, ax=ax)

# --- Panel 6: eta(beta_h, lambda) --- fixed: bc=10, wc=2, wh=20
ax = axes[1, 2]
bh_v = np.linspace(0.01, 3, N)
lam_v = np.linspace(0.0, 1.0, N)
BH, LAM = np.meshgrid(bh_v, lam_v)
ETA = np.vectorize(calculate_efficiency)(10.0, BH, 2.0, 20.0, LAM)
im = ax.pcolormesh(BH, LAM, ETA, cmap=cmap, vmin=0, vmax=1)
ax.set_xlabel(r'$\beta_h$'); ax.set_ylabel(r'$\lambda$')
ax.set_title(r'$\eta(\beta_h, \lambda)$' + '\n' + r'[$\beta_c=10, \omega_c=2, \omega_h=20$]')
fig.colorbar(im, ax=ax)

plt.tight_layout()
plt.savefig('eta_parameter_space.png', dpi=200, bbox_inches='tight')
print("Multi-panel heatmap saved to eta_parameter_space.png")
