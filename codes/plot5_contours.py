"""
Plot 5: 2D filled contour plots — eta for each pair of parameters.
Contour lines reveal the "plateau" shape of the infinite global maximum region.
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

N = 80
cmap = 'RdYlGn'
BASE = dict(bc=10.0, bh=0.5, wc=2.0, wh=10.0, lam=0.1)

panels = [
    ('wc', 'wh', np.linspace(0.5, 9, N), np.linspace(1.0, 50, N), r'$\omega_c$', r'$\omega_h$',
     r'$\eta(\omega_c, \omega_h)$' + f"\n[bc={BASE['bc']},bh={BASE['bh']},λ={BASE['lam']}]",
     lambda X, Y: np.vectorize(eta)(BASE['bc'], BASE['bh'], X, Y, BASE['lam'])),
    ('bc', 'bh', np.linspace(1, 20, N), np.linspace(0.05, 5, N), r'$\beta_c$', r'$\beta_h$',
     r'$\eta(\beta_c, \beta_h)$' + f"\n[wc={BASE['wc']},wh={BASE['wh']},λ={BASE['lam']}]",
     lambda X, Y: np.vectorize(eta)(X, Y, BASE['wc'], BASE['wh'], BASE['lam'])),
    ('wc', 'lam', np.linspace(0.5, 9, N), np.linspace(0, 1, N), r'$\omega_c$', r'$\lambda$',
     r'$\eta(\omega_c, \lambda)$' + f"\n[bc={BASE['bc']},bh={BASE['bh']},wh={BASE['wh']}]",
     lambda X, Y: np.vectorize(eta)(BASE['bc'], BASE['bh'], X, BASE['wh'], Y)),
    ('wh', 'lam', np.linspace(2.5, 50, N), np.linspace(0, 1, N), r'$\omega_h$', r'$\lambda$',
     r'$\eta(\omega_h, \lambda)$' + f"\n[bc={BASE['bc']},bh={BASE['bh']},wc={BASE['wc']}]",
     lambda X, Y: np.vectorize(eta)(BASE['bc'], BASE['bh'], BASE['wc'], X, Y)),
    ('bc', 'lam', np.linspace(1, 20, N), np.linspace(0, 1, N), r'$\beta_c$', r'$\lambda$',
     r'$\eta(\beta_c, \lambda)$' + f"\n[bh={BASE['bh']},wc={BASE['wc']},wh={BASE['wh']}]",
     lambda X, Y: np.vectorize(eta)(X, BASE['bh'], BASE['wc'], BASE['wh'], Y)),
    ('bh', 'lam', np.linspace(0.05, 3, N), np.linspace(0, 1, N), r'$\beta_h$', r'$\lambda$',
     r'$\eta(\beta_h, \lambda)$' + f"\n[bc={BASE['bc']},wc={BASE['wc']},wh={BASE['wh']}]",
     lambda X, Y: np.vectorize(eta)(BASE['bc'], X, BASE['wc'], BASE['wh'], Y)),
]

fig, axes = plt.subplots(2, 3, figsize=(16, 10))
fig.suptitle(r'Contour plots: $\eta$ for each pair of free parameters'
             '\n(Flat top = plateau of infinite global maxima)', fontsize=13)

levels = np.linspace(0, 1, 21)

for ax, (p1, p2, xv, yv, xl, yl, title, fn) in zip(axes.flat, panels):
    X, Y = np.meshgrid(xv, yv)
    Z = fn(X, Y)
    Z = np.where(np.isnan(Z), 0, Z)
    cf = ax.contourf(X, Y, Z, levels=levels, cmap=cmap)
    cs = ax.contour(X, Y, Z, levels=[0.5, 0.7, 0.8, 0.9, 0.95], colors='k', linewidths=0.7, alpha=0.6)
    ax.clabel(cs, fmt='%.2f', fontsize=8)
    fig.colorbar(cf, ax=ax, label=r'$\eta$')
    ax.set_xlabel(xl); ax.set_ylabel(yl)
    ax.set_title(title)

plt.tight_layout()
plt.savefig('plot5_contours.png', dpi=200, bbox_inches='tight')
print("Saved: plot5_contours.png")
