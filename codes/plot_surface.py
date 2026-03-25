import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def calculate_efficiency(bc, bh, wc, wh, lam=0.0):
    if wc >= wh or bh >= bc or lam < 0:
        return np.nan
    if bc * wc <= bh * wh:
        return np.nan

    X = np.cosh(bh * wh / 2) / np.sinh(bh * wh / 2)
    Y = np.cosh(bc * wc / 2) / np.sinh(bc * wc / 2)
    
    Q_h = (wh / 2) * (X - Y) + (3 * lam / (8 * wh**2)) * (X**2 - Y**2)
    Q_c = (wc / 2) * (Y - X) + (3 * lam / (8 * wc**2)) * (Y**2 - X**2)
    
    W_ext = Q_h + Q_c
    if Q_h <= 0 or W_ext <= 0:
        return np.nan
    
    return W_ext / Q_h

def plot_surface():
    # Fix the temperatures to extreme values to provide a wide valid physical domain
    beta_c = 100.0  # T_c is very low
    beta_h = 0.01   # T_h is very high
    lam = 0.0       # Harmonic case gives maximum efficiency

    # Generate a range for omega_c and omega_h
    omega_c_vals = np.linspace(0.1, 20.0, 100)
    omega_h_vals = np.linspace(20.1, 100.0, 100)
    
    WC, WH = np.meshgrid(omega_c_vals, omega_h_vals)
    ETA = np.zeros_like(WC)

    # Calculate eta for every combination
    for i in range(WC.shape[0]):
        for j in range(WC.shape[1]):
            eta = calculate_efficiency(beta_c, beta_h, WC[i, j], WH[i, j], lam)
            ETA[i, j] = eta if not np.isnan(eta) else 0

    # Plotting the 3D surface
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(WC, WH, ETA, cmap='magma', edgecolor='none')

    ax.set_xlabel(r'$\omega_c$ (Cold Frequency)')
    ax.set_ylabel(r'$\omega_h$ (Hot Frequency)')
    ax.set_zlabel(r'Efficiency $\eta$')
    ax.set_title(r'3D Surface: Countless parameter combinations lead to $\eta \to 1$')

    # Force viewing angle to clearly see the plateau
    ax.view_init(elev=20, azim=-135)
    
    fig.colorbar(surf, shrink=0.5, aspect=5, label=r'Efficiency $\eta$')
    
    plt.savefig('eta_vs_omegas_surface.png', dpi=300, bbox_inches='tight')
    print("3D surface plot saved to eta_vs_omegas_surface.png")

if __name__ == "__main__":
    plot_surface()
