import numpy as np
import matplotlib.pyplot as plt

def calculate_efficiency(beta_c, beta_h, omega_c, omega_h, lam):
    """
    Calculates the efficiency of a quantum Otto cycle with a quartic oscillator.
    """
    if omega_c >= omega_h or beta_h >= beta_c or lam < 0:
        return np.nan
    if beta_c * omega_c <= beta_h * omega_h:
        return np.nan

    X = np.cosh(beta_h * omega_h / 2) / np.sinh(beta_h * omega_h / 2)
    Y = np.cosh(beta_c * omega_c / 2) / np.sinh(beta_c * omega_c / 2)
    
    Q_h = (omega_h / 2) * (X - Y) + (3 * lam / (8 * omega_h**2)) * (X**2 - Y**2)
    Q_c = (omega_c / 2) * (Y - X) + (3 * lam / (8 * omega_c**2)) * (Y**2 - X**2)
    
    W_ext = Q_h + Q_c
    
    if Q_h <= 0 or W_ext <= 0:
        return np.nan
        
    return W_ext / Q_h

def plot_eta_vs_lambda():
    # To plot lambda up to 1 while keeping physics valid (lambda << 2/3 * omega_c^3),
    # we need omega_c >= 2.0. We will use omega_c = 2.0 and omega_h = 20.0 giving
    # a theoretical max efficiency of 1 - (2/20) = 0.90
    beta_c = 10.0
    beta_h = 0.1
    omega_c = 2.0
    omega_h = 20.0

    lambdas = np.linspace(0.0, 1.0, 500)
    etas = []

    for lam in lambdas:
        eta = calculate_efficiency(beta_c, beta_h, omega_c, omega_h, lam)
        etas.append(eta)

    plt.figure(figsize=(8, 6))
    plt.plot(lambdas, etas, 'b-', linewidth=2, label=r'$\omega_c=2.0, \omega_h=20.0$')
    plt.axhline(0.9, color='r', linestyle='--', label=r'Harmonic Limit $\eta=0.9$')
    
    plt.xlabel(r'Anharmonic parameter $\lambda$', fontsize=14)
    plt.ylabel(r'Efficiency $\eta$', fontsize=14)
    plt.title(r'Efficiency $\eta$ vs $\lambda$ for strongly bounded parameters', fontsize=14)
    plt.legend(loc='best')
    plt.grid(True)
    
    # Save the plot
    plt.savefig('eta_vs_lambda_plot.png', dpi=300)
    print("Plot saved to eta_vs_lambda_plot.png")

if __name__ == "__main__":
    plot_eta_vs_lambda()
