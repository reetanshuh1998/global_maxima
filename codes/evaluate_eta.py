import numpy as np

def calculate_efficiency(beta_c, beta_h, omega_c, omega_h, lam):
    """
    Calculates the efficiency of a quantum Otto cycle with a quartic oscillator.
    Returns the efficiency (eta) based on the exact formulas derived from the paper.
    """
    # Check basic physical constraints
    if omega_c >= omega_h:
        raise ValueError("Constraint violated: omega_c must be less than omega_h")
    if beta_h >= beta_c:
        raise ValueError("Constraint violated: beta_h must be less than beta_c (T_h > T_c)")
    if lam < 0:
        raise ValueError("Constraint violated: lambda must be non-negative")
    if beta_c * omega_c <= beta_h * omega_h:
        raise ValueError("Condition for positive work output (engine mode) is not met")

    # Hyperbolic cotangent calculations for average energies
    X = np.cosh(beta_h * omega_h / 2) / np.sinh(beta_h * omega_h / 2)
    Y = np.cosh(beta_c * omega_c / 2) / np.sinh(beta_c * omega_c / 2)
    
    # Heat extracted and absorbed directly from Eqs. (8) and (9)
    Q_h = (omega_h / 2) * (X - Y) + (3 * lam / (8 * omega_h**2)) * (X**2 - Y**2)
    Q_c = (omega_c / 2) * (Y - X) + (3 * lam / (8 * omega_c**2)) * (Y**2 - X**2)
    
    W_ext = Q_h + Q_c
    
    if Q_h <= 0 or W_ext <= 0:
        return 0.0 # Operates as a refrigerator or dissipator, not an engine
        
    return W_ext / Q_h

if __name__ == "__main__":
    # Optimal parameters found for maximum global efficiency (approaching eta = 1)
    # The efficiency is mathematically maximized when lambda = 0 (harmonic limit),
    # omega_c -> 0, and omega_h -> infinity.
    # Below is a set of valid parameters near these optimal bounds.
    optimal_params = {
        "beta_c":  3700.123351,
        "beta_h": 0.0020996,
        "omega_c": 0.0100000,
        "omega_h": 1000.000000,
        "lam": 0.0
    }
    
    print("Evaluating efficiency for optimal parameters:")
    for k, v in optimal_params.items():
        if k == 'lam':
            print(f"lambda  = {v:.6f}")
        else:
            print(f"{k:<7} = {v:.6f}")
    
    eta = calculate_efficiency(**optimal_params)
    print("---------------------------------------------")
    print(f"Absolute Maximum Efficiency (eta) = {eta:.6f}")
    
    # For comparison, evaluating the harmonic limit explicitly
    eta_harmonic = 1 - (optimal_params['omega_c'] / optimal_params['omega_h'])
    print(f"Theoretical Harmonic limit (eta)  = {eta_harmonic:.6f}")
