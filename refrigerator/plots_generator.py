import numpy as np
import matplotlib.pyplot as plt
from physics_model import compute_metrics, is_valid_physics, BOUNDS

def generate_tradeoff_data(n_points=50):
    # Fixed reference point (found during scratch search)
    # bc: 4.1647, bh: 3.4112, wc: 3.7520, wh: 4.5808, lam: 0.5
    bc, bh, wc, wh_base, lam = 4.1647, 3.4112, 3.7520, 4.5808, 0.5
    
    # Sweep omega_h to move along the Pareto curve
    wh_vals = np.linspace(wc + 0.1, 15.0, n_points)
    
    results = []
    for wh in wh_vals:
        m = compute_metrics(bc, bh, wc, wh, lam)
        if m[-1]: # if valid refrigerator
            results.append(m)
            
    return np.array(results)

def generate_lambda_effect_data(n_points=20):
    # bc: 4.1647, bh: 3.4112, wc: 3.7520, wh: 4.5808
    bc, bh, wc, wh = 4.1647, 3.4112, 3.7520, 5.0
    lams = np.linspace(0, 0.5, n_points)
    
    results = []
    for l in lams:
        if is_valid_physics(bc, bh, wc, wh, l):
            m = compute_metrics(bc, bh, wc, wh, l)
            if m[-1]:
                results.append(m + (l,))
                
    return np.array(results)

def plot_refrigerator_analysis():
    tradeoff = generate_tradeoff_data()
    lam_effect = generate_lambda_effect_data()
    
    plt.style.use('seaborn-v0_8-muted')
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Plot 1: COP vs Cooling Power (Trade-off)
    if len(tradeoff) > 0:
        cops = tradeoff[:, 0]
        qcs = tradeoff[:, 1]
        ax1.plot(qcs, cops, 'o-', color='#e74c3c', label='Refrigerator Pareto Front')
        ax1.set_xlabel('Cooling Power (Qc)')
        ax1.set_ylabel('COP')
        ax1.set_title('COP vs Cooling Power Trade-off')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
    
    # Plot 2: COP vs Lambda (Anharmonic Enhancement)
    if len(lam_effect) > 0:
        cops_lam = lam_effect[:, 0]
        lams = lam_effect[:, -1]
        ax2.plot(lams, cops_lam, 's-', color='#3498db', label='Fixed Operating Point')
        ax2.set_xlabel('Anharmonicity (lambda)')
        ax2.set_ylabel('COP')
        ax2.set_title('Anharmonic COP Enhancement')
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        
    plt.tight_layout()
    os.makedirs('plots', exist_ok=True)
    plt.savefig('plots/refrigerator_analysis.png', dpi=300)
    print("Refrigerator analysis plot saved to plots/refrigerator_analysis.png")

if __name__ == "__main__":
    import os
    plot_refrigerator_analysis()
