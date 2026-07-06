import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

# Import functions from workspace model
import sys
sys.path.append("/home/reet/monika/modified codes")
from physics_model_modified import compute_metrics, stable_coth

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PLOTS_DIR = os.path.join(SCRIPT_DIR, 'plots')
os.makedirs(PLOTS_DIR, exist_ok=True)

# Styling
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'Times', 'DejaVu Serif'],
    'mathtext.fontset': 'stix',
    'font.size': 10,
    'axes.labelsize': 11,
    'axes.titlesize': 11,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'axes.grid': True,
    'grid.alpha': 0.2,
    'grid.linestyle': '--',
    'figure.dpi': 300,
})

alphas = [0.0, 0.05, 0.1, 0.15, 0.2]
alpha_colors = {
    0.0: "#111111",   # Black
    0.05: "#2980b9",  # Blue
    0.1: "#8e44ad",   # Purple
    0.15: "#d35400",  # Orange
    0.2: "#c0392b"    # Red
}

def plot_comparison():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5.5))
    
    # ── Panel 1: Paper's Parameters ──────────────────────────────────────────
    # Note: Typos in paper caption is resolved (fixed bc=3.643, sweep bh from 0.01 to 2.0)
    bc_p, wc_p, wh_p = 3.643, 0.594, 4.161
    bh_vals = np.linspace(0.01, 2.0, 200)
    
    for alpha in alphas:
        etas = []
        for bh in bh_vals:
            # Bypass constraint checks (just like the paper does) to get the mathematical curves
            eta, _, _, _, _, ok = compute_metrics(bc_p, bh, wc_p, wh_p, alpha, 1.0, 1.0)
            # Ensure bh < bc and bc*wc > bh*wh for engine operation
            if bh < bc_p and bc_p*wc_p > bh*wh_p:
                etas.append(eta)
            else:
                etas.append(np.nan)
        ax1.plot(bh_vals, etas, label=f"$\\alpha = {alpha}$", color=alpha_colors[alpha], lw=1.8)
        
    ax1.set_title("Paper's Parameters (Figure 3 Reproduction)\n"
                  r"$\beta_c = 3.643$, $\omega_c = 0.594$, $\omega_h = 4.161$")
    ax1.set_xlabel(r"Hot inverse temperature $\beta_h$")
    ax1.set_ylabel(r"Efficiency $\eta$")
    ax1.set_xlim(0.0, 2.0)
    ax1.set_ylim(0.0, 0.90)
    ax1.legend()

    # ── Panel 2: Our Optimized Parameters ─────────────────────────────────────
    # Under strict physical checks (bc/bh <= 25, perturbation ratio <= 0.10)
    bc_o, wc_o, wh_o = 9.122322, 1.578911, 11.052378
    # Sweep bh over the valid range 0.0 to 1.5
    bh_vals_o = np.linspace(0.01, 1.5, 200)
    
    for alpha in alphas:
        etas_o = []
        for bh in bh_vals_o:
            # Enforce all constraints: bc/bh <= 25 => bh >= bc/25 => bh >= 0.365
            if bh >= bc_o / 25.0 and bh < bc_o and bc_o*wc_o > bh*wh_o:
                # Perturbation check
                valid = True
                for w, b in [(wc_o, bc_o), (wh_o, bh)]:
                    coth_val = stable_coth(b*w/2.0)
                    harmonic = w/2.0 * abs(coth_val)
                    anharmonic = 3.0*alpha/(8.0*w**2) * coth_val**2
                    if (anharmonic/harmonic) > 0.1:
                        valid = False
                if valid:
                    eta, _, _, _, _, _ = compute_metrics(bc_o, bh, wc_o, wh_o, alpha, 1.0, 1.0)
                    etas_o.append(eta)
                else:
                    etas_o.append(np.nan)
            else:
                etas_o.append(np.nan)
        ax2.plot(bh_vals_o, etas_o, label=f"$\\alpha = {alpha}$", color=alpha_colors[alpha], lw=1.8)
        
    ax2.set_title("Our Optimized Parameters (Case 1)\n"
                  r"$\beta_c = 9.122$, $\omega_c = 1.579$, $\omega_h = 11.052$")
    ax2.set_xlabel(r"Hot inverse temperature $\beta_h$")
    ax2.set_ylabel(r"Efficiency $\eta$")
    ax2.set_xlim(0.0, 1.5)
    ax2.set_ylim(0.0, 0.90)
    ax2.legend()
    
    plt.tight_layout()
    out_path = os.path.join(PLOTS_DIR, 'reproduction_comparison.png')
    plt.savefig(out_path, dpi=300)
    plt.savefig(out_path.replace(".png", ".pdf"), dpi=300)
    plt.close()
    print(f"Saved comparison plot to: {out_path} and PDF version.")

if __name__ == '__main__':
    plot_comparison()
