import os
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from physics_model_modified import compute_metrics, is_valid_physics, stable_coth

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PLOTS_DIR = os.path.join(SCRIPT_DIR, 'plots')
os.makedirs(PLOTS_DIR, exist_ok=True)

def get_lambdas(wc, wh, case):
    l_fn = (wc**2 + wh**2) / (2.0 * wc * wh)
    if case == 1:
        return 1.0, 1.0
    elif case == 2:
        return l_fn, 1.0
    elif case == 3:
        return 1.0, l_fn
    else:  # case 4: both asymmetric
        return l_fn, l_fn

def check_point_validity(bc, bh, wc, wh, alpha, case):
    if wc >= wh: return False
    if bh >= bc: return False
    if bc*wc <= bh*wh: return False
    if wh/wc > R_OMEGA_MAX: return False
    if bc/bh > R_BETA_MAX: return False
    
    # Anharmonic perturbation check
    for w, b in [(wc, bc), (wh, bh)]:
        coth_val  = stable_coth(b*w/2.0)
        harmonic  = w/2.0 * abs(coth_val)
        if harmonic <= 0: return False
        anharmonic = 3.0*alpha/(8.0*w**2) * coth_val**2
        if (anharmonic/harmonic) > 0.1:
            return False
            
    # Metrics validity
    l_ab, l_cd = get_lambdas(wc, wh, case)
    eta, qh, qc, w_ext, _, ok = compute_metrics(bc, bh, wc, wh, alpha, l_ab, l_cd)
    if not ok or qh <= 0 or w_ext <= 0:
        return False
        
    return True

# Constants from physics_model_modified
R_OMEGA_MAX = 7.0
R_BETA_MAX = 25.0

# 5 alpha values overlayed
alphas = [0.0, 0.05, 0.1, 0.15, 0.2]
cases = [1, 2, 3, 4]

# Optimal parameters at alpha=0, obtained from the global search verification:
optimal_params_alpha0 = {
    1: {"bc": 9.122322, "bh": 0.633769, "wh": 11.052378},
    2: {"bc": 2.500000, "bh": 0.100000, "wh": 3.600154},
    3: {"bc": 2.500000, "bh": 0.100000, "wh": 3.056955},
    4: {"bc": 2.500000, "bh": 0.100000, "wh": 2.258097}
}

# ─── Plot Styling ────────────────────────────────────────────────────────
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
    'savefig.dpi': 300,
})

# Create 2x2 subplot grid
fig, axes = plt.subplots(2, 2, figsize=(11, 8))

# Color palette for the alpha curves (harmonic case is black/dark, others are gradient)
alpha_colors = {
    0.0: "#111111",   # Black
    0.05: "#2980b9",  # Blue
    0.1: "#8e44ad",   # Purple
    0.15: "#d35400",  # Orange
    0.2: "#c0392b"    # Red
}

# Custom wc ranges to cover the full physical domain for each case
w_c_ranges = {
    1: np.linspace(1.5, 11.2, 200),
    2: np.linspace(0.05, 3.7, 200),
    3: np.linspace(0.05, 3.2, 200),
    4: np.linspace(0.05, 2.4, 200)
}

x_limits = {
    1: (1.5, 11.2),
    2: (0.0, 3.7),
    3: (0.0, 3.2),
    4: (0.0, 2.4)
}

for idx, case in enumerate(cases):
    r_idx = idx // 2
    c_idx = idx % 2
    ax = axes[r_idx, c_idx]
    
    # Get the fixed optimal parameters at alpha=0
    params = optimal_params_alpha0[case]
    bc, bh, wh = params["bc"], params["bh"], params["wh"]
    
    w_c_vals = w_c_ranges[case]
    
    # Create inset zoom axes for Case 1
    if case == 1:
        axins = ax.inset_axes([0.15, 0.15, 0.45, 0.45])
    
    for alpha in alphas:
        w_ext_valid = []
        for wc in w_c_vals:
            if check_point_validity(bc, bh, wc, wh, alpha, case):
                l_ab, l_cd = get_lambdas(wc, wh, case)
                _, _, _, w_val, _, _ = compute_metrics(bc, bh, wc, wh, alpha, l_ab, l_cd)
                w_ext_valid.append(w_val)
            else:
                w_ext_valid.append(np.nan)
        
        # Plot only valid region (as in the paper plot, work drops to 0 or is cut off)
        ax.plot(w_c_vals, w_ext_valid, label=f"$\\alpha = {alpha}$",
                color=alpha_colors[alpha], lw=1.8)
                
        # Also plot in the inset zoom box for Case 1
        if case == 1:
            axins.plot(w_c_vals, w_ext_valid, color=alpha_colors[alpha], lw=1.2)
        
    ax.set_title(f"Case {case}: " + (
        r"$\lambda_{ab}=\lambda_{cd}=1$" if case == 1 else
        r"$\lambda_{ab}=f(\omega),\,\lambda_{cd}=1$" if case == 2 else
        r"$\lambda_{ab}=1,\,\lambda_{cd}=f(\omega)$" if case == 3 else
        r"$\lambda_{ab}=\lambda_{cd}=f(\omega)$"
    ), fontsize=10, pad=8)
    
    # Set custom x-axis limit and ensure y-axis starts at 0
    ax.set_xlim(x_limits[case])
    ax.set_ylim(bottom=0.0)
    
    # Format y-axis using scientific notation
    ax.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
    ax.yaxis.major.formatter._useMathText = True
    
    # Configure Case 1 inset zoom parameters
    if case == 1:
        axins.set_xlim(1.6, 2.0)
        axins.set_ylim(0.0079, 0.0088)
        axins.tick_params(labelsize=7)
        axins.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
        axins.yaxis.major.formatter._useMathText = True
        ax.indicate_inset_zoom(axins, edgecolor="gray", alpha=0.5)
    
    # Labels
    if c_idx == 0:
        ax.set_ylabel(r"Work Output $W_{\mathrm{ext}}$")
    
    # Show xlabel on all subplots since x-axes scales are customized
    ax.set_xlabel(r"Cold frequency $\omega_c$")
        
    # Add textbox with fixed parameters in upper right or upper left
    eta_max_val = {1: 0.857143, 2: 0.629889, 3: 0.397207, 4: 0.343051}[case]
    param_text = f"$\\beta_c={bc:.2f}, \\beta_h={bh:.2f}$\n$\\omega_h={wh:.2f}$\n$\\eta_{{\\mathrm{{max}}}}(\\alpha=0)={eta_max_val:.4f}$"
    ax.text(0.95, 0.95, param_text, transform=ax.transAxes, ha='right', va='top',
            fontsize=8, bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8, ec="gray", lw=0.5))

# Add a shared legend for the entire figure at the bottom
handles, labels = axes[0, 0].get_legend_handles_labels()
fig.legend(handles, labels, loc='lower center', ncol=5, bbox_to_anchor=(0.5, -0.02), frameon=True, fontsize=10)

plt.tight_layout()
plt.subplots_adjust(bottom=0.06, hspace=0.22, wspace=0.22)

out_path = os.path.join(PLOTS_DIR, "work_vs_wc_grid_overlay.png")
plt.savefig(out_path, dpi=300, bbox_inches='tight')
plt.savefig(out_path.replace(".png", ".pdf"), dpi=300, bbox_inches='tight')
plt.close()
print(f"\nSaved grid overlay plot to: {out_path} and PDF version.")
