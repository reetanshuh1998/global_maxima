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
    if wh/wc > R_OMEGA_MAX + 1e-4: return False
    if bc/bh > R_BETA_MAX + 1e-4: return False
    
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

# Constants
R_OMEGA_MAX = 7.0
R_BETA_MAX = 25.0

# Fixed optimal parameters at alpha=0
optimal_params_alpha0 = {
    1: {"bc": 9.122322, "bh": 0.633769, "wc": 1.578911, "wh": 11.052378},
    2: {"bc": 2.500000, "bh": 0.100000, "wc": 1.000000, "wh": 3.600154},
    3: {"bc": 2.500000, "bh": 0.100000, "wc": 1.000000, "wh": 3.056955},
    4: {"bc": 2.500000, "bh": 0.100000, "wc": 1.000000, "wh": 2.258097}
}

alphas = [0.0, 0.05, 0.1, 0.15, 0.2]
cases = [1, 2, 3, 4]

alpha_colors = {
    0.0: "#111111",   # Black
    0.05: "#2980b9",  # Blue
    0.1: "#8e44ad",   # Purple
    0.15: "#d35400",  # Orange
    0.2: "#c0392b"    # Red
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

def make_panel_plot(sweep_var, get_params_func, x_limits, y_limits, x_label, file_name, title_prefix):
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    
    for idx, case in enumerate(cases):
        r_idx = idx // 2
        c_idx = idx % 2
        ax = axes[r_idx, c_idx]
        
        params = optimal_params_alpha0[case]
        
        xlim = x_limits[case] if isinstance(x_limits, dict) else x_limits
        ylim = y_limits[case] if isinstance(y_limits, dict) else y_limits
        sweep_vals = np.linspace(xlim[0], xlim[1], 200)
        
        for alpha in alphas:
            etas = []
            for val in sweep_vals:
                # Get parameters dynamically with the swept variable replaced
                bc, bh, wc, wh = get_params_func(params, val)
                
                if check_point_validity(bc, bh, wc, wh, alpha, case):
                    l_ab, l_cd = get_lambdas(wc, wh, case)
                    eta, _, _, _, _, _ = compute_metrics(bc, bh, wc, wh, alpha, l_ab, l_cd)
                    etas.append(eta)
                else:
                    etas.append(np.nan)
            
            ax.plot(sweep_vals, etas, label=f"$\\alpha = {alpha}$", color=alpha_colors[alpha], lw=1.8)
            
        ax.set_title(f"Case {case}: " + (
            r"$\lambda_{ab}=\lambda_{cd}=1$" if case == 1 else
            r"$\lambda_{ab}=f(\omega),\,\lambda_{cd}=1$" if case == 2 else
            r"$\lambda_{ab}=1,\,\lambda_{cd}=f(\omega)$" if case == 3 else
            r"$\lambda_{ab}=\lambda_{cd}=f(\omega)$"
        ), fontsize=10, pad=8)
        
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        
        if c_idx == 0:
            ax.set_ylabel(r"Efficiency $\eta$")
        ax.set_xlabel(x_label)
        
        # Text box listing fixed values
        bc_fix, bh_fix, wc_fix, wh_fix = get_params_func(params, None)
        fixed_text = []
        if bc_fix is not None: fixed_text.append(f"$\\beta_c={bc_fix:.2f}$")
        if bh_fix is not None: fixed_text.append(f"$\\beta_h={bh_fix:.2f}$")
        if wc_fix is not None: fixed_text.append(f"$\\omega_c={wc_fix:.2f}$")
        if wh_fix is not None: fixed_text.append(f"$\\omega_h={wh_fix:.2f}$")
        
        ax.text(0.95, 0.95, "\n".join(fixed_text), transform=ax.transAxes, ha='right', va='top',
                fontsize=8, bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8, ec="gray", lw=0.5))
        
        # Warning annotations for invalid alpha curves
        if case in [2, 3]:
            ax.text(0.95, 0.35, r"$\alpha \geq 0.15$ invalid" + "\n" + r"(violates $\varepsilon_c \leq 0.10$)",
                    transform=ax.transAxes, ha='right', va='center', fontsize=8, color='#c0392b',
                    bbox=dict(boxstyle="round,pad=0.2", fc="#fadbd8", ec="#e74c3c", lw=0.5, alpha=0.9))
        elif case == 4:
            ax.text(0.95, 0.35, r"$\alpha \geq 0.10$ invalid" + "\n" + r"($W_{\mathrm{ext}} \leq 0$ or $\varepsilon_c > 0.10$)",
                    transform=ax.transAxes, ha='right', va='center', fontsize=8, color='#c0392b',
                    bbox=dict(boxstyle="round,pad=0.2", fc="#fadbd8", ec="#e74c3c", lw=0.5, alpha=0.9))
            
        # Inset zoom axes for Case 1 in wh sweep
        if sweep_var == "wh" and case == 1:
            from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset
            axins = inset_axes(ax, width="42%", height="38%", loc="lower right", borderpad=1.5)
            
            sweep_vals_in = np.linspace(2.0, 4.0, 100)
            for alpha in alphas:
                etas_in = []
                for val in sweep_vals_in:
                    bc, bh, wc, wh = get_params_func(params, val)
                    if check_point_validity(bc, bh, wc, wh, alpha, case):
                        l_ab, l_cd = get_lambdas(wc, wh, case)
                        eta, _, _, _, _, _ = compute_metrics(bc, bh, wc, wh, alpha, l_ab, l_cd)
                        etas_in.append(eta)
                    else:
                        etas_in.append(np.nan)
                axins.plot(sweep_vals_in, etas_in, color=alpha_colors[alpha], lw=1.2)
                
            axins.set_xlim(2.0, 4.0)
            axins.set_ylim(0.30, 0.50)
            axins.tick_params(labelsize=7)
            axins.grid(True, alpha=0.15, linestyle='--')
            mark_inset(ax, axins, loc1=2, loc2=4, fc="none", ec="0.5", lw=0.6, ls="--")
                
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=5, bbox_to_anchor=(0.5, -0.02), frameon=True, fontsize=10)
    
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.06, hspace=0.22, wspace=0.22)
    
    out_path = os.path.join(PLOTS_DIR, file_name)
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.savefig(out_path.replace(".png", ".pdf"), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {out_path}")

# 1. Sweep beta_h
def get_params_bh(p, val):
    if val is None:
        return p["bc"], None, p["wc"], p["wh"]
    return p["bc"], val, p["wc"], p["wh"]

bh_limits = {
    1: (0.0, 1.5),
    2: (0.0, 0.35),
    3: (0.0, 0.50),
    4: (0.0, 0.40)
}

y_limits_bh = {
    1: (0.80, 0.90),
    2: (0.0, 0.70),
    3: (0.0, 0.45),
    4: (0.0, 0.40)
}

make_panel_plot(
    sweep_var="bh",
    get_params_func=get_params_bh,
    x_limits=bh_limits,
    y_limits=y_limits_bh,
    x_label=r"Hot inverse temperature $\beta_h$",
    file_name="eta_vs_beta_h_grid_overlay.png",
    title_prefix="Efficiency vs beta_h"
)

# 2. Sweep beta_c
def get_params_bc(p, val):
    if val is None:
        return None, p["bh"], p["wc"], p["wh"]
    return val, p["bh"], p["wc"], p["wh"]

bc_limits = {
    1: (0.0, 15.0),
    2: (0.0, 4.0),
    3: (0.0, 4.0),
    4: (0.0, 4.0)
}

y_limits_bc = {
    1: (0.80, 0.90),
    2: (0.0, 0.70),
    3: (0.0, 0.45),
    4: (0.0, 0.40)
}

make_panel_plot(
    sweep_var="bc",
    get_params_func=get_params_bc,
    x_limits=bc_limits,
    y_limits=y_limits_bc,
    x_label=r"Cold inverse temperature $\beta_c$",
    file_name="eta_vs_beta_c_grid_overlay.png",
    title_prefix="Efficiency vs beta_c"
)

# 3. Sweep omega_h
def get_params_wh(p, val):
    if val is None:
        return p["bc"], p["bh"], p["wc"], None
    return p["bc"], p["bh"], p["wc"], val

wh_limits = {
    1: (1.5, 12.0),
    2: (1.0, 7.5),
    3: (1.0, 7.5),
    4: (1.0, 7.5)
}

y_limits_wh = {
    1: (0.0, 0.90),
    2: (0.0, 0.80),
    3: (0.0, 0.70),
    4: (0.0, 0.60)
}

make_panel_plot(
    sweep_var="wh",
    get_params_func=get_params_wh,
    x_limits=wh_limits,
    y_limits=y_limits_wh,
    x_label=r"Hot frequency $\omega_h$",
    file_name="eta_vs_wh_grid_overlay.png",
    title_prefix="Efficiency vs wh"
)
