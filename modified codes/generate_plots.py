import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from physics_model_modified import compute_metrics, is_valid_physics, lamba_ab

# Setup paths relative to the script directory
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PLOTS_DIR = os.path.join(SCRIPT_DIR, 'plots')
os.makedirs(PLOTS_DIR, exist_ok=True)

# ─── Operating Points ────────────────────────────────────────────────────────
# Hardcoded from refrigerator_optima.json for accuracy and self-containment
OPERATING_POINTS = {
    "Max COP Pure": {
        "params": {"bc": 3.61208, "bh": 2.21266, "wc": 10.0, "wh": 16.48855},
        "description": r"Max COP Pure ($\beta_c \approx 3.61, \beta_h \approx 2.21, \omega_c = 10.0, \omega_h \approx 16.49$)"
    },
    "Max Cooling Power (Qc)": {
        "params": {"bc": 14.60724, "bh": 7.71499, "wc": 0.83304, "wh": 11.07474},
        "description": r"Max Cooling Power $Q_c$ ($\beta_c \approx 14.61, \beta_h \approx 7.71, \omega_c \approx 0.83, \omega_h \approx 11.07$)"
    },
    "Max Chi (Trade-off)": {
        "params": {"bc": 3.77197, "bh": 2.10140, "wc": 3.76767, "wh": 14.79343},
        "description": r"Max Chi $\chi = Q_c \times \mathrm{COP}$ ($\beta_c \approx 3.77, \beta_h \approx 2.10, \omega_c \approx 3.77, \omega_h \approx 14.79$)"
    }
}

# ─── Coupling configurations to compare ──────────────────────────────────────
# lamba_ab default is None (computed as function), lamba_cd default is None (set to 1)
CONFIGS = [
    {
        "label": r"Default $\lambda_{ab}(\omega_c, \omega_h)$, $\lambda_{cd}=1$",
        "l_ab": None,
        "l_cd": None,
        "color": "#8e44ad",  # Amethyst Purple
        "ls": "-",
        "lw": 2.2
    },
    {
        "label": r"Symmetric $\lambda_{ab}=1.0$, $\lambda_{cd}=1.0$",
        "l_ab": 1.0,
        "l_cd": 1.0,
        "color": "#2980b9",  # Belize Hole Blue
        "ls": "--",
        "lw": 1.8
    },
    {
        "label": r"Asymmetric $\lambda_{ab}=1.5$, $\lambda_{cd}=1.0$",
        "l_ab": 1.5,
        "l_cd": 1.0,
        "color": "#e67e22",  # Carrot Orange
        "ls": "-.",
        "lw": 1.8
    },
    {
        "label": r"Asymmetric $\lambda_{ab}=1.0$, $\lambda_{cd}=1.5$",
        "l_ab": 1.0,
        "l_cd": 1.5,
        "color": "#1abc9c",  # Turquoise
        "ls": ":",
        "lw": 2.0
    },
    {
        "label": r"Shifted Symmetric $\lambda_{ab}=1.5$, $\lambda_{cd}=1.5$",
        "l_ab": 1.5,
        "l_cd": 1.5,
        "color": "#c0392b",  # Alizarin Red
        "ls": "-",
        "lw": 1.2
    }
]

# ─── Plot Styling ───────────────────────────────────────────────────────────
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'Times', 'DejaVu Serif'],
    'mathtext.fontset': 'stix',
    'font.size': 10,
    'axes.labelsize': 11,
    'axes.titlesize': 12,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'axes.grid': True,
    'grid.alpha': 0.25,
    'grid.linestyle': '--',
    'figure.dpi': 300,
    'savefig.dpi': 300,
})

# Grid of alpha (lambda) values
alphas = np.linspace(0.0, 0.5, 300)

def generate_plot_for_op(op_name, op_details, ax=None, save_individual=True):
    bc = op_details["params"]["bc"]
    bh = op_details["params"]["bh"]
    wc = op_details["params"]["wc"]
    wh = op_details["params"]["wh"]
    desc = op_details["description"]

    # Calculate Carnot and Harmonic COPs (constant for a given operating point)
    # The compute_metrics function returns these at any evaluation
    _, _, _, _, cop_carnot, cop_harmonic, _ = compute_metrics(bc, bh, wc, wh, 0.0)

    standalone = False
    if ax is None:
        standalone = True
        fig, ax = plt.subplots(figsize=(6.5, 4.8))

    # Determine perturbation validity limit
    valid_alphas = [a for a in alphas if is_valid_physics(bc, bh, wc, wh, a)]
    if valid_alphas:
        max_valid_alpha = max(valid_alphas)
        # Highlight validity range
        ax.axvspan(0.0, max_valid_alpha, color='#2ecc71', alpha=0.1, label=r'Perturbation Valid ($\leq 10\%$)')
        ax.axvline(max_valid_alpha, color='#27ae60', linestyle=':', alpha=0.5)

    # Plot Carnot and Harmonic Reference Lines
    ax.axhline(cop_carnot, color='#e74c3c', linestyle=':', alpha=0.8, lw=1.2, label=rf'Carnot limit ($\mathrm{{COP}} = {cop_carnot:.4f}$)')
    ax.axhline(cop_harmonic, color='#7f8c8d', linestyle='--', alpha=0.7, lw=1.0, label=rf'Harmonic limit ($\mathrm{{COP}} = {cop_harmonic:.4f}$)')

    # Compute and plot COP vs Alpha for each configuration
    for cfg in CONFIGS:
        cops = []
        for alpha in alphas:
            cop, _, _, _, _, _, is_ref = compute_metrics(bc, bh, wc, wh, alpha, custom_lamba_ab=cfg["l_ab"], custom_lamba_cd=cfg["l_cd"])
            cops.append(cop if is_ref else np.nan)
        
        cops = np.array(cops)
        ax.plot(alphas, cops, label=cfg["label"], color=cfg["color"], linestyle=cfg["ls"], lw=cfg["lw"])

    ax.set_xlabel(r'Anharmonicity Parameter $\alpha$ (lambda)')
    ax.set_ylabel('Coefficient of Performance (COP)')
    ax.set_title(desc, pad=10)
    ax.set_xlim(0.0, 0.5)
    
    # Sensible y limit (up to a bit above Carnot, or max plotted COP)
    y_max_plotted = np.nanmax([np.nanmax(
        np.array([compute_metrics(bc, bh, wc, wh, a, custom_lamba_ab=cfg["l_ab"], custom_lamba_cd=cfg["l_cd"])[0] 
                  for a in alphas if compute_metrics(bc, bh, wc, wh, a, custom_lamba_ab=cfg["l_ab"], custom_lamba_cd=cfg["l_cd"])[6]])
    ) for cfg in CONFIGS])
    
    # Cap y limit gracefully
    ylim_top = min(cop_carnot * 1.05, max(y_max_plotted * 1.1, cop_harmonic * 1.2)) if cop_carnot > 0 else max(y_max_plotted * 1.1, cop_harmonic * 1.2)
    ax.set_ylim(0.0, ylim_top)
    
    if standalone:
        ax.legend(loc='best', frameon=True, fancybox=True, shadow=False, framealpha=0.9, fontsize=9)
        plt.tight_layout()
        filename = f"cop_vs_alpha_{op_name.lower().replace(' ', '_').replace('(', '').replace(')', '')}.png"
        filepath = os.path.join(PLOTS_DIR, filename)
        plt.savefig(filepath, dpi=300)
        plt.close()
        print(f"Saved stand-alone plot for '{op_name}' to: {filepath}")

# 1. Generate stand-alone plots
for op_name, op_details in OPERATING_POINTS.items():
    generate_plot_for_op(op_name, op_details)

# 2. Generate publication-ready combined 3-panel layout
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
for i, (op_name, op_details) in enumerate(OPERATING_POINTS.items()):
    generate_plot_for_op(op_name, op_details, ax=axes[i], save_individual=False)

# Place legend in a clean way (e.g. above or below subplots, or inside first one)
# We can create a single shared legend for the entire figure to keep it clean
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc='lower center', ncol=4, bbox_to_anchor=(0.5, -0.08), frameon=True, fontsize=10)

plt.tight_layout()
# Adjust layout to make room for the bottom legend
plt.subplots_adjust(bottom=0.15)

panel_filepath = os.path.join(PLOTS_DIR, 'cop_vs_alpha_comparison_panel.png')
plt.savefig(panel_filepath, dpi=300, bbox_inches='tight')
plt.close()
print(f"Saved 3-panel comparison panel to: {panel_filepath}")
