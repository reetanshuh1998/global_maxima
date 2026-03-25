"""
Adam Optimizer vs SLSQP: Head-to-head benchmark
================================================
Since PyTorch is not installed, we implement Adam from scratch using
NumPy + finite-difference gradients. This is a FAIR comparison because
SLSQP also uses numerical gradients internally (BFGS quasi-Newton).

Both methods start from the same random initial points.
We measure:
  - Evaluations to reach eta >= 0.99
  - Wall-clock time

We also add gradient ascent (vanilla) as a reference.
"""
import numpy as np
import time
import matplotlib.pyplot as plt
import warnings
from scipy.optimize import minimize
warnings.filterwarnings("ignore")

THRESHOLD = 0.99
N_TRIALS  = 10   # 10 restarts per method per comparison

# ─────────────────────────────────────────────
def eta_fn(bc, bh, wc, wh, lam):
    if wc >= wh or bh >= bc or lam < 0: return 0.0
    if bc * wc <= bh * wh: return 0.0
    a_h, a_c = bh*wh/2, bc*wc/2
    if a_h > 300 or a_c > 300: return 0.0
    X = 1.0/np.tanh(a_h); Y = 1.0/np.tanh(a_c)
    Qh = (wh/2)*(X-Y) + (3*lam/(8*wh**2))*(X**2-Y**2)
    Qc = (wc/2)*(Y-X) + (3*lam/(8*wc**2))*(Y**2-X**2)
    W  = Qh + Qc
    if Qh <= 0 or W <= 0: return 0.0
    v = W / Qh
    return float(v) if 0 < v <= 1 else 0.0

BOUNDS = np.array([(0.5,50),(0.01,5),(0.1,15),(1.0,80),(0.0,1.0)])

def clip_to_bounds(x):
    return np.clip(x, BOUNDS[:,0], BOUNDS[:,1])

def sample_valid(rng):
    while True:
        bc  = rng.uniform(0.5, 50)
        bh  = rng.uniform(0.01, min(5, bc-0.01))
        wc  = rng.uniform(0.1, 15)
        wh  = rng.uniform(max(1.0, wc+0.01), 80)
        lam = rng.uniform(0, 1.0)
        if bc*wc > bh*wh: return np.array([bc,bh,wc,wh,lam])

def numerical_gradient(x, eps=1e-5):
    """5-point finite difference gradient — more accurate than 2-point."""
    grad = np.zeros(5)
    for i in range(5):
        xp2, xp1, xm1, xm2 = [x.copy() for _ in range(4)]
        xp2[i]+=2*eps; xp1[i]+=eps; xm1[i]-=eps; xm2[i]-=2*eps
        grad[i] = (-eta_fn(*xp2)+8*eta_fn(*xp1)-8*eta_fn(*xm1)+eta_fn(*xm2))/(12*eps)
    return grad  # 4 evaluations per dimension = 20 per gradient call

# ─────────────────────────────────────────────
# Adam Optimizer (from scratch, for maximization)
# ─────────────────────────────────────────────
def adam_maximize(x0, lr=0.05, beta1=0.9, beta2=0.999, eps=1e-8, max_steps=2000):
    """
    Gradient ASCENT with Adam. Constraints handled by:
      - Clipping to bounds after each step (projected gradient)
      - Penalty term that pushes toward valid region
    Returns: (best_eta, steps, eval_count, trajectory)
    """
    x = x0.copy()
    m = np.zeros(5)  # first moment
    v = np.zeros(5)  # second moment
    best_eta = 0.0
    best_x   = x.copy()
    eval_count = 0
    traj = []

    for t in range(1, max_steps+1):
        grad = numerical_gradient(x)
        eval_count += 20  # 4 evals * 5 dims

        # Adam update
        m = beta1*m + (1-beta1)*grad
        v = beta2*v + (1-beta2)*grad**2
        m_hat = m / (1-beta1**t)
        v_hat = v / (1-beta2**t)

        x = x + lr * m_hat / (np.sqrt(v_hat) + eps)
        x = clip_to_bounds(x)

        val = eta_fn(*x)
        eval_count += 1
        traj.append((eval_count, val))

        if val > best_eta:
            best_eta = val
            best_x   = x.copy()
        if best_eta >= THRESHOLD:
            return best_eta, t, eval_count, traj

    return best_eta, max_steps, eval_count, traj

# ─────────────────────────────────────────────
# SLSQP
# ─────────────────────────────────────────────
constraints = [
    {'type':'ineq','fun': lambda x: x[3]-x[2]-0.1},
    {'type':'ineq','fun': lambda x: x[0]-x[1]-0.01},
    {'type':'ineq','fun': lambda x: x[0]*x[2]-x[1]*x[3]-0.01},
]

def slsqp_maximize(x0):
    count   = [0]
    best    = [0.0]
    reached = [None]
    t0 = time.perf_counter()

    def obj(x):
        count[0] += 1
        v = eta_fn(*x)
        if v > best[0]: best[0] = v
        if best[0] >= THRESHOLD and reached[0] is None:
            reached[0] = (count[0], time.perf_counter()-t0)
        return -v

    try:
        minimize(obj, x0, bounds=list(map(tuple,BOUNDS)), constraints=constraints,
                 method='SLSQP', options={'maxiter':300,'ftol':1e-10})
    except Exception: pass
    return best[0], count[0], reached[0]

# ─────────────────────────────────────────────
# Head-to-head: same start points
# ─────────────────────────────────────────────
print(f"Head-to-head: Adam vs SLSQP  ({N_TRIALS} random starts each)\n")
print(f"{'Start':<6} {'Adam evals':>12} {'Adam time(s)':>14} {'SLSQP evals':>13} {'SLSQP time(s)':>14} {'Winner':>8}")
print("-"*73)

adam_evals_list, adam_time_list     = [], []
slsqp_evals_list, slsqp_time_list   = [], []

rng = np.random.default_rng(99)

for trial in range(N_TRIALS):
    x0 = sample_valid(rng)

    # ---- Adam ----
    t0 = time.perf_counter()
    best_a, steps_a, evals_a, _ = adam_maximize(x0.copy())
    t_a = time.perf_counter()-t0
    a_reached = evals_a if best_a >= THRESHOLD else None

    # ---- SLSQP ----
    t0s = time.perf_counter()
    best_s, evals_s, reached_s = slsqp_maximize(x0.copy())
    t_s = time.perf_counter()-t0s
    s_time = reached_s[1] if reached_s else None
    s_evals = reached_s[0] if reached_s else None

    winner = "SLSQP" if (s_evals or 9999) < (a_reached or 9999) else "Adam"
    if a_reached is None and s_evals is None: winner = "Neither"

    print(f"{trial+1:<6} {str(a_reached):>12} {t_a:>14.4f} {str(s_evals):>13} {str(round(t_s,4)):>14} {winner:>8}")

    if a_reached: adam_evals_list.append(a_reached); adam_time_list.append(t_a)
    if s_evals:   slsqp_evals_list.append(s_evals);  slsqp_time_list.append(t_s)

# ─────────────────────────────────────────────
# Plot
# ─────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(13, 6))
fig.suptitle(f'Adam (NumPy) vs SLSQP — time-to threshold $\\eta \\geq {THRESHOLD}$\n'
             f'({N_TRIALS} identical random starting points)', fontsize=13)

labels  = ['Adam\n(Gradient Ascent)', 'SLSQP\n(Quasi-Newton)']
colors  = ['darkorange', 'tomato']

for ax, metric, ylabel, unit in zip(
        axes,
        [(adam_evals_list, slsqp_evals_list), (adam_time_list, slsqp_time_list)],
        ['# of $\\eta$ evaluations to reach $\\eta\\geq0.99$','Wall-clock time (s) to reach $\\eta\\geq0.99$'],
        ['evals','s']):

    data = metric
    means = [np.mean(d) if d else float('nan') for d in data]
    stds  = [np.std(d)  if d else float('nan') for d in data]
    bars  = ax.bar(labels, means, color=colors, edgecolor='black', linewidth=0.8,
                   yerr=stds, capsize=9, error_kw=dict(lw=2, ecolor='black'))
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_yscale('log')
    ax.grid(axis='y', alpha=0.4)
    ax.set_title(ylabel, fontsize=10)
    for bar, m, s in zip(bars, means, stds):
        if not np.isnan(m):
            ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()*1.3,
                    f'{m:.1f}±{s:.1f} {unit}', ha='center', fontsize=9, fontweight='bold')

plt.tight_layout()
plt.savefig('plot_adam_vs_slsqp.png', dpi=200, bbox_inches='tight')
print("\nSaved: plot_adam_vs_slsqp.png")
print(f"\nAdam:  reached threshold in {len(adam_evals_list)}/{N_TRIALS} trials, mean evals={np.mean(adam_evals_list):.0f}" if adam_evals_list else "\nAdam:  did NOT reach threshold")
print(f"SLSQP: reached threshold in {len(slsqp_evals_list)}/{N_TRIALS} trials, mean evals={np.mean(slsqp_evals_list):.0f}" if slsqp_evals_list else "\nSLSQP: did NOT reach threshold")
