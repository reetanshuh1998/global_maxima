"""
cross_verify.py
===============
Fully independent verification of the efficiency calculations for
Case 1 (lambda_ab = lambda_cd = 1) and Case 2 (lambda_ab = (wc^2+wh^2)/(2*wc*wh), lambda_cd = 1).
No imports from physics_model_modified.py - everything derived from scratch.
"""
import numpy as np
import json
import os

# ─── Constants ────────────────────────────────────────────────────────────────
R_OMEGA_MAX = 7.0
R_BETA_MAX  = 25.0
LAM_MAX     = 0.2

def coth(x):
    ax = abs(x)
    if ax < 1e-10: return np.sign(x) * 1e10
    if ax < 1e-3:  return 1.0/x + x/3.0 - x**3/45.0
    if ax > 20.0:  return float(np.sign(x)) * (1.0 + 2.0*np.exp(-2.0*ax))
    return 1.0 / np.tanh(x)

# ─── Case 1: lambda_ab = 1, lambda_cd = 1 ────────────────────────────────────
def compute_eta_case1(bc, bh, wc, wh, alpha):
    X = coth(bh * wh / 2.0)
    Y = coth(bc * wc / 2.0)
    l_ab = 1.0
    l_cd = 1.0
    Qh = (wh/2.0)*(X - l_ab*Y) + (3.0*alpha/(8.0*wh**2))*(X**2 - l_ab*Y**2)
    Qc = (wc/2.0)*(Y - l_cd*X) + (3.0*alpha/(8.0*wc**2))*(Y**2 - l_cd*X**2)
    W  = Qh + Qc
    if Qh <= 0 or W <= 0:
        return None
    return W / Qh

# ─── Case 2: lambda_ab = (wc^2+wh^2)/(2*wc*wh), lambda_cd = 1 ───────────────
def compute_eta_case2(bc, bh, wc, wh, alpha):
    X = coth(bh * wh / 2.0)
    Y = coth(bc * wc / 2.0)
    l_ab = (wc**2 + wh**2) / (2.0 * wc * wh)   # always >= 1 by AM-GM
    l_cd = 1.0
    Qh = (wh/2.0)*(X - l_ab*Y) + (3.0*alpha/(8.0*wh**2))*(X**2 - l_ab*Y**2)
    Qc = (wc/2.0)*(Y - l_cd*X) + (3.0*alpha/(8.0*wc**2))*(Y**2 - l_cd*X**2)
    W  = Qh + Qc
    if Qh <= 0 or W <= 0:
        return None
    return W / Qh

# ─── Physical validity check ──────────────────────────────────────────────────
def is_valid(bc, bh, wc, wh, alpha):
    if wc >= wh: return False
    if bh >= bc: return False
    if alpha < 0 or alpha > LAM_MAX: return False
    if bc*wc <= bh*wh: return False
    if wh/wc > R_OMEGA_MAX: return False
    if bc/bh > R_BETA_MAX:  return False
    # perturbation validity: 3*alpha*coth(b*w/2) / (4*w^3) <= 0.10
    for w, b in [(wc, bc), (wh, bh)]:
        eps = (3.0 * alpha * coth(b*w/2.0)) / (4.0 * w**3)
        if eps > 0.10:
            return False
    return True

# ─── Spot-check optimal parameters from Case 2 json ─────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
json_path = os.path.join(SCRIPT_DIR, "case2_optimal_parameters.json")
with open(json_path, "r") as f:
    case2_data = json.load(f)

print("=" * 80)
print("CROSS-VERIFICATION: CASE 2 (lambda_ab = (wc^2+wh^2)/(2*wc*wh), lambda_cd = 1)")
print("=" * 80)
print(f"{'Alpha':>8}  {'wc':>9}  {'wh':>9}  {'wh/wc':>8}  {'l_ab':>8}  "
      f"{'eta_stored':>12}  {'eta_verify':>12}  {'Match':>8}  {'Valid':>8}")
print("-" * 90)

all_case2_ok = True
for row in case2_data:
    a  = row["alpha"]
    bc = row["beta_c"]
    bh = row["beta_h"]
    wc = row["omega_c"]
    wh = row["omega_h"]
    eta_stored  = row["eta"]
    l_ab = (wc**2 + wh**2) / (2.0*wc*wh)
    eta_verify  = compute_eta_case2(bc, bh, wc, wh, a)
    valid       = is_valid(bc, bh, wc, wh, a)
    match       = eta_verify is not None and abs(eta_verify - eta_stored) < 1e-5
    if not match or not valid:
        all_case2_ok = False
    print(f"{a:8.3f}  {wc:9.5f}  {wh:9.5f}  {wh/wc:8.4f}  {l_ab:8.5f}  "
          f"{eta_stored:12.6f}  {eta_verify if eta_verify else float('nan'):12.6f}  "
          f"{'PASS' if match else 'FAIL':>8}  {'PASS' if valid else 'FAIL':>8}")

print()
print("Case 2 overall:", "ALL PASS ✓" if all_case2_ok else "FAILURES FOUND ✗")

# ─── Case 1 spot-check at alpha = 0 using known NV-centre optimal parameters ─
print()
print("=" * 80)
print("CROSS-VERIFICATION: CASE 1 (lambda_ab = lambda_cd = 1)")
print("=" * 80)

# The harmonic limit (alpha=0) maximum is exactly 1 - 1/R_omega
# Verified by using parameters that saturate the frequency cap
harmonic_bound = 1.0 - 1.0/R_OMEGA_MAX
print(f"Theoretical harmonic maximum eta = 1 - 1/{R_OMEGA_MAX} = {harmonic_bound:.6f}")
print()

# Sample parameters that saturate wh/wc = 7 and bc/bh = 25
bc, bh, wc, wh = 25.0, 1.0, 1.0, 7.0
for alpha in [0.0, 0.05, 0.10, 0.15, 0.20]:
    eta = compute_eta_case1(bc, bh, wc, wh, alpha)
    valid = is_valid(bc, bh, wc, wh, alpha)
    print(f"  alpha={alpha:.2f}: eta={eta:.6f}  valid={valid}")

# ─── Envelope data spot-check ─────────────────────────────────────────────────
print()
print("=" * 80)
print("CROSS-VERIFICATION: ENVELOPE DATA (optimized_envelope_data.json)")
print("=" * 80)
env_path = os.path.join(SCRIPT_DIR, "optimized_envelope_data.json")
with open(env_path, "r") as f:
    env_data = json.load(f)

alphas   = env_data["alphas"]
case1_etas = env_data["case1_etas"]
case2_etas = env_data["case2_etas"]

print(f"  {'alpha':>8}  {'eta_c1_stored':>14}  {'eta_c2_stored':>14}")
print("  " + "-"*42)
for a, e1, e2 in zip(alphas, case1_etas, case2_etas):
    print(f"  {a:8.4f}  {e1:14.6f}  {e2:14.6f}")

print()
# sanity checks
print("Sanity checks on envelope:")
print(f"  Case 1 at alpha=0  expected ≈ {harmonic_bound:.6f}, got {case1_etas[0]:.6f}  "
      f"{'PASS ✓' if abs(case1_etas[0] - harmonic_bound) < 1e-4 else 'FAIL ✗'}")
print(f"  Case 2 at alpha=0  expected ≈ 0.629889, got {case2_etas[0]:.6f}  "
      f"{'PASS ✓' if abs(case2_etas[0] - 0.629889) < 1e-4 else 'FAIL ✗'}")
print(f"  Case 1 monotone decreasing: "
      f"{'PASS ✓' if all(case1_etas[i] >= case1_etas[i+1] for i in range(len(case1_etas)-1)) else 'FAIL ✗'}")
print(f"  Case 2 monotone decreasing: "
      f"{'PASS ✓' if all(case2_etas[i] >= case2_etas[i+1] for i in range(len(case2_etas)-1)) else 'FAIL ✗'}")
print(f"  Case 1 eta <= 1 always:     "
      f"{'PASS ✓' if all(e <= 1.0 for e in case1_etas) else 'FAIL ✗'}")
print(f"  Case 2 eta <= 1 always:     "
      f"{'PASS ✓' if all(e <= 1.0 for e in case2_etas) else 'FAIL ✗'}")
print(f"  Case 1 >= Case 2 always:    "
      f"{'PASS ✓' if all(e1 >= e2 for e1, e2 in zip(case1_etas, case2_etas)) else 'FAIL ✗'}")

print()
print("VERIFICATION COMPLETE.")
