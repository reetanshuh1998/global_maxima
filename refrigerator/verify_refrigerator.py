import json
import os
import sys
sys.path.append('/home/reet/monika/anharmonic_otto_study/refrigerator')
from physics_model import compute_metrics, is_valid_physics

def verify_all():
    res_path = '/home/reet/monika/anharmonic_otto_study/refrigerator/results/refrigerator_optima.json'
    if not os.path.exists(res_path):
        print("Results file not found.")
        return

    with open(res_path, 'r') as f:
        data = json.load(f)

    print("=== Refrigerator Mode Verification ===")
    for key, val in data.items():
        p = val['params']
        m = compute_metrics(p['beta_c'], p['beta_h'], p['omega_c'], p['omega_h'], p['lambda'])
        
        cop, qc, qh, w, carnot, _, ok = m
        
        print(f"\nTarget: {key}")
        print(f"  COP: {cop:.4f}, Carnot Limit: {carnot:.4f}")
        print(f"  Qc: {qc:.6e}, Qh: {qh:.6e}, W: {w:.6e}")
        
        # Check 1: COP < Carnot
        if cop > carnot + 1e-6:
            print("  [FAIL] COP exceeds Carnot limit!")
        else:
            print("  [PASS] COP <= Carnot limit.")
            
        # Check 2: Physical signs
        if qc > 0 and qh < 0 and w < 0:
            print("  [PASS] Physical signs (Qc>0, Qh<0, W<0) correct.")
        elif cop == 0:
            print("  [INFO] Stagnation/Infeasible point.")
        else:
            print("  [FAIL] Incorrect physical signs!")

if __name__ == "__main__":
    verify_all()
