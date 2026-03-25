# Anharmonic Quantum Otto Cycle — Constrained Efficiency Optimisation

> A publication-ready study benchmarking classical gradient-based solvers against
> evolution-strategy reinforcement learning on the maximum-efficiency problem of an
> anharmonic quantum heat engine, with physically motivated and literature-cited constraints.

---

## Table of Contents
1. [Motivation](#1-motivation)
2. [The Physics Model](#2-the-physics-model)
3. [Why Constraints Matter — Removing Unphysical Samples](#3-why-constraints-matter)
4. [How the Constraint Values Were Chosen](#4-how-the-constraint-values-were-chosen)
5. [Experimental Platform Survey](#5-experimental-platform-survey)
6. [Optimisation Methods Compared](#6-optimisation-methods-compared)
7. [Results](#7-results)
8. [Verification](#8-verification)
9. [File Index](#9-file-index)
10. [How to Reproduce](#10-how-to-reproduce)
11. [References](#11-references)

---

## 1. Motivation

Quantum heat engines are at the frontier of quantum thermodynamics, with
direct experimental realisations in superconducting qubits [1], trapped ions [2],
and nitrogen-vacancy (NV) centres in diamond [3]. The central engineering question
is: *given a particular quantum working medium and a set of experimentally
realisable control parameters, what is the maximum thermodynamic efficiency the
engine can achieve, and which parameter values achieve it?*

This study addresses that question for the **anharmonic quantum Otto cycle** with
a quartic working medium, using the perturbation-theoretic energy spectrum derived
in the reference paper [0]. We benchmark four optimisation algorithms — Random
Search, SLSQP, CMA-ES, and ES-RL — across four experimental platforms, providing
a reproducible computational benchmark suitable for *Computer Physics Communications (CPC)*.

---

## 2. The Physics Model

### 2.1 Working Medium

The working medium is a one-dimensional quartic anharmonic oscillator:

$$\hat{H} = \frac{\hat{p}^2}{2m} + \frac{1}{2}m\omega^2\hat{x}^2 + \lambda\hat{x}^4$$

To first order in the anharmonicity parameter $\lambda$, the energy eigenvalues are
(Eq. 3 of Ref. [0]):

$$E_n(\omega, \lambda) = \left(n+\frac{1}{2}\right)\hbar\omega
+ \frac{3\lambda\hbar^2}{4m^2\omega^2}\left(2n^2+2n+1\right)$$

### 2.2 The Otto Cycle

The quantum Otto cycle consists of four strokes:
- **A→B (isentropic compression):** frequency $\omega_c \to \omega_h$, no heat exchange
- **B→C (hot isochoric):** thermalisation with bath at inverse temperature $\beta_h = 1/k_BT_h$
- **C→D (isentropic expansion):** frequency $\omega_h \to \omega_c$
- **D→A (cold isochoric):** thermalisation with bath at inverse temperature $\beta_c = 1/k_BT_c$

### 2.3 Efficiency Formula (Eqs. 8–10 of Ref. [0])

Defining the short-hand:

$$X = \coth\!\left(\frac{\beta_h\omega_h}{2}\right), \qquad
Y = \coth\!\left(\frac{\beta_c\omega_c}{2}\right)$$

the heat exchanges and efficiency are:

$$Q_h = \frac{\omega_h}{2}(X-Y) + \frac{3\lambda}{8\omega_h^2}(X^2-Y^2) \tag{Eq. 8}$$

$$Q_c = \frac{\omega_c}{2}(Y-X) + \frac{3\lambda}{8\omega_c^2}(Y^2-X^2) \tag{Eq. 9}$$

$$W_{\mathrm{ext}} = Q_h + Q_c, \qquad \eta = \frac{W_{\mathrm{ext}}}{Q_h} \tag{Eq. 10}$$

**Numerical implementation:** we use a numerically stable `coth` that avoids
overflow/underflow:
- Laurent series $\coth(x) \approx 1/x + x/3 - x^3/45$ for $|x| < 10^{-3}$
- Asymptotic form $\coth(x) \approx \text{sgn}(x)(1 + 2e^{-2|x|})$ for $|x| > 20$
- Direct ratio $\cosh(x)/\sinh(x)$ otherwise

### 2.4 Five Free Parameters

The full design space is:

$$\mathbf{x} = (\beta_c,\; \beta_h,\; \omega_c,\; \omega_h,\; \lambda) \in \mathbb{R}^5$$

---

## 3. Why Constraints Matter

### 3.1 Engine-Mode Constraints (necessary)

For the cycle to operate as a heat engine (not a refrigerator), the following
ordering constraints must hold [0]:

| Constraint | Physical meaning |
|---|---|
| $\omega_c < \omega_h$ | Compression stroke increases frequency |
| $\beta_h < \beta_c$ | Hot bath hotter than cold bath |
| $\beta_c\omega_c > \beta_h\omega_h$ | Positive work output (engine mode) |
| $\lambda \geq 0$ | Physical anharmonicity |

### 3.2 The Boundary-Pushing Problem

The engine-mode constraints above are *necessary but not sufficient* to
produce a meaningful optimisation benchmark.

Without additional caps, the optimizer can exploit:

$$\eta \xrightarrow[\omega_c/\omega_h \to 0]{\lambda \to 0} 1$$

by sending $\omega_c \to 0$, $\omega_h \to \infty$, $\beta_h \to 0$,
$\beta_c \to \infty$ — all while satisfying every engine-mode inequality.
This is **physically unphysical**: it exploits parameter regimes where
(a) the first-order perturbation theory breaks down, and (b) the frequencies
and temperatures are not experimentally realisable.

The result is that any reasonably written optimiser finds $\eta \approx 0.99$
trivially by running to the parameter boundary — the benchmark measures
"who runs fastest to the corner," not "who solves the physics problem."

---

## 4. How the Constraint Values Were Chosen

We add three constraint layers, each with a literature-cited justification.

### Layer 1 — Frequency Ratio Cap: $R_\omega = \omega_h / \omega_c \leq 6$

**Analytical consequence:** In the harmonic limit ($\lambda \to 0$),
the Otto efficiency is bounded by $\eta \leq 1 - \omega_c/\omega_h$.
Capping $\omega_h/\omega_c \leq R_\omega$ directly caps $\eta \leq 1 - 1/R_\omega$.

**How $R_\omega = 6$ was chosen:**

The reference paper [0] uses $\omega_c = 2$, $\omega_h = 3$ in its Fig. 2 
illustrative example, giving a ratio of $R_\omega^{(0)} = 1.5$.
We scale this by a factor of 4 to allow broad exploration while remaining
in a physically realisable regime:

$$R_\omega = 4 \times R_\omega^{(0)} = 4 \times 1.5 = 6 \implies \eta_{\max} = 1 - \tfrac{1}{6} = 0.833$$

A factor of 4 is chosen (rather than 2) after surveying the literature
(Section 5 below): trapped-ion experiments [2] achieve ratios up to 4–5,
and NV-centre experiments [3] up to 7. Our $R_\omega = 6$ covers
superconducting and trapped-ion platforms and is conservative for NV centres.

### Layer 2 — Temperature Ratio Cap: $R_\beta = \beta_c / \beta_h \leq 8$

**Analytical consequence:** The Carnot efficiency is $\eta_{\mathrm{Carnot}} = 1 - T_h/T_c = 1 - \beta_c^{-1}/\beta_h^{-1} = 1 - \beta_h/\beta_c$.
Capping $\beta_c/\beta_h \leq R_\beta$ caps $\eta_{\mathrm{Carnot}} \leq 1 - 1/R_\beta$.

**How $R_\beta = 8$ was chosen:**

The reference paper [0] Fig. 2 uses $\beta_h = 0.5\,\beta_c$, giving
$R_\beta^{(0)} = 2$. We scale by 4:

$$R_\beta = 4 \times R_\beta^{(0)} = 4 \times 2 = 8 \implies \eta_{\mathrm{Carnot}} = 1 - \tfrac{1}{8} = 0.875$$

**Consistency check:** $\eta_{\mathrm{max}} = 0.833 < \eta_{\mathrm{Carnot}} = 0.875$ ✓  
The Otto efficiency stays below the Carnot bound, as thermodynamics requires.
The frequency ratio cap is the *binding* constraint in our domain.

### Layer 3 — Perturbation Validity: $\lambda \leq 0.2$

The paper's energy eigenvalues (Eq. 3 of Ref. [0]) are derived to
*first order in $\lambda$*. For the perturbative expansion to be valid,
the anharmonic correction must remain small compared to the harmonic term:

$$\frac{3\lambda}{8\omega^2}\,\coth^2\!\left(\tfrac{\beta\omega}{2}\right)
\;\ll\; \frac{\omega}{2}\,\coth\!\left(\tfrac{\beta\omega}{2}\right)$$

Evaluating at typical operating points and requiring the ratio $\leq 10\%$
gives $\lambda \lesssim 0.2$, which we adopt as a hard cap.

### Summary Table

| Constraint | Value | Anchor | Source |
|---|---|---|---|
| $\omega_h / \omega_c \leq R_\omega$ | **6** | $4\times$ paper Fig. 2 ratio (1.5) | Ref. [0] Fig. 2 |
| $\beta_c / \beta_h \leq R_\beta$ | **8** | $4\times$ paper Fig. 2 ratio (2.0) | Ref. [0] Fig. 2 |
| $\lambda \leq \lambda_{\max}$ | **0.2** | First-order perturbation validity | Ref. [0] Eq. 3 |
| $\beta_c\omega_c > \beta_h\omega_h$ | — | Engine mode | Standard |

---

## 5. Experimental Platform Survey

We study four experimental platforms and their realistic ratio ranges,
used to define per-platform optimisation problems.

| Platform | $R_\omega$ | $R_\beta$ | $\eta_{\max} = 1 - 1/R_\omega$ | Reference |
|---|---|---|---|---|
| Reference paper (Fig. 2) | 1.5 | 2 | 0.333 | Ref. [0] |
| Superconducting qubits | 2.0 | 6 | 0.500 | Peterson et al. [1] |
| Trapped ions | 4.0 | 12 | 0.750 | Roßnagel et al. [2] |
| NV centres (diamond) | 7.0 | 25 | 0.857 | Klatzow et al. [3] |

**Superconducting qubits [1]:** Transmon qubit frequencies are tunable in the
range ~4–9 GHz using flux bias; achievable ratios $\omega_h/\omega_c \approx 1.3$–2.0.
Dilution refrigerator base temperatures ~15 mK achievable; hot bath
injected electronically up to ~100 mK, giving $\beta_c/\beta_h \approx 3$–8.

**Trapped ions [2]:** The secular trapping frequency can be modulated
over a factor of ~2–5 by varying the trap voltage. Ion temperatures span
~50 μK (Doppler cooled) to ~1 mK (thermal), giving $\beta_c/\beta_h$ up to ~20.

**NV centres [3]:** The zero-field splitting of the NV spin triplet is
2.87 GHz; microwave driving can effectively set $\omega_c$ and $\omega_h$
over a wider range (ratio ~3–8) using dressed-state ladders.
The spin temperature can span a very wide range due to laser initialisation.

---

## 6. Optimisation Methods Compared

All methods operate on the same constrained 5-dimensional domain.

| Method | Type | Notes |
|---|---|---|
| **Random** | Stochastic baseline | Uniform sampling over feasible set |
| **SLSQP** | Gradient-based | Multi-start scipy.optimize; second-order |
| **CMA-ES** | Evolution Strategy | Pure-NumPy; covariance matrix adaptation |
| **ES-RL** | Evolutionary RL | Antithetic sampling + Adam update |

Two benchmarking tracks:
- **Cold start:** random feasible initialisation — tests raw algorithm power
- **Warm start:** physics-informed initialisation near the paper's Fig. 2 regime

---

## 7. Results

### 7.1 Benchmark (threshold η ≥ 0.75, 10 seeds, R_ω=6, R_β=8)

| Method | η_max | Success (cold) | Median evals | Median time |
|---|---|---|---|---|
| Random | 0.8313 | 10/10 | 18 | 0.0018 s |
| **SLSQP** | **0.8333** | **10/10** | **31** | 0.0088 s |
| CMA-ES | 0.8333 | 9/10 | 47 | 0.0018 s |
| ES-RL (warm) | 0.8311 | 10/10 | 45 | 0.0018 s |

### 7.2 Platform-Optimal Parameter Sets (verified independently)

| Platform | $\beta_c$ | $\beta_h$ | $\omega_c$ | $\omega_h$ | $\lambda$ | $\eta$ verified |
|---|---|---|---|---|---|---|
| Reference paper | 7.093 | 4.622 | 3.671 | 5.506 | 0.000 | **0.3333 ✓** |
| Superconducting | 8.445 | 1.857 | 4.348 | 8.696 | 0.000 | **0.5000 ✓** |
| Trapped ions | 26.569 | 2.923 | 0.760 | 3.039 | 0.000 | **0.7500 ✓** |
| NV centres | 30.000 | 3.643 | 0.594 | 4.161 | 0.000 | **0.8571 ✓** |

**Key physical observations:**

1. **$\lambda = 0$ at every optimum.** The anharmonic term always *reduces*
   efficiency relative to the harmonic case (confirmed by Ref. [0] Fig. 2,
   which shows $\eta$ decreasing as $\lambda$ increases). The optimizer
   independently discovers this, setting $\lambda = 0$ in every case.

2. **$\omega_h/\omega_c = R_\omega$ exactly** at every optimum. The frequency
   ratio cap is always the binding constraint, consistent with the analytic
   harmonic bound $\eta_{\max} = 1 - 1/R_\omega$.

3. **$\beta_c/\beta_h < R_\beta$** at every optimum. The temperature ratio cap
   never binds first — confirming that $R_\beta$ is well-chosen to be non-trivially
   looser than $R_\omega$.

4. **All solutions satisfy $\eta < \eta_{\mathrm{Carnot}}$** — the second law
   of thermodynamics is respected in every case.

---

## 8. Verification

`verify_eta_max.py` provides a fully independent cross-check:
- **Re-implements Eqs. 8–10 from scratch** with no imports from project code
- Loads optimal parameters from `results/optimal_parameters.json`
- Substitutes each optimal set and computes $\eta$, $Q_h$, $Q_c$, $W_{\mathrm{ext}}$
- Checks: $\eta_{\mathrm{verify}} = \eta_{\mathrm{optimizer}}$ to $10^{-4}$ precision

All four platforms pass with ✓.

---

## 9. File Index

```
anharmonic_otto_study/
├── codes/
│   ├── physics_model.py              # Constrained physics engine (stable coth, ratio caps)
│   ├── run_cpc_benchmark.py          # 4-method benchmark + plots fig1–fig4 (one command)
│   ├── find_optimal_parameters.py    # SLSQP finds eta_max for 4 platforms → JSON
│   ├── verify_eta_max.py             # STANDALONE Eqs 8-10 verification (no shared imports)
│   └── plot_constraint_justification.py   # fig5: eta_max vs R_omega/R_beta maps
├── plots/
│   ├── fig1_convergence.png          # Best-so-far η vs evaluations (median ± IQR)
│   ├── fig2_threshold.png            # Evaluations + time to threshold
│   ├── fig3_violin.png               # Final η distribution across seeds
│   ├── fig4_feasible_space.png       # Constrained feasible domain scatter
│   ├── fig5_constraint_justification.png  # η_max = f(R_omega, R_beta) — 3 panels
│   ├── fig6_platform_comparison.png  # Platform-specific eta_max comparison
│   └── fig7_verification.png         # Independent verification figure
├── results/
│   ├── cpc_benchmark.json            # Raw benchmark data (all seeds, all methods)
│   └── optimal_parameters.json       # Optimal (βc, βh, ωc, ωh, λ) per platform
└── README.md                         # This file
```

---

## 10. How to Reproduce

```bash
# Install dependencies
pip install numpy scipy matplotlib

cd codes/

# Step 1: Run the full 4-method benchmark (generates fig1–fig4)
python3 run_cpc_benchmark.py

# Step 2: Generate the constraint justification figure (fig5)
python3 plot_constraint_justification.py

# Step 3: Find optimal parameters for all 4 experimental platforms (generates fig6)
python3 find_optimal_parameters.py

# Step 4: Independently verify all results (generates fig7)
python3 verify_eta_max.py
```

All scripts are self-contained and write their outputs to `../results/` and `../plots/`.

---

## 11. References

**[0]** "Quantum Otto cycle with inner friction: finite-time and disorder effects" —
the primary reference paper for the anharmonic energy eigenvalues (Eq. 3),
heat fluxes (Eqs. 8–9), and efficiency formula (Eq. 10); Fig. 2 provides the
operating-point anchor for our ratio caps ($\omega_c=2$, $\omega_h=3$,
$\beta_h=0.5\beta_c$).

**[1]** J. P. S. Peterson, T. B. Batalhão, M. Herrera, A. M. Souza, R. S. Sarthour,
I. S. Oliveira, R. M. Serra,
*"Experimental Characterization of a Spin Quantum Heat Engine"*,
**Physical Review Letters** 123, 240601 (2019).
DOI: [10.1103/PhysRevLett.123.240601](https://doi.org/10.1103/PhysRevLett.123.240601)
→ *Used for:* superconducting platform ratio bounds ($R_\omega=2$, $R_\beta=6$).

**[2]** J. Roßnagel, S. T. Dawkins, K. N. Tolazzi, O. Abah, E. Lutz, F. Schmidt-Kaler,
K. Singer,
*"A single-atom heat engine"*,
**Science** 352, 325–329 (2016).
DOI: [10.1126/science.aad6320](https://doi.org/10.1126/science.aad6320)
→ *Used for:* trapped-ion platform ratio bounds ($R_\omega=4$, $R_\beta=12$).

**[3]** J. Klatzow, J. N. Becker, P. M. Ledingham, C. Weinzetl, K. T. Kaczmarek,
D. J. Saunders, J. Nunn, I. A. Walmsley, R. Uzdin, E. Poem,
*"Experimental Demonstration of Quantum Effects in the Operation of Microscopic Heat Engines"*,
**Physical Review Letters** 122, 110601 (2019).
DOI: [10.1103/PhysRevLett.122.110601](https://doi.org/10.1103/PhysRevLett.122.110601)
→ *Used for:* NV-centre (diamond) platform ratio bounds ($R_\omega=7$, $R_\beta=25$).
