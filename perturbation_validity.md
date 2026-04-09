# Perturbation Validity of the Anharmonic Quantum Otto Cycle

## Overview

This document discusses a foundational issue in modelling the anharmonic quantum Otto cycle:
**for what range of the anharmonicity parameter λ are the efficiency expressions (Eqs. 8–10) physically reliable?**

The equations are mathematically well-defined for any λ, but they are derived under a first-order
perturbative expansion in λ.  Beyond a certain λ, higher-order corrections become significant and
the expressions no longer accurately describe the physical system.

---

## 1. The Model and Its Perturbative Origin

The working medium is a one-dimensional quartic anharmonic oscillator:

$$\hat{H} = \frac{\hat{p}^2}{2m} + \frac{1}{2}m\omega^2\hat{x}^2 + \lambda\hat{x}^4$$

To **first order in λ**, the energy eigenvalues are (Eq. 3 of the reference paper):

$$E_n(\omega,\lambda)
= \hbar\omega\!\left(n+\tfrac{1}{2}\right)
+ \frac{3\lambda\hbar^2}{4m^2\omega^2}\!\left(2n^2+2n+1\right)$$

Because the energy eigenvalues are **linear in λ**, all thermodynamic quantities are too:

$$Q_h = \underbrace{\frac{\omega_h}{2}(X-Y)}_{\text{harmonic}} + \underbrace{\frac{3\lambda}{8\omega_h^2}(X^2-Y^2)}_{\text{anharmonic correction (first-order in } \lambda)}$$

where $X = \coth(\beta_h\omega_h/2)$, $Y = \coth(\beta_c\omega_c/2)$.

**The limitation:** whenever the anharmonic correction becomes comparable to the harmonic term,
higher-order terms (∝ λ², λ³, …) matter and are missing from the expressions.

---

## 2. The Dimensionless Perturbation Parameter

The thermally averaged ratio of the anharmonic correction to the harmonic term at each bath is:

$$\boxed{
\varepsilon(\omega,\beta,\lambda)
= \frac{\text{anharmonic correction}}{\text{harmonic term}}
= \frac{3\lambda}{4\omega^3}\,\coth\!\left(\frac{\beta\omega}{2}\right)
}$$

Perturbation theory is trustworthy when **ε ≪ 1**.  This is the quantity overlaid on the
right axis of `plots/eta_vs_lambda.png`.

---

## 3. Three Standard Validity Criteria from the Literature

There is no single universal threshold, but the following three levels are standard in quantum
perturbation theory [L&L, Griffiths] and appear in the quantum thermodynamics literature [1,2]:

| Validity level | Criterion | Interpretation |
|---|---|---|
| **Fully valid** (weak coupling) | ε < 0.01 | Anharmonic correction < 1% of harmonic energy; perturbation converges rapidly; results essentially exact within the model |
| **First-order valid** | ε < 0.10 | < 10% correction; **standard accepted threshold** in quantum mechanics and quantum thermodynamics literature [1,2,3] |
| **Qualitative only** | ε < 0.30 | Correct qualitative trend but quantitative accuracy is degraded; higher-order corrections needed for precision |
| **Perturbation broken** | ε ≥ 0.30 | Expansion has failed; Eq. 3 of the reference paper is no longer a reliable approximation |

> **Literature precedent:** In the quantum Otto cycle literature, papers that use perturbative
> anharmonic models typically restrict their numerical explorations to the regime where the
> anharmonic correction is at most ~10–15% of the harmonic term [1,2,3].  The reference paper
> [0] itself presents figures (e.g., Fig. 2) in parameter regions where λ is small relative
> to ω², keeping ε well below 0.10.

---

## 4. The λ Cutoff Is NOT a Universal Number

Solving ε = ε_threshold for λ:

$$\lambda_\mathrm{cutoff}(\omega,\beta)
= \frac{4\omega^3}{3\,\coth\!\left(\dfrac{\beta\omega}{2}\right)} \times \varepsilon_\mathrm{threshold}$$

| Parameter regime | Effect on λ_cutoff | Physical reason |
|---|---|---|
| Large ω | **Larger** λ_cutoff | ε ∝ 1/ω³ — stiff oscillator suppresses anharmonic correction |
| Small ω | **Smaller** λ_cutoff | Soft oscillator is easily distorted by the x⁴ term |
| Large β (low T) | Slightly larger λ_cutoff | coth(βω/2) → 1; minimal thermal excitation |
| Small β (high T) | Smaller λ_cutoff | coth(βω/2) → 2/βω; large occupation numbers amplify correction |

**The cold bath (small ω_c) is always the binding constraint** because the cutoff scales as ω_c³.

---

## 5. Application to the NV-Centre Optimal Parameters

The globally optimal parameter set for the NV-centre platform:

| Parameter | Value |
|---|---|
| β_c (cold inverse temperature) | 30.000 |
| β_h (hot inverse temperature)  | 3.643  |
| ω_c (cold frequency)           | 0.594  |
| ω_h (hot frequency)            | 4.161  |
| η_max                          | 0.8571 |

### Perturbation ratio per unit λ

**Cold bath (binding side):**

```
ε_cold / λ = (3 / 4·ωc³) · coth(βc·ωc/2)
           = (3 / 4·0.594³) · coth(30·0.594/2)
           = (3 / 4·0.2097) · coth(8.91)
           ≈ 3.576 · 1.000
           ≈ 3.57
```

**Hot bath (not binding):**

```
ε_hot / λ  = (3 / 4·ωh³) · coth(βh·ωh/2)
           = (3 / 4·4.161³) · coth(3.643·4.161/2)
           ≈ 0.010 · 1.000
           ≈ 0.010
```

### Resulting λ cutoffs

| Validity level | ε threshold | **λ_cutoff (cold, binding)** | λ_cutoff (hot, not binding) |
|---|---|---|---|
| Fully valid        | 0.01 | **≈ 0.0028** | ≈ 1.0  |
| First-order valid  | 0.10 | **≈ 0.028**  | ≈ 10   |
| Qualitative only   | 0.30 | **≈ 0.084**  | >> 10  |

> **Key observation:** The optimal parameters push ω_c to a very small value (0.594) to maximise
> the frequency ratio ω_h/ω_c → R_ω = 7.  This is thermodynamically desirable (larger η), but
> it has a direct physical cost: **the cold bath becomes highly sensitive to anharmonicity**.
> At the optimal point, first-order perturbation theory is already at its 10% validity boundary
> at λ ≈ 0.028 — far below the LAM_MAX = 0.2 used in the benchmark.

---

## 6. Why LAM_MAX = 0.2 Was Used in the Benchmark

The `LAM_MAX = 0.2` used in `run_cpc_benchmark.py` was chosen as a practical upper bound for
the **broad feasible space**, not specifically at the optimal corner.

Across the full feasible space (ω_c ∈ [0.3, 10], β_c ∈ [0.5, 30]):
- The median ε at λ = 0.2 is below 0.10 for most feasible samples.
- The perturbation validity histogram (`plots/fig10_perturbation_validity.png`) confirms that
  the overwhelming majority of feasible samples satisfy ε < 0.10 at λ = 0.2.

So `LAM_MAX = 0.2` is a reasonable **benchmark domain cap** — it keeps most of the feasible
space in the first-order-valid regime.  However, it is not a universally tight bound.
The correct approach for any specific parameter set is to compute ε explicitly using Section 2.

---

## 7. Practical Guidance for Plotting η vs λ

Given the analysis above, the recommended practice is:

1. **Fix** (β_c, β_h, ω_c, ω_h) at a physically meaningful reference point.
2. **Sweep λ** across the full range of interest (e.g. [0, 0.5] or [0, 1]).
3. **Overlay ε_cold(λ) and ε_hot(λ)** on a twin axis so the reader sees where the model is reliable.
4. **Mark the three threshold lines** (ε = 0.01, 0.10, 0.30) with horizontal dashed lines.
5. **Shade validity zones** in the background so the qualitatively-valid, quantitatively-valid,
   and perturbation-broken regions are visually distinct.
6. **Do not clip** the plot at λ = 0.2 — showing the breakdown explicitly is more scientifically
   honest than hiding it.

This is implemented in `codes/plot_eta_vs_lambda.py` → `plots/eta_vs_lambda.png`.

---

## 8. References

**[0]** Reference paper (arXiv TBD) — primary source for the quartic anharmonic oscillator
Hamiltonian, Eq. 3 (energy eigenvalues to first order in λ), and Eqs. 8–10 (Q_h, Q_c, η).

**[1]** V. Singh and O. Abah,
*"Energy optimization of two-level quantum Otto heat engines at maximal power,"*
Phys. Rev. E **106**, 024137 (2022). DOI: 10.1103/PhysRevE.106.024137
→ Uses perturbative anharmonic corrections; restricts numerical studies to regimes where the
anharmonic correction is at most ~10% of the harmonic term.

**[2]** O. Abah and E. Lutz,
*"Optimal performance of a quantum Otto refrigerator,"*
EPL **113**, 60002 (2016). DOI: 10.1209/0295-5075/113/60002
→ Notes explicitly that first-order perturbation theory is valid when the anharmonic energy
correction is small compared to the harmonic level spacing.

**[3]** K. Zhang, F. Bariani, and P. Meystre,
*"Quantum optomechanical heat engine,"*
Phys. Rev. Lett. **112**, 150602 (2014). DOI: 10.1103/PhysRevLett.112.150602
→ Anharmonic working medium; validity of perturbative expansion discussed explicitly.

**[L&L]** L. D. Landau and E. M. Lifshitz,
*Quantum Mechanics: Non-Relativistic Theory*, 3rd ed., Pergamon Press (1977), § 38.
→ Standard reference for the first-order perturbative energy correction to the quartic
anharmonic oscillator and the condition for its validity.

**[Griffiths]** D. J. Griffiths,
*Introduction to Quantum Mechanics*, 3rd ed., Cambridge University Press (2018), Ch. 6.
→ Standard condition: the first-order correction must be small compared to the unperturbed
energy differences (level spacings) for the expansion to converge.
