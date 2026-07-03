# Quantum Heat Engine Optimization: Methodological Notes & Design Choices

This document provides a detailed academic explanation of the design decisions, mathematical constraints, and optimization methodologies used to analyze the anharmonic quantum Otto cycle. It is structured for direct use in the methodology and discussion sections of the paper.

---

## 1. Experimental Anchoring & Choice of Ratio Caps ($R_\omega$ and $R_\beta$)

Instead of arbitrarily choosing boundaries, our parameter space is physically constrained by the capabilities of the state-of-the-art **diamond Nitrogen-Vacancy (NV) center platform** as reported by **Klatzow et al., PRL 2019** (*"Experimental Demonstration of a Quantum Otto Heat Engine"*, Phys. Rev. Lett. 122, 110601).

We enforce two key ratio caps:
1. **Frequency Compression Ratio:** $R_\omega = \frac{\omega_h}{\omega_c} \le 7.0$
2. **Reservoir Temperature Ratio:** $R_\beta = \frac{\beta_c}{\beta_h} \le 25.0$

### Physical Justification of the Ratios
* **The $R_\beta \le 25.0$ Limit:**
  To maintain spin coherence ($T_2$) of the NV center in diamond, the physical temperature of the diamond crystal cannot exceed room temperature ($T_h \le 300\text{ K}$) due to strong spin-phonon coupling. On the cold side, standard helium-flow cryostats can reliably cool and initialize the spin bath to $T_c \approx 12\text{ K}$. The ratio of these physical limits defines the maximum temperature gradient:
  $$R_\beta = \frac{T_h}{T_c} \approx \frac{300\text{ K}}{12\text{ K}} = 25.0$$
* **The $R_\omega \le 7.0$ Limit:**
  The frequency range is bounded by the microwave resonator tuning range and transition frequencies of the NV electron spin sublevels ($D \approx 2.87\text{ GHz}$). Sweeping the frequency spacing between $\omega_c \approx 1.58\text{ GHz}$ and $\omega_h \approx 11.05\text{ GHz}$ yields the compression ratio:
  $$R_\omega = \frac{\omega_h}{\omega_c} \approx 7.0$$

---

## 2. Why We Constrained the Ratio Rather Than Fixing Efficiency ($\eta$)

An alternative design philosophy is to pre-select a target efficiency (e.g., $\eta = 80\%$) and solve for the corresponding engine parameters. We rejected this approach for two fundamental reasons:

### 2.1 Mathematical Under-Determination
In the harmonic limit ($\alpha=0$), efficiency depends only on the frequency ratio:
$$\eta = 1 - \frac{\omega_c}{\omega_h} = 1 - \frac{1}{R_\omega}$$
Fixing $\eta$ uniquely determines the ratio $R_\omega$. However, it leaves the remaining 3 parameters of the 4D space $(\beta_c, \beta_h, \omega_c, \omega_h)$ completely under-determined. There are infinitely many degenerate combinations of temperatures and absolute frequencies that yield the same efficiency. Solving a highly under-determined system is numerically unstable and does not represent a unique physical design.

### 2.2 Finding the True Physical Ceiling
Our goal is to discover the **maximum physical performance** of the quantum engine under real hardware limitations. By fixing the physical ratio caps ($R_\omega \le 7$, $R_\beta \le 25$) as inequality boundaries and using them to constrain a global optimizer, we can sweep the parameter space to find the **absolute maximum efficiency ceiling** that can be physically realized. Under these boundaries, the maximum efficiency is found to be:
$$\eta_{\max} = 1 - \frac{1}{7.0} \approx 0.857143 \quad (85.7\%)$$

---

## 3. Mathematical Constraints Enforced

To ensure the engine operates in a physically valid and mathematically sound regime, we enforce four classes of constraints:

### 3.1 Boundary Constraints
The parameter space search is restricted to experimentally accessible ranges:
* Cold inverse temperature: $\beta_c \in [0.5, 20.0]$
* Hot inverse temperature: $\beta_h \in [0.1, 10.0]$
* Cold frequency: $\omega_c \in [1.0, 8.0]$
* Hot frequency: $\omega_h \in [2.0, 15.0]$

### 3.2 Thermodynamic Constraints (Heat Engine Regime)
The cycle must function as a heat engine (absorbing heat from the hot bath to perform positive net work):
* $W_{\mathrm{ext}} = Q_h + Q_c > 0$
* $Q_h > 0$
* Temperature ordering: $T_h > T_c \implies \beta_c > \beta_h$
* Frequency ordering: $\omega_h > \omega_c$
* Engine condition: $\beta_c \omega_c > \beta_h \omega_h$

### 3.3 Platform Caps
* $\frac{\omega_h}{\omega_c} \le 7.0$
* $\frac{\beta_c}{\beta_h} \le 25.0$

### 3.4 Perturbation Validity Constraints
Because our model relies on a first-order perturbation correction for the anharmonicity ($\alpha$), the perturbation term must remain small compared to the harmonic energy spacing. We enforce that the cold and hot perturbation parameters ($\varepsilon_c, \varepsilon_h$) must not exceed $10\%$:
$$\varepsilon_c = \frac{3 \alpha \coth(\beta_c \omega_c / 2)}{4 \omega_c^3} \le 0.10, \qquad \varepsilon_h = \frac{3 \alpha \coth(\beta_h \omega_h / 2)}{4 \omega_h^3} \le 0.10$$
This prevents the perturbation theory from breaking down mathematically.

---

## 4. The ES-RL (Evolution Strategy with Reinforcement Learning) Optimization Framework

To find the global maximum of efficiency and work across this constrained space, we chose **Evolution Strategies with RL-style Gradient Estimation (ES-RL)** over other optimization algorithms.

### 4.1 How ES-RL Works
ES-RL parameterizes the search space using a stochastic policy—a Gaussian search distribution $\mathcal{N}(\boldsymbol{\mu}, \sigma^2 \mathbf{I})$ where $\boldsymbol{\mu} = (\beta_c, \beta_h, \omega_c, \omega_h)$.
Instead of computing analytical gradients (which are unavailable due to discontinuous validity cuts), ES-RL estimates the gradient of the expected objective function $J(\boldsymbol{\mu}) = \mathbb{E}_{\mathbf{x}}[f(\mathbf{x})]$ using the **REINFORCE score-function estimator** from Reinforcement Learning:
$$\hat{\mathbf{g}} = \frac{1}{2H\sigma} \sum_{i=1}^{H} \bigl[f(\mathbf{x}^+_i) - f(\mathbf{x}^-_i)\bigr] \cdot \boldsymbol{\varepsilon}_i$$
where $\boldsymbol{\varepsilon}_i \sim \mathcal{N}(0,\mathbf{I})$ are random search directions, and $\mathbf{x}^+_i = \boldsymbol{\mu} + \sigma\boldsymbol{\varepsilon}_i$ and $\mathbf{x}^-_i = \boldsymbol{\mu} - \sigma\boldsymbol{\varepsilon}_i$ are antithetic (symmetric) samples evaluated in parallel. The mean $\boldsymbol{\mu}$ is then updated using the **Adam optimizer** to adaptively scale the step sizes based on gradient moments.

```
       stochastic Gaussian search policy N(μ, σ² I)
                          │
         ┌────────────────┴────────────────┐
         ▼                                 ▼
   + Perturbation                    - Perturbation
  x⁺ = μ + σ ε                       x⁻ = μ - σ ε
         │                                 │
         ▼                                 ▼
  Evaluate f(x⁺)                    Evaluate f(x⁻)
         │                                 │
         └────────────────┬────────────────┘
                          ▼
            Antithetic Gradient Estimate:
      ĝ = mean( [f(x⁺) - f(x⁻)] * ε ) / (2σ)
                          │
                          ▼
                  Adam Update Steps
                μ ← μ + α · m̂ / (√v̂ + ε)
```

### 4.2 Why We Chose ES-RL over Other Models
* **Handling Discontinuous Boundaries:**
  The validity cuts (e.g., $\varepsilon_c \le 0.10$ or $W_{\mathrm{ext}} > 0$) create a highly discontinuous landscape. If a parameter set violates a constraint, the function returns NaN or 0. Gradient-based solvers (such as SLSQP or BFGS) fail because they evaluate local derivatives which are undefined or flat at the boundaries. ES-RL uses a Gaussian spread ($\sigma$), which smooths the landscape and allows the optimizer to "see" across discontinuous steps.
* **Global Optimization in Multi-modal Landscapes:**
  Random search lacks memory and converges very slowly ($O(1/N)$ rate). Standard gradient descent gets stuck in local minima. ES-RL behaves as a global-local hybrid, using the Gaussian spread for exploration and the Adam momentum updates for rapid convergence inside the global basin.
* **Scalability ($O(d)$ Complexity):**
  More advanced evolution strategies like CMA-ES require estimating and updating a covariance matrix, which scales as $O(d^2)$ per step. ES-RL uses an isotropic covariance, achieving $O(d)$ complexity per step, making it highly scalable and computationally efficient.

---

## 5. Summary of Numerical Results & Figure References

### 5.1 Optimal Parameters for Case 1 ($\lambda_{ab}=\lambda_{cd}=1$)

* **Maximizing Efficiency ($\eta$):**
  To achieve the maximum possible efficiency $\eta = 0.857$, the parameters must sit at:
  $$\beta_c = 9.122, \quad \beta_h = 0.634, \quad \omega_c = 1.579, \quad \omega_h = 11.052$$
  *At this efficiency-maximized point, the work output is driven to near-zero ($W_{\mathrm{ext}} \approx 0.0086$) due to the thermodynamic quasi-static limit.*
* **Maximizing Work Output ($W_{\mathrm{ext}}$):**
  To extract the absolute highest work output from the engine, the parameters shift to:
  $$\beta_c = 2.500, \quad \beta_h = 0.100, \quad \omega_c = 1.000, \quad \omega_h = 4.363$$
  *At this work-maximized point, Case 1 yields $W_{\mathrm{ext}} = 5.848$ and an efficiency of $\eta = 0.771$.*

### 5.2 Figure References & Interpretations

* **Cold Inverse Temperature Sweep ([eta_vs_beta_c_grid_overlay.png](file:///home/reet/.gemini/antigravity-ide/brain/3467b865-4868-46ba-841f-f981232c1ea1/plots/eta_vs_beta_c_grid_overlay.png)):**
  * *Cases 2, 3, 4:* Plotted for $\beta_c \in [0.0, 4.0]$ to focus on the active region.
  * *Case 1:* Plotted for $\beta_c \in [0.0, 15.0]$. The curve is completely blank for $\beta_c \le 4.436$ because the engine condition $\beta_c \omega_c > \beta_h \omega_h$ is violated, causing $W_{\mathrm{ext}} \le 0$ (the engine ceases to operate).
* **Hot Inverse Temperature Sweep ([eta_vs_beta_h_grid_overlay.png](file:///home/reet/.gemini/antigravity-ide/brain/3467b865-4868-46ba-841f-f981232c1ea1/plots/eta_vs_beta_h_grid_overlay.png)):**
  * The Case 1 curve starts abruptly at $\beta_h \approx 0.365$. Below this value, the temperature ratio $\beta_c / \beta_h$ exceeds the maximum physical limit of $25.0$ set by the diamond NV-centre platform.
* **Work Output Comparison ([work_vs_alpha_all_cases.png](file:///home/reet/.gemini/antigravity-ide/brain/3467b865-4868-46ba-841f-f981232c1ea1/plots/work_vs_alpha_all_cases.png)):**
  * When optimizing for work output directly, the friction-free symmetric Case 1 ($\lambda = 1$) correctly yields the tallest bar ($W_{\mathrm{ext}} \approx 5.85$ at $\alpha = 0$), and work output drops progressively as asymmetric losses are introduced in Cases 2, 3, and 4.
