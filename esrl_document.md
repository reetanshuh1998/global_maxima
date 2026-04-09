# Evolution Strategies with RL-style Gradient Estimation (ES-RL)
## A Mathematical Derivation and Literature Review

---

> **Abstract.** This document presents a self-contained mathematical exposition of the
> Evolution Strategy with Reinforcement-Learning-style gradient estimation (ES-RL)
> algorithm.  Starting from the first principles of stochastic optimisation, we derive
> the score-function (REINFORCE) gradient estimator, motivate antithetic sampling as a
> variance-reduction technique, and derive the Adam adaptive moment update.  The
> complete ES-RL algorithm is presented in a unified framework, connections to natural
> evolution strategies and policy-gradient methods are established, and convergence
> properties are summarised.  All results are supported by primary literature references.

---

## Table of Contents

1. [Introduction and Motivation](#1-introduction-and-motivation)
2. [Problem Formulation](#2-problem-formulation)
3. [Evolution Strategies: Historical Background](#3-evolution-strategies-historical-background)
4. [The Score-Function Gradient Estimator](#4-the-score-function-gradient-estimator)
5. [Antithetic Sampling: Variance Reduction](#5-antithetic-sampling-variance-reduction)
6. [The Adam Optimiser](#6-the-adam-optimiser)
7. [The Complete ES-RL Algorithm](#7-the-complete-es-rl-algorithm)
8. [Convergence Analysis](#8-convergence-analysis)
9. [Connections to Related Methods](#9-connections-to-related-methods)
10. [Summary and Practical Guidance](#10-summary-and-practical-guidance)
11. [References](#11-references)

---

## 1. Introduction and Motivation

Classical gradient-based optimisation requires access to $\nabla f(\mathbf{x})$,
either analytically or via automatic differentiation.  In many scientifically and
industrially important settings this is not possible:

- the objective $f$ is a **black box** (a physics simulation, an experiment, a quantum circuit);
- $f$ is **non-differentiable** or **discontinuous**;
- the search space involves **discrete decisions** mixed with continuous parameters;
- the computational graph is too large for backpropagation.

Evolution Strategies (ES) address all of these cases by estimating the gradient of
a *smoothed* version of $f$ using only function evaluations.  The modern
Reinforcement-Learning framing — due primarily to Salimans et al. [1] and Wierstra
et al. [2, 3] — reinterprets ES as a policy-gradient algorithm on the distribution
of candidate solutions, enabling the use of powerful optimisers (Adam [4]) developed
for deep learning.

---

## 2. Problem Formulation

We seek to **maximise** a scalar objective:

$$\mathbf{x}^* = \arg\max_{\mathbf{x} \in \mathcal{X}} f(\mathbf{x})$$

where $f : \mathcal{X} \subseteq \mathbb{R}^d \to \mathbb{R}$ may be **multimodal**,
**noisy**, and **non-differentiable**.  Direct access to $\nabla f$ is not assumed.

### 2.1 Smoothing via a Search Distribution

Rather than optimising $f$ directly, ES introduces a family of probability
distributions $\{p_{\boldsymbol{\theta}}\}_{\boldsymbol{\theta}}$ over $\mathcal{X}$
and maximises the **expected fitness**:

$$J(\boldsymbol{\theta}) = \mathbb{E}_{\mathbf{x} \sim p_{\boldsymbol{\theta}}}[f(\mathbf{x})]
= \int_{\mathcal{X}} f(\mathbf{x})\, p_{\boldsymbol{\theta}}(\mathbf{x})\, d\mathbf{x}$$

$J(\boldsymbol{\theta})$ is a **smoothed, continuous, everywhere differentiable**
surrogate for $f$, even when $f$ itself is neither [5].  Maximising $J(\boldsymbol{\theta})$
with respect to $\boldsymbol{\theta}$ concentrates probability mass near the true
optimum $\mathbf{x}^*$.

### 2.2 Isotropic Gaussian Search Distribution

Throughout this document we use the **isotropic Gaussian**:

$$p_{\boldsymbol{\theta}}(\mathbf{x}) = \mathcal{N}(\mathbf{x};\, \boldsymbol{\mu},\, \sigma^2 \mathbf{I})$$

where $\boldsymbol{\theta} = \boldsymbol{\mu} \in \mathbb{R}^d$ is the only
optimisable parameter (the **mean** or centre of the search distribution),
and $\sigma > 0$ is a fixed or slowly decayed **exploration spread**.

The goal reduces to: **find the $\boldsymbol{\mu}$ that maximises $J(\boldsymbol{\mu})$.**

---

## 3. Evolution Strategies: Historical Background

Evolution Strategies were introduced independently by **Rechenberg** (1965, published
1973 [6]) and **Schwefel** (1965, published 1977 [7]) at the Technical University
of Berlin.  The original formulation used a **$(\mu, \lambda)$-ES**:

1. Generate $\lambda$ offspring from $\mu$ parents by adding Gaussian noise.
2. Rank offspring by fitness.
3. Select the $\rho$ best as the next generation's parents.
4. Repeat.

The self-adaptation of the step size $\sigma$ was later formalised by **Hansen and
Ostermeier** in the Covariance Matrix Adaptation ES (CMA-ES) [8], which adapts a
full covariance matrix.

The modern **differentiable ES / Natural ES** framing due to **Wierstra et al.**
[2, 3] reinterprets the fitness-weighted selection step as a gradient ascent step
on $J(\boldsymbol{\theta})$, enabling the use of any first-order optimiser.

---

## 4. The Score-Function Gradient Estimator

### 4.1 Derivation (Log-Derivative Trick)

To perform gradient ascent on $J(\boldsymbol{\mu})$, we need $\nabla_{\boldsymbol{\mu}} J$.

$$\nabla_{\boldsymbol{\mu}} J(\boldsymbol{\mu})
= \nabla_{\boldsymbol{\mu}} \int f(\mathbf{x})\, p_{\boldsymbol{\mu}}(\mathbf{x})\, d\mathbf{x}$$

Under mild regularity conditions (Lebesgue dominated convergence), we may
exchange gradient and integral:

$$= \int f(\mathbf{x})\, \nabla_{\boldsymbol{\mu}} p_{\boldsymbol{\mu}}(\mathbf{x})\, d\mathbf{x}$$

Applying the **log-derivative (score function) identity**:

$$\nabla_{\boldsymbol{\mu}} p_{\boldsymbol{\mu}}(\mathbf{x})
= p_{\boldsymbol{\mu}}(\mathbf{x})\, \nabla_{\boldsymbol{\mu}} \log p_{\boldsymbol{\mu}}(\mathbf{x})$$

we obtain:

$$\boxed{
\nabla_{\boldsymbol{\mu}} J(\boldsymbol{\mu})
= \mathbb{E}_{\mathbf{x} \sim p_{\boldsymbol{\mu}}}
  \!\left[f(\mathbf{x})\cdot \nabla_{\boldsymbol{\mu}} \log p_{\boldsymbol{\mu}}(\mathbf{x})\right]
}$$

This is the **score-function estimator**, also known as the **REINFORCE gradient**
in the reinforcement learning literature [9], and the **natural gradient** direction
in natural evolution strategies [2, 3].

**Key property:** the right-hand side is an expectation that can be estimated by
Monte Carlo sampling — requiring *only function evaluations*, not analytic gradients.

### 4.2 Closed Form for the Isotropic Gaussian

For $p_{\boldsymbol{\mu}}(\mathbf{x}) = \mathcal{N}(\mathbf{x};\boldsymbol{\mu}, \sigma^2 \mathbf{I})$:

$$\log p_{\boldsymbol{\mu}}(\mathbf{x})
= -\frac{d}{2}\log(2\pi\sigma^2)
  - \frac{\|\mathbf{x} - \boldsymbol{\mu}\|^2}{2\sigma^2}$$

$$\nabla_{\boldsymbol{\mu}} \log p_{\boldsymbol{\mu}}(\mathbf{x})
= \frac{\mathbf{x} - \boldsymbol{\mu}}{\sigma^2}
= \frac{\boldsymbol{\varepsilon}}{\sigma}$$

where $\mathbf{x} = \boldsymbol{\mu} + \sigma\boldsymbol{\varepsilon}$,
$\;\boldsymbol{\varepsilon} \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$.

Substituting:

$$\nabla_{\boldsymbol{\mu}} J(\boldsymbol{\mu})
= \frac{1}{\sigma}\,\mathbb{E}_{\boldsymbol{\varepsilon} \sim \mathcal{N}(\mathbf{0},\mathbf{I})}
  \!\left[f\!\left(\boldsymbol{\mu} + \sigma\boldsymbol{\varepsilon}\right) \cdot \boldsymbol{\varepsilon}\right]$$

### 4.3 Monte Carlo Estimator

Using $H$ independent samples $\{\boldsymbol{\varepsilon}_i\}_{i=1}^{H}$:

$$\hat{\mathbf{g}}_H = \frac{1}{H\sigma} \sum_{i=1}^{H}
  f\!\left(\boldsymbol{\mu} + \sigma\boldsymbol{\varepsilon}_i\right)\boldsymbol{\varepsilon}_i$$

This estimator is **unbiased**: $\mathbb{E}[\hat{\mathbf{g}}_H] = \nabla_{\boldsymbol{\mu}} J(\boldsymbol{\mu})$.

**Variance** of this estimator:

$$\mathrm{Var}[\hat{\mathbf{g}}_H]
= \frac{1}{H\sigma^2}\,\mathrm{Var}_{\boldsymbol{\varepsilon}}\!\left[f(\boldsymbol{\mu}+\sigma\boldsymbol{\varepsilon})\,\boldsymbol{\varepsilon}\right]
= O\!\left(\frac{1}{H}\right)$$

The variance scales as $1/H$, motivating the use of as many samples as the
evaluation budget allows — but balanced against the cost per evaluation.

---

## 5. Antithetic Sampling: Variance Reduction

### 5.1 The Problem with One-Sided Sampling

The estimator $\hat{\mathbf{g}}_H$ has variance $O(1/H)$.  For finite $H$, individual
estimates can be very noisy, especially when $f$ is multimodal or when $\sigma$ is
small relative to the scale of features.

### 5.2 Antithetic Pairs

The **antithetic variates** method [10] exploits the symmetry of the Gaussian by
evaluating **symmetric pairs** $(\boldsymbol{\varepsilon}_i, -\boldsymbol{\varepsilon}_i)$:

$$\hat{\mathbf{g}}_{\mathrm{anti}}
= \frac{1}{2H\sigma} \sum_{i=1}^{H}
  \left[f(\boldsymbol{\mu} + \sigma\boldsymbol{\varepsilon}_i)
       - f(\boldsymbol{\mu} - \sigma\boldsymbol{\varepsilon}_i)\right] \boldsymbol{\varepsilon}_i$$

### 5.3 Proof of Unbiasedness

Since $\boldsymbol{\varepsilon}_i \overset{d}{=} -\boldsymbol{\varepsilon}_i$ (symmetric Gaussian):

$$\mathbb{E}\!\left[\hat{\mathbf{g}}_{\mathrm{anti}}\right]
= \frac{1}{2\sigma}\,\mathbb{E}\!\left[f(\boldsymbol{\mu}+\sigma\boldsymbol{\varepsilon})\boldsymbol{\varepsilon}\right]
+ \frac{1}{2\sigma}\,\mathbb{E}\!\left[-f(\boldsymbol{\mu}-\sigma\boldsymbol{\varepsilon})\boldsymbol{\varepsilon}\right]$$

Because $(-\boldsymbol{\varepsilon})$ has the same distribution as $\boldsymbol{\varepsilon}$:

$$= \frac{1}{2\sigma}\,\mathbb{E}\!\left[f(\boldsymbol{\mu}+\sigma\boldsymbol{\varepsilon})\boldsymbol{\varepsilon}\right]
+ \frac{1}{2\sigma}\,\mathbb{E}\!\left[f(\boldsymbol{\mu}+\sigma\boldsymbol{\varepsilon})\boldsymbol{\varepsilon}\right]
= \frac{1}{\sigma}\,\mathbb{E}\!\left[f(\boldsymbol{\mu}+\sigma\boldsymbol{\varepsilon})\boldsymbol{\varepsilon}\right]
= \nabla_{\boldsymbol{\mu}} J(\boldsymbol{\mu}) \checkmark$$

### 5.4 Variance Reduction Analysis

Let $A_i = f(\boldsymbol{\mu}+\sigma\boldsymbol{\varepsilon}_i)\varepsilon_i$
and $B_i = -f(\boldsymbol{\mu}-\sigma\boldsymbol{\varepsilon}_i)\varepsilon_i$.
The antithetic pair uses $\frac{A_i + B_i}{2}$ instead of $A_i$.

Since $\mathrm{Cov}(A_i, B_i) < 0$ for typical smooth $f$
(when $\boldsymbol{\mu}+\sigma\boldsymbol{\varepsilon}$ gives high $f$,
$\boldsymbol{\mu}-\sigma\boldsymbol{\varepsilon}$ tends to give lower $f$):

$$\mathrm{Var}\!\left[\frac{A_i+B_i}{2}\right]
= \frac{\mathrm{Var}[A_i] + \mathrm{Var}[B_i] + 2\,\mathrm{Cov}(A_i,B_i)}{4}
< \frac{\mathrm{Var}[A_i]}{2}$$

**Variance is at least halved** compared to one-sided sampling, using the same
number of perturbation directions $H$ (though $2H$ total evaluations).

This result was demonstrated empirically in the ES context by Salimans et al. [1]
who showed that antithetic sampling roughly doubles effective sample efficiency.

---

## 6. The Adam Optimiser

### 6.1 Motivation

Applying the raw gradient estimate $\hat{\mathbf{g}}$ directly in a fixed-step-size
update $\boldsymbol{\mu} \leftarrow \boldsymbol{\mu} + \alpha\hat{\mathbf{g}}$
suffers from:

- **Sensitivity to $\alpha$:** too large → oscillation; too small → slow convergence
- **Non-stationarity:** the optimal step size changes as $\boldsymbol{\mu}$ approaches the optimum
- **Noise amplification:** gradient estimates from a small batch of perturbations are noisy

**Adam** (Adaptive Moment Estimation, Kingma and Ba 2015 [4]) addresses all three
by maintaining per-parameter running statistics of the gradient.

### 6.2 Derivation from Exponential Moving Averages

Let $\hat{\mathbf{g}}_t$ be the gradient estimate at step $t$.  Define:

**First moment (mean of gradient — "momentum"):**
$$\mathbf{m}_t = \beta_1 \mathbf{m}_{t-1} + (1-\beta_1)\hat{\mathbf{g}}_t, \qquad \mathbf{m}_0 = \mathbf{0}$$

**Second moment (uncentred variance of gradient):**
$$\mathbf{v}_t = \beta_2 \mathbf{v}_{t-1} + (1-\beta_2)\hat{\mathbf{g}}_t^{\,2}, \qquad \mathbf{v}_0 = \mathbf{0}$$

where $\beta_1, \beta_2 \in [0,1)$ are decay rates and squaring is elementwise.

### 6.3 Bias Correction

Unrolling the recursion from $\mathbf{m}_0 = \mathbf{0}$:

$$\mathbf{m}_t = (1-\beta_1)\sum_{i=1}^{t} \beta_1^{t-i}\hat{\mathbf{g}}_i$$

Taking expectations (assuming stationarity, $\mathbb{E}[\hat{\mathbf{g}}_i] = \mathbf{g}$):

$$\mathbb{E}[\mathbf{m}_t] = \mathbf{g}\,(1-\beta_1)\sum_{i=1}^{t}\beta_1^{t-i}
= \mathbf{g}\,(1-\beta_1^t)$$

The estimator $\mathbf{m}_t$ is biased toward zero by a factor $(1-\beta_1^t)$.
We correct this with:

$$\hat{\mathbf{m}}_t = \frac{\mathbf{m}_t}{1 - \beta_1^t}, \quad
\hat{\mathbf{v}}_t = \frac{\mathbf{v}_t}{1 - \beta_2^t}$$

so that $\mathbb{E}[\hat{\mathbf{m}}_t] \approx \mathbf{g}$ and
$\mathbb{E}[\hat{\mathbf{v}}_t] \approx \mathbb{E}[\mathbf{g}^2]$.

### 6.4 The Update Rule

$$\boxed{
\boldsymbol{\mu}_{t+1}
= \boldsymbol{\mu}_t + \alpha \cdot \frac{\hat{\mathbf{m}}_t}{\sqrt{\hat{\mathbf{v}}_t} + \varepsilon}
}$$

where $\varepsilon \sim 10^{-8}$ prevents division by zero.

**Interpretation of each term:**

| Term | Role |
|---|---|
| $\hat{\mathbf{m}}_t$ | Smoothed gradient direction (reduces noise, provides momentum) |
| $\sqrt{\hat{\mathbf{v}}_t}$ | Running RMS of gradient magnitude (normalises the step size) |
| $\alpha / \sqrt{\hat{\mathbf{v}}_t}$ | Per-parameter adaptive learning rate |

**Effect:** coordinates with consistently large gradients get a smaller effective
learning rate; coordinates with small or noisy gradients get a larger relative rate.
This is crucial in high-dimensional problems where the gradient varies greatly across
dimensions.

### 6.5 Standard Hyperparameters

| Parameter | Typical value | Role |
|---|---|---|
| $\alpha$ | 0.001 – 0.05 | Global learning rate |
| $\beta_1$ | 0.9 | Momentum decay (controls memory of past gradients) |
| $\beta_2$ | 0.999 | RMS decay (controls memory of past squared gradients) |
| $\varepsilon$ | $10^{-8}$ | Numerical stability |

These defaults are theoretically and empirically verified in [4].  The learning rate
$\alpha$ is the primary hyperparameter requiring tuning.

---

## 7. The Complete ES-RL Algorithm

### 7.1 State Variables

| Variable | Dimension | Description |
|---|---|---|
| $\boldsymbol{\mu}$ | $d$ | Current mean of the search distribution |
| $\sigma$ | scalar | Fixed exploration spread |
| $\mathbf{m}$ | $d$ | Adam first-moment accumulator |
| $\mathbf{v}$ | $d$ | Adam second-moment accumulator |
| $t$ | integer | Adam timestep counter |

### 7.2 One Iteration (Step)

**Input:** current $\boldsymbol{\mu}$, $\sigma$, $H$, $\alpha$, $\beta_1$, $\beta_2$, $\varepsilon$

**Step 1 — Sample perturbations:**
$$\boldsymbol{\varepsilon}_1, \ldots, \boldsymbol{\varepsilon}_H \overset{i.i.d.}{\sim} \mathcal{N}(\mathbf{0}, \mathbf{I}_d)$$

**Step 2 — Evaluate antithetic pairs:**
$$f^+_i = f(\boldsymbol{\mu} + \sigma\boldsymbol{\varepsilon}_i), \quad
f^-_i = f(\boldsymbol{\mu} - \sigma\boldsymbol{\varepsilon}_i), \qquad i = 1,\ldots,H$$

*Total function evaluations this step:* $2H$

**Step 3 — Gradient estimate:**
$$\hat{\mathbf{g}} = \frac{1}{2H\sigma}\sum_{i=1}^{H}(f^+_i - f^-_i)\boldsymbol{\varepsilon}_i$$

**Step 4 — Adam moment update:**
$$t \leftarrow t + 1, \quad
\mathbf{m} \leftarrow \beta_1\mathbf{m} + (1-\beta_1)\hat{\mathbf{g}}, \quad
\mathbf{v} \leftarrow \beta_2\mathbf{v} + (1-\beta_2)\hat{\mathbf{g}}^{\,2}$$

**Step 5 — Bias correction:**
$$\hat{\mathbf{m}} = \frac{\mathbf{m}}{1-\beta_1^t}, \quad
\hat{\mathbf{v}} = \frac{\mathbf{v}}{1-\beta_2^t}$$

**Step 6 — Update mean:**
$$\boldsymbol{\mu} \leftarrow \mathrm{clip}\!\left(
\boldsymbol{\mu} + \alpha\cdot\frac{\hat{\mathbf{m}}}{\sqrt{\hat{\mathbf{v}}}+\varepsilon},
\; \mathbf{x}_{\min}, \; \mathbf{x}_{\max}\right)$$

**Output:** updated $\boldsymbol{\mu}$, best $\mathbf{x}$ seen so far

### 7.3 Full Pseudocode

```
Algorithm: ES-RL with Antithetic Sampling and Adam
═══════════════════════════════════════════════════

Require: f, domain [x_min, x_max], σ, H, α, β₁=0.9, β₂=0.999, ε=1e-8, budget N

  # Initialise
  μ  ← random point in [x_min, x_max]         # start anywhere
  m  ← 0,  v ← 0,  t ← 0                      # Adam state
  best_f ← -∞,  best_x ← None

  # Main loop
  While evaluations < N:

    # 1. Sample H perturbation directions
    {εᵢ}ᵢ₌₁ᴴ  ←  i.i.d. N(0, I_d)

    # 2. Antithetic evaluation
    For i = 1, …, H:
        f⁺ᵢ ← f( clip(μ + σεᵢ, x_min, x_max) )    // +1 eval
        f⁻ᵢ ← f( clip(μ - σεᵢ, x_min, x_max) )    // +1 eval
        Update best_f if f⁺ᵢ or f⁻ᵢ exceeds current best

    # 3. Gradient estimate
    ĝ ← (1 / 2Hσ) Σᵢ (f⁺ᵢ - f⁻ᵢ) εᵢ

    # 4. Adam update
    t ← t + 1
    m ← β₁ m + (1-β₁) ĝ
    v ← β₂ v + (1-β₂) ĝ²
    m̂ ← m / (1 - β₁ᵗ)
    v̂ ← v / (1 - β₂ᵗ)
    μ ← clip( μ + α · m̂ / (√v̂ + ε),  x_min,  x_max )

  Return best_x, best_f
```

---

## 8. Convergence Analysis

### 8.1 Relationship to Gaussian Smoothing

The expected fitness $J(\boldsymbol{\mu}) = \mathbb{E}_{\mathbf{x}\sim\mathcal{N}(\boldsymbol{\mu},\sigma^2\mathbf{I})}[f(\mathbf{x})]$
is the **Gaussian-smoothed** surrogate.  By convolution:

$$J(\boldsymbol{\mu}) = (f * \phi_\sigma)(\boldsymbol{\mu}), \qquad
\phi_\sigma(\mathbf{x}) = \mathcal{N}(\mathbf{x};\mathbf{0},\sigma^2\mathbf{I})$$

Nesterov and Spokoiny [5] showed that $J$ inherits Lipschitz smoothness from $f$:
if $f$ is $L$-Lipschitz, then $J$ is $L$-Lipschitz and its gradient is
$(L/\sigma)$-Lipschitz.  This guarantees that gradient ascent on $J$ converges.

### 8.2 Convergence Rate for Convex $J$

For $J$ that is $\mu$-strongly concave (near the maximum), projected gradient
ascent with step $\alpha$ converges geometrically [11]:

$$J(\boldsymbol{\mu}^*) - J(\boldsymbol{\mu}_t) \leq (1 - \alpha\mu)^t \,[J(\boldsymbol{\mu}^*) - J(\boldsymbol{\mu}_0)]$$

In practice $J$ is not globally concave (it inherits the multimodality of $f$),
but convergence to a **local maximum** of $J$ is guaranteed under standard conditions.

### 8.3 Approximation Error from Finite $H$

With $H$ antithetic pairs, the gradient estimate has variance $O(1/H)$.
By the convergence theory of stochastic gradient ascent in the non-convex setting [12]:

$$\frac{1}{T}\sum_{t=1}^{T}\mathbb{E}\!\left[\|\nabla J(\boldsymbol{\mu}_t)\|^2\right]
= O\!\left(\frac{1}{\sqrt{T}} + \frac{\sigma_\mathrm{grad}}{\sqrt{H}}\right)$$

where $\sigma_\mathrm{grad}$ is the variance of a single gradient sample.
This shows that more antithetic pairs (larger $H$) tighten the convergence guarantee.

### 8.4 Bias from Finiteness of $\sigma$

The gradient of $J$ approximates the gradient of $f$ to order $O(\sigma^2)$:

$$\nabla_{\boldsymbol{\mu}} J(\boldsymbol{\mu}) = \nabla f(\boldsymbol{\mu}) + O(\sigma^2)$$

(when $f$ is twice differentiable, via Taylor expansion [1]).  Therefore:
- **Small $\sigma$:** nearly unbiased, but high variance and inability to escape local maxima.
- **Large $\sigma$:** low variance, strong regularisation, but may miss narrow peaks.

This tension is the fundamental trade-off in ES, analogous to exploration vs. exploitation.

---

## 9. Connections to Related Methods

### 9.1 REINFORCE (Williams, 1992)

Williams [9] derived the same score-function estimator in the context of reinforcement
learning, where $f(\mathbf{x})$ is a **return** and $p_{\boldsymbol{\theta}}(\mathbf{x})$
is a **stochastic policy**.  ES-RL is structurally identical: the policy is the Gaussian
search distribution and the "reward" is the objective function.

$$\hat{\mathbf{g}}_{\mathrm{REINFORCE}} = \frac{1}{H}\sum_{i=1}^{H}
R(\tau_i)\,\nabla_{\boldsymbol{\theta}}\log\pi_{\boldsymbol{\theta}}(\tau_i)$$

Substituting $R = f$, $\pi_{\boldsymbol{\theta}} = p_{\boldsymbol{\mu}}$,
$\tau_i = \mathbf{x}_i$ recovers the ES gradient estimator exactly.

### 9.2 Natural Evolution Strategies (NES)

Wierstra et al. [2, 3] proposed updating $\boldsymbol{\mu}$ along the **natural gradient**:

$$\boldsymbol{\mu}_{t+1} = \boldsymbol{\mu}_t + \alpha\,\mathbf{F}^{-1}(\boldsymbol{\mu}_t)\,\hat{\mathbf{g}}$$

where $\mathbf{F}(\boldsymbol{\mu}) = \mathbb{E}[(\nabla_{\boldsymbol{\mu}}\log p_{\boldsymbol{\mu}})(\nabla_{\boldsymbol{\mu}}\log p_{\boldsymbol{\mu}})^\top]$
is the Fisher information matrix.

For the isotropic Gaussian, $\mathbf{F} = \sigma^{-2}\mathbf{I}$, so the natural
gradient update reduces to:

$$\boldsymbol{\mu}_{t+1} = \boldsymbol{\mu}_t + \frac{\alpha}{\sigma^2}\,\hat{\mathbf{g}} \cdot \sigma
= \boldsymbol{\mu}_t + \frac{\alpha}{\sigma}\,\hat{\mathbf{g}}_{\mathrm{normalised}}$$

Natural gradient removes the scale dependence on $\sigma$ that affects the vanilla estimator.

### 9.3 CMA-ES

Covariance Matrix Adaptation ES [8] extends the search distribution to a full
multivariate Gaussian $\mathcal{N}(\boldsymbol{\mu}, \mathbf{C})$, adapting the
covariance matrix $\mathbf{C}$ to align with the local curvature of $f$.
This gives CMA-ES second-order-like convergence near the optimum but at a cost
of $O(d^2)$ per step.  ES-RL with an isotropic Gaussian is a $O(d)$-per-step
first-order version, trading some convergence speed for scalability.

---

## 10. Summary and Practical Guidance

### 10.1 Algorithm Summary

$$\boxed{
\text{ES-RL}: \quad
\boldsymbol{\mu}_{t+1}
= \mathrm{clip}\!\left(
  \boldsymbol{\mu}_t
  + \alpha\cdot\frac{\hat{\mathbf{m}}_t}{\sqrt{\hat{\mathbf{v}}_t}+\varepsilon},\;
  \mathbf{x}_{\min},\; \mathbf{x}_{\max}\right)
}$$

where $\hat{\mathbf{m}}_t$ and $\hat{\mathbf{v}}_t$ are the bias-corrected Adam
moments of the antithetic gradient estimator.

### 10.2 Hyperparameter Guidance

| Hyperparameter | Recommended | Effect of increasing |
|---|---|---|
| $\sigma$ (spread) | 0.05 – 0.20 (fraction of domain) | Better exploration, more bias |
| $H$ (pairs per step) | 5 – 20 | Lower variance, more evals per step |
| $\alpha$ (learning rate) | 0.01 – 0.10 | Faster but less stable convergence |
| $\beta_1$ (momentum) | 0.9 (fixed) | Smoother steps |
| $\beta_2$ (RMS) | 0.999 (fixed) | More stable norm estimation |

### 10.3 When ES-RL Outperforms Random Search

ES-RL is advantageous when:
- The landscape has a **clear basin** around the optimum (the gradient points there)
- The **evaluation budget is limited** relative to the domain size
- Dimensionality $d > 1$ (random search scales poorly: needs $O((1/p_\varepsilon)^{1/d})$ samples)

Random search remains competitive or superior when:
- The landscape is **pathological** (no gradient information useful, e.g., needle-in-haystack)
- The domain is **1-dimensional** with a small budget
- A **diverse exploration** (rather than fast convergence) is the primary goal

---

## 11. References

**[1]** T. Salimans, J. Ho, X. Chen, S. Sidor, and I. Sutskever,
*"Evolution Strategies as a Scalable Alternative to Reinforcement Learning,"*
arXiv:1703.03864, OpenAI (2017).
→ Primary reference for the modern ES-RL framework, antithetic sampling, and parallelism.

**[2]** D. Wierstra, T. Forster, J. Peters, and J. Schmidhuber,
*"Natural Evolution Strategies,"*
Proceedings of the IEEE Congress on Evolutionary Computation (CEC), 2008.
→ First formulation of ES as gradient ascent on expected fitness (natural gradient).

**[3]** D. Wierstra, T. Forster, J. Peters, and J. Schmidhuber,
*"Natural Evolution Strategies,"*
Journal of Machine Learning Research (JMLR) **15**, 949–980, 2014.
→ Extended journal version with convergence analysis and natural gradient derivation.

**[4]** D. P. Kingma and J. Ba,
*"Adam: A Method for Stochastic Optimization,"*
International Conference on Learning Representations (ICLR), 2015.
→ Original Adam paper; derivation of bias-corrected moment estimates.

**[5]** Y. Nesterov and V. Spokoiny,
*"Random Gradient-Free Minimization of Convex Functions,"*
Foundations of Computational Mathematics **17**(2), 527–566, 2017.
→ Theoretical basis for Gaussian smoothing and gradient approximation in black-box optimisation.

**[6]** I. Rechenberg,
*Evolutionsstrategie: Optimierung technischer Systeme nach Prinzipien der biologischen Evolution,*
Friedrich Frommann Verlag, Stuttgart, 1973.
→ Original doctoral thesis introducing Evolution Strategies.

**[7]** H.-P. Schwefel,
*Numerische Optimierung von Computer-Modellen mittels der Evolutionsstrategie,*
Birkhäuser, Basel, 1977.
→ Independent founding work on Evolution Strategies.

**[8]** N. Hansen and A. Ostermeier,
*"Completely Derandomized Self-Adaptation in Evolution Strategies,"*
Evolutionary Computation **9**(2), 159–195, 2001.
→ CMA-ES; full covariance adaptation for second-order ES.

**[9]** R. J. Williams,
*"Simple Statistical Gradient-Following Algorithms for Connectionist Reinforcement Learning,"*
Machine Learning **8**(3–4), 229–256, 1992.
→ Original REINFORCE algorithm; score-function gradient estimator in RL.

**[10]** P. Bratley, B. L. Fox, and L. E. Schrage,
*A Guide to Simulation,* 2nd ed., Springer, 1987.
→ Antithetic variates as a variance reduction technique for Monte Carlo estimation.

**[11]** S. Bubeck,
*"Convex Optimization: Algorithms and Complexity,"*
Foundations and Trends in Machine Learning **8**(3–4), 231–357, 2015.
→ Convergence theorems for gradient ascent on (strongly) concave objectives.

**[12]** L. Bottou, F. E. Curtis, and J. Nocedal,
*"Optimization Methods for Large-Scale Machine Learning,"*
SIAM Review **60**(2), 223–311, 2018.
→ Stochastic gradient convergence under non-convexity; O(1/√T) rate.

**[13]** R. S. Sutton and A. G. Barto,
*Reinforcement Learning: An Introduction,* 2nd ed., MIT Press, 2018. Chapter 13.
→ Policy gradient methods; REINFORCE; baseline variance reduction.

**[14]** N. Hansen,
*"The CMA Evolution Strategy: A Tutorial,"*
arXiv:1604.00772, 2016.
→ Comprehensive CMA-ES tutorial; step-size adaptation; comparison with (1+1)-ES.

**[15]** A. Graves,
*"Generating Sequences with Recurrent Neural Networks,"*
arXiv:1308.0850, 2013.
→ Adam used in a recurrent context; practical evidence for β₁=0.9, β₂=0.999.
