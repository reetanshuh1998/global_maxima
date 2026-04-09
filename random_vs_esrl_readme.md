# Random Search vs ES-RL: How These Algorithms Work

## Overview

This document explains — with full mathematical detail — how two
optimization algorithms find the maximum of a complex function:

1. **Random Search** — the conceptually simplest possible optimizer
2. **ES-RL** — Evolution Strategy with antithetic sampling and Adam update

The demo code (`codes/demo_random_vs_esrl.py`) applies both to the
single-variable test function:

$$f(x) = 2\sin(2.3x) + 1.5\sin(4.1x+0.7) + \cos(1.3x) + 0.6\sin(7.7x+1.2) - 0.05(x-5)^2 + 3$$

on the domain $x \in [0, 10]$.  This function has **many local maxima**
of similar height, making it a meaningful test for any optimizer.

---

## 1. The Test Function — Why Is It Hard?

The function is a sum of sinusoids with **incommensurate frequencies**
(2.3, 4.1, 1.3, 7.7), so the peaks do not repeat periodically.
A gradient-following method starting at a random point will almost
certainly land on a local maximum that is not the global one.

The mild quadratic tilt $-0.05(x-5)^2$ breaks perfect symmetry and
shifts the global maximum slightly away from the center, making it
non-obvious from the function shape alone.

As a result:
- Any method that only evaluates a few points is likely to miss the global max.
- A method that *learns* which region is promising does better than pure luck.

---

## 2. Random Search

### 2.1 Algorithm (conceptual)

```
Initialize: best_f = -∞,  best_x = None
For i = 1, 2, …, N:
    Draw  x_i ~ Uniform[a, b]
    Evaluate  v = f(x_i)
    If v > best_f:
        best_f ← v,  best_x ← x_i
Return best_x, best_f
```

That is all.  Random search has no memory, no learning, and no sense
of direction.  Every evaluation is completely independent of all previous ones.

### 2.2 Mathematical Analysis

**Probability of finding a point within ε of the global maximum f*:**

Let $\mathcal{X}_\varepsilon$ be the set of $x$ values where $f(x) \geq f^* - \varepsilon$.
Define its measure (relative length):

$$p_\varepsilon = \frac{|\mathcal{X}_\varepsilon|}{b - a}$$

After $N$ independent uniform draws, the probability that **none** of
them land in $\mathcal{X}_\varepsilon$ is $(1 - p_\varepsilon)^N$.
So the probability of **success** (at least one draw in $\mathcal{X}_\varepsilon$) is:

$$\boxed{P(\text{success after } N \text{ evals}) = 1 - (1 - p_\varepsilon)^N}$$

For small $p_\varepsilon$ this is approximately $Np_\varepsilon$ for small $N$,
and approaches 1 as $N \to \infty$ (guaranteed convergence, eventually).

**Expected value of the best sample:**

For the uniform distribution on $[a,b]$ and a continuous $f$,
the expected gap to the optimum after $N$ samples is:

$$\mathbb{E}[f^* - \max_i f(x_i)] = O\!\left(\frac{1}{N+1}\right)$$

This $O(1/N)$ convergence rate is **very slow** for functions with
a narrow optimal region — which is why random search needs many more
evaluations than gradient-guided methods.

### 2.3 Strengths and Weaknesses

| Strength | Weakness |
|---|---|
| Zero hyperparameters | Ignores all structure in f |
| Never gets stuck in local optima | O(1/N) convergence — very slow |
| Embarrassingly parallel | Becomes impractical in high dimensions |
| Provides an unbiased coverage baseline | Cannot exploit a promising region |

---

## 3. ES-RL (Evolution Strategy + Adam)

### 3.1 Core Idea

Instead of sampling uniformly, ES-RL maintains a **probability
distribution** over the search space — a Gaussian with mean $\mu$ and
spread $\sigma$ — and **updates the mean** to move toward higher function values.

The key insight: even without computing derivatives of $f$ analytically,
we can **estimate a gradient** of the expected performance
$J(\mu) = \mathbb{E}_{x \sim \mathcal{N}(\mu, \sigma^2)}[f(x)]$
using function evaluations alone.

### 3.2 The Gradient Estimate (REINFORCE / ES trick)

We want to move $\mu$ in the direction that increases $J(\mu)$.
The gradient is:

$$\nabla_\mu J(\mu)
= \nabla_\mu \int f(x) \, p_\mu(x) \, dx
= \int f(x) \nabla_\mu \log p_\mu(x) \, dx
= \mathbb{E}_{x \sim p_\mu}\!\left[f(x) \cdot \nabla_\mu \log p_\mu(x)\right]$$

For a Gaussian $p_\mu(x) = \mathcal{N}(x;\,\mu,\sigma^2)$:

$$\nabla_\mu \log p_\mu(x) = \frac{x - \mu}{\sigma^2}$$

So the gradient estimate using $H$ random samples $\varepsilon_i \sim \mathcal{N}(0,1)$,
i.e., $x_i = \mu + \sigma\varepsilon_i$, is:

$$\hat{g} = \frac{1}{H\sigma} \sum_{i=1}^{H} f(\mu + \sigma\varepsilon_i) \cdot \varepsilon_i$$

### 3.3 Antithetic Sampling (Variance Reduction)

Instead of sampling $H$ independent $\varepsilon_i$, we evaluate **symmetric pairs**:

$$x^+_i = \mu + \sigma\varepsilon_i, \qquad x^-_i = \mu - \sigma\varepsilon_i$$

The improved gradient estimate is:

$$\boxed{\hat{g} = \frac{1}{2H\sigma} \sum_{i=1}^{H} \bigl[f(x^+_i) - f(x^-_i)\bigr] \cdot \varepsilon_i}$$

**Why this helps:** The symmetric pairs are negatively correlated.
When the positive perturbation overshoots, the negative perturbation
undershoots by the same amount.  The variance of $\hat{g}$ is
roughly **halved** compared to one-sided sampling, at no extra cost
in the number of new directions tested.

### 3.4 Adam Optimizer (Adaptive Moment Estimation)

Raw gradient estimates are noisy.  Instead of directly updating
$\mu \leftarrow \mu + \alpha \hat{g}$, we use Adam — which adapts the
effective learning rate for each coordinate using running statistics
of the gradient:

**State variables:** first moment $m$, second moment $v$, timestep $t$

**Update rule (one step):**

$$m_t = \beta_1 m_{t-1} + (1 - \beta_1)\,\hat{g}_t$$

$$v_t = \beta_2 v_{t-1} + (1 - \beta_2)\,\hat{g}_t^{\,2}$$

**Bias correction** (removes initialisation bias from $m_0 = v_0 = 0$):

$$\hat{m}_t = \frac{m_t}{1 - \beta_1^t}, \qquad \hat{v}_t = \frac{v_t}{1 - \beta_2^t}$$

**Parameter update:**

$$\boxed{\mu_{t+1} = \text{clip}\!\left(\mu_t + \alpha \cdot \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \varepsilon},\; 0,\; 1\right)}$$

**Typical values:** $\alpha = 0.05$, $\beta_1 = 0.9$, $\beta_2 = 0.999$, $\varepsilon = 10^{-8}$.

**Why Adam helps:**
- $\hat{m}_t$ gives the **direction** with momentum (smooths out noise)
- $\hat{v}_t$ gives the **magnitude** of recent gradient fluctuation
- Dividing by $\sqrt{\hat{v}_t}$ normalises step size: large for infrequently
  updated directions, small for noisily updated ones

### 3.5 Full ES-RL Algorithm (Pseudocode)

```
Initialise:
    μ ← random point in [0, 1]     (normalised domain)
    σ ← 0.12                        (exploration spread)
    m, v ← 0,  t ← 0               (Adam state)

best_f ← -∞,  best_x ← None

While evaluations < budget:
    Sample ε₁, …, εH ~ N(0, 1)

    For i = 1, …, H:
        x⁺ = clip(μ + σ εᵢ, 0, 1)  →  x_real = a + x⁺(b-a);  evaluate f(x_real)
        x⁻ = clip(μ - σ εᵢ, 0, 1)  →  x_real = a + x⁻(b-a);  evaluate f(x_real)
        Update best_f, best_x if improved

    Gradient estimate:
        ĝ = mean_i( [f(x⁺ᵢ) - f(x⁻ᵢ)] · εᵢ ) / (2σ)

    Adam update:
        t ← t + 1
        m ← β₁·m + (1-β₁)·ĝ
        v ← β₂·v + (1-β₂)·ĝ²
        m̂ = m/(1-β₁ᵗ),   v̂ = v/(1-β₂ᵗ)
        μ ← clip(μ + α · m̂ / (√v̂ + ε),  0,  1)

Return best_x, best_f
```

### 3.6 Strengths and Weaknesses

| Strength | Weakness |
|---|---|
| Learns which region is promising | Has hyperparameters (σ, α, H) to tune |
| Converges faster than random in practice | Can converge prematurely if σ too small |
| Variance reduction via antithetic pairs | Gradient estimate still noisy for small H |
| Adam gives stable, adaptive step sizes | May miss narrow peaks far from initial μ |
| Works even when f is non-differentiable | Not guaranteed globally optimal |

---

## 4. Side-by-Side Comparison

| Property | Random Search | ES-RL |
|---|---|---|
| **Mechanism** | Pure uniform sampling | Directed gradient-estimated walk |
| **Memory** | None | μ, m, v (3 scalars in 1D) |
| **Convergence rate** | O(1/N) probabilistic | Super-linear near the optimum |
| **Multi-modal handling** | Good (global coverage) | Risk of local optimum if σ too small |
| **Evaluation efficiency** | Poor | Good |
| **Hyperparameters** | Zero | σ, α, β₁, β₂, H |
| **Parallelism** | Trivial | Easy (each pair independent) |
| **Best use case** | Initial exploration, baseline | Refinement once a good region found |

---

## 5. What the Plots Show

| Plot | What to look for |
|---|---|
| `demo_function.png` | The many local maxima — the true max is not at a visually obvious location |
| `demo_convergence.png` | ES-RL rises faster early and has lower variance; Random is slower but eventually catches up |
| `demo_search_paths.png` | Random: uniform scatter; ES-RL: dense cluster near the optimum after initial exploration |
| `demo_distribution.png` | ES-RL median closer to f* and lower spread; Random has wider distribution |

---

## 6. Key Takeaway

Random search is **unbiased** — it covers the whole space equally —
but it is **inefficient** because it never exploits what it has learned.

ES-RL is **adaptive** — it concentrates evaluations where $f$ is
largest — but it can get stuck if the initial mean is far from the
global optimum or if $\sigma$ is too small.

**In practice**, the two are complementary:
- Use random search (or a few diverse seeds) for **global exploration**
- Use ES-RL (or gradient-based methods) for **local refinement**

This is exactly the strategy used in `final_benchmark.py`:
the multi-start multi-seed framework combines global diversity
(different random starting points) with adaptive local search (ES-RL).
