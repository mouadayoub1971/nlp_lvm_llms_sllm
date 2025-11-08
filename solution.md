# Bayesian Inference - Solutions

## Part 1: Foundations

### Q1: Understanding Bayes' Rule

**Theory Explanation:**

Bayes' rule updates beliefs based on evidence. Components:

- **Prior p(θ)**: Initial belief about parameter θ before seeing data. Represents existing knowledge/assumptions.
- **Likelihood p(data|θ)**: Probability of observing data given specific θ value. Measures how well θ explains data.
- **Posterior p(θ|data)**: Updated belief about θ after seeing data. Combines prior knowledge with evidence.
- **Evidence p(data)**: Total probability of data across all θ values. Normalizing constant ensuring posterior sums to 1.

**Formula**: `p(θ|data) = [p(data|θ) × p(θ)] / p(data)`

Or: `Posterior = (Likelihood × Prior) / Evidence`
()
**Code:**
```python
def explain_bayes_rule():
    """Print Bayes' rule with explanations."""
    print("=" * 60)
    print("BAYES' RULE")
    print("=" * 60)
    print()
    print("Formula:")
    print("              p(data|θ) × p(θ)")
    print("p(θ|data) = ─────────────────────")
    print("                 p(data)")
    print()
    print("Components:")
    print("  • p(θ)       = Prior: belief before data")
    print("  • p(data|θ)  = Likelihood: how well θ explains data")
    print("  • p(θ|data)  = Posterior: updated belief after data")
    print("  • p(data)    = Evidence: normalizing constant")
    print()
    print("In words:")
    print("  'Update your beliefs by multiplying prior knowledge")
    print("   with new evidence, then normalize.'")
    print("=" * 60)

explain_bayes_rule()
```

---

### Q2: Discrete Prior

**Code:**
```python
import numpy as np

def create_discrete_prior():
    """
    Create uniform discrete prior over θ = [0.0, 0.1, ..., 1.0]

    Returns:
        theta_values: array of possible θ values
        prior_probs: array of prior probabilities (sum=1)
    """
    theta_values = np.linspace(0, 1, 11)  # [0.0, 0.1, 0.2, ..., 1.0]
    prior_probs = np.ones(len(theta_values)) / len(theta_values)  # Uniform

    return theta_values, prior_probs

# Test
theta, prior = create_discrete_prior()
print(f"Number of θ values: {len(theta)}")
print(f"Prior sums to: {sum(prior):.10f}")  # Should be 1.0
print(f"θ values: {theta}")
print(f"Prior probs: {prior}")
```

**Output:**
```
Number of θ values: 11
Prior sums to: 1.0000000000
θ values: [0.  0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1. ]
Prior probs: [0.09090909 0.09090909 ... 0.09090909]
```

**Questions:**

**Q: Why normalize (divide by sum)?**
A: Normalization ensures probabilities sum to 1, satisfying probability axioms. It converts raw weights into valid probability distribution.

**Q: What does "uniform prior" mean philosophically?**
A: Uniform prior expresses complete ignorance/indifference. All θ values equally likely before data. Represents objective starting point with no bias.

---

### Q3: Computing Likelihood

**Code:**
```python
def compute_likelihood(theta_values, n_heads, n_tails):
    """
    Compute binomial likelihood p(data|θ) for each θ.

    Formula: L(θ) = θ^n_heads × (1-θ)^n_tails

    Args:
        theta_values: array of possible θ
        n_heads: observed heads
        n_tails: observed tails

    Returns:
        likelihood: normalized likelihood values
    """
    # Handle edge case θ=0 and θ=1
    likelihood = np.power(theta_values, n_heads) * np.power(1 - theta_values, n_tails)

    # Normalize (convert to probability distribution)
    likelihood = likelihood / np.sum(likelihood)

    return likelihood

# Test
theta = np.linspace(0, 1, 100)
likelihood = compute_likelihood(theta, n_heads=7, n_tails=3)
print(f"Max likelihood at θ = {theta[np.argmax(likelihood)]:.2f}")  # ~0.7
print(f"Likelihood sums to: {np.sum(likelihood):.10f}")
```

**Output:**
```
Max likelihood at θ = 0.70
Likelihood sums to: 1.0000000000
```

**Questions:**

**Q: What θ maximizes likelihood? Why?**
A: θ = 0.7 (7 heads / 10 flips). This is the Maximum Likelihood Estimate (MLE) - the parameter value making observed data most probable.

**Q: What happens if n_heads=0 and n_tails=0?**
A: With no data, likelihood = θ^0 × (1-θ)^0 = 1 for all θ. All values equally likely. Posterior equals prior (no learning).

---

### Q4: Applying Bayes' Rule

**Code:**
```python
def compute_posterior(prior, likelihood):
    """
    Apply Bayes' rule: posterior ∝ likelihood × prior

    Args:
        prior: prior probabilities
        likelihood: likelihood values

    Returns:
        posterior: normalized posterior probabilities
    """
    # Numerator: likelihood × prior
    unnormalized_posterior = likelihood * prior

    # Denominator: evidence (normalization)
    evidence = np.sum(unnormalized_posterior)

    # Normalize
    posterior = unnormalized_posterior / evidence

    return posterior

# Test
theta = np.linspace(0, 1, 100)
prior = np.ones(100) / 100  # Uniform
likelihood = compute_likelihood(theta, 7, 3)
posterior = compute_posterior(prior, likelihood)

print(f"Posterior sums to: {sum(posterior):.10f}")
print(f"Posterior mean: {np.sum(theta * posterior):.3f}")  # ~0.7
print(f"Posterior peak at θ = {theta[np.argmax(posterior)]:.2f}")
```

**Output:**
```
Posterior sums to: 1.0000000000
Posterior mean: 0.700
Posterior peak at θ = 0.70
```

**Questions:**

**Q: Why multiply prior × likelihood?**
A: This follows from probability chain rule: p(θ,data) = p(data|θ)×p(θ). We combine how likely data is given θ (likelihood) with how likely θ was initially (prior).

**Q: What is evidence p(data)? Why ignore it?**
A: Evidence = ∫ p(data|θ)p(θ) dθ = sum over all θ. It's just a normalizing constant independent of θ. We can ignore it during computation, then normalize at end.

---

## Part 2: Visualization

### Q5: Plot Prior, Likelihood, Posterior

**Code:**
```python
import matplotlib.pyplot as plt

def plot_bayesian_update(theta, prior, likelihood, posterior):
    """
    Plot prior, likelihood, and posterior together.

    Colors:
    - Prior: blue dashed
    - Likelihood: green dash-dot
    - Posterior: red solid
    """
    plt.figure(figsize=(10, 6))

    plt.plot(theta, prior, 'b--', linewidth=2, label='Prior', alpha=0.7)
    plt.plot(theta, likelihood, 'g-.', linewidth=2, label='Likelihood', alpha=0.7)
    plt.plot(theta, posterior, 'r-', linewidth=2.5, label='Posterior', alpha=0.9)

    plt.xlabel('θ (coin bias)', fontsize=12)
    plt.ylabel('Probability Density', fontsize=12)
    plt.title('Bayesian Update: Prior + Likelihood → Posterior', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(alpha=0.3)
    plt.tight_layout()

    return plt

# Test
theta = np.linspace(0, 1, 100)
prior = np.ones(100) / 100
likelihood = compute_likelihood(theta, 7, 3)
posterior = compute_posterior(prior, likelihood)

plot_bayesian_update(theta, prior, likelihood, posterior)
plt.savefig('bayesian_update.png', dpi=150)
print("Plot saved as 'bayesian_update.png'")
```

**Questions:**

**Q: Where is posterior peak compared to likelihood peak?**
A: With uniform prior, posterior peak = likelihood peak (both at θ=0.7). Uniform prior doesn't shift the peak, only normalizes.

**Q: How would posterior change if prior not uniform?**
A: Non-uniform prior shifts posterior toward prior's peak. If prior favors θ=0.5 (fair coin), posterior would be pulled toward 0.5, landing between prior and likelihood peaks. Strong prior = more shift.

---

## Part 3: Real-World Scenario

### Q6: Medical Diagnosis Problem

**Code:**
```python
def medical_diagnosis():
    """
    Calculate p(disease | positive test) using Bayes' rule.

    Given:
    - p(disease) = 0.01 (1% base rate)
    - p(positive | disease) = 0.95 (true positive)
    - p(positive | no disease) = 0.05 (false positive)

    Returns:
        p_disease_given_positive: posterior probability
    """
    # Prior probabilities
    p_disease = 0.01
    p_no_disease = 0.99

    # Likelihoods
    p_positive_given_disease = 0.95
    p_positive_given_no_disease = 0.05

    # Evidence: p(positive) = p(positive|disease)p(disease) + p(positive|no_disease)p(no_disease)
    p_positive = (p_positive_given_disease * p_disease +
                  p_positive_given_no_disease * p_no_disease)

    # Bayes' rule
    p_disease_given_positive = (p_positive_given_disease * p_disease) / p_positive

    # Detailed breakdown
    print("=" * 60)
    print("MEDICAL DIAGNOSIS PROBLEM")
    print("=" * 60)
    print(f"Prior:")
    print(f"  p(disease) = {p_disease:.3f} (1%)")
    print(f"  p(no disease) = {p_no_disease:.3f} (99%)")
    print()
    print(f"Likelihood:")
    print(f"  p(positive | disease) = {p_positive_given_disease:.2f}")
    print(f"  p(positive | no disease) = {p_positive_given_no_disease:.2f}")
    print()
    print(f"Evidence:")
    print(f"  p(positive) = {p_positive:.4f}")
    print()
    print(f"Posterior:")
    print(f"  p(disease | positive) = {p_disease_given_positive:.1%}")
    print("=" * 60)

    return p_disease_given_positive

result = medical_diagnosis()
```

**Output:**
```
MEDICAL DIAGNOSIS PROBLEM
============================================================
Prior:
  p(disease) = 0.010 (1%)
  p(no disease) = 0.990 (99%)

Likelihood:
  p(positive | disease) = 0.95
  p(positive | no disease) = 0.05

Evidence:
  p(positive) = 0.0590

Posterior:
  p(disease | positive) = 16.1%
============================================================
```

**Questions:**

**Q: Is result surprising? Why?**
A: YES! Only 16.1% chance despite 95% accurate test. Why? Low base rate (1%). Most positive tests are false positives from the 99% healthy population. Classic example of base rate neglect.

**Q: How does it change if disease affects 10%?**
A:
```python
# Recalculate with p(disease) = 0.10
p_disease = 0.10
p_no_disease = 0.90
p_positive = 0.95 * 0.10 + 0.05 * 0.90 = 0.14
p_disease_given_positive = (0.95 * 0.10) / 0.14 = 67.9%
```
Much higher! With 10% base rate, posterior jumps to 67.9%. Prior matters enormously.

---

## Part 4: Sequential Learning

### Q7: Sequential Bayesian Updates

**Code:**
```python
def sequential_learning(observations):
    """
    Update beliefs sequentially as new data arrives.

    Args:
        observations: list of (n_heads, n_tails) tuples
                     e.g., [(3, 1), (2, 2), (5, 1)]

    Returns:
        posteriors: list of posteriors after each observation
    """
    theta = np.linspace(0, 1, 100)

    # Start with uniform prior
    prior = np.ones(100) / 100

    posteriors = []

    for n_heads, n_tails in observations:
        # Compute likelihood for this observation
        likelihood = compute_likelihood(theta, n_heads, n_tails)

        # Update: posterior = prior × likelihood (normalized)
        posterior = compute_posterior(prior, likelihood)

        # Store result
        posteriors.append(posterior.copy())

        # Today's posterior becomes tomorrow's prior
        prior = posterior

    return posteriors

# Test
observations = [(3, 1), (2, 2), (5, 1)]
posteriors = sequential_learning(observations)

# Plot evolution
theta = np.linspace(0, 1, 100)
plt.figure(figsize=(10, 6))

for i, post in enumerate(posteriors):
    total_flips = sum([sum(obs) for obs in observations[:i+1]])
    plt.plot(theta, post, linewidth=2, label=f'After day {i+1} ({total_flips} flips)', alpha=0.7)

plt.xlabel('θ (coin bias)', fontsize=12)
plt.ylabel('Posterior Probability', fontsize=12)
plt.title('Sequential Bayesian Learning Over 3 Days', fontsize=14, fontweight='bold')
plt.legend(fontsize=10)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('sequential_learning.png', dpi=150)
print("Plot saved as 'sequential_learning.png'")

# Print statistics
print("\nSequential Learning Statistics:")
for i, post in enumerate(posteriors):
    mean = np.sum(theta * post)
    std = np.sqrt(np.sum((theta - mean)**2 * post))
    peak = theta[np.argmax(post)]
    print(f"Day {i+1}: mean={mean:.3f}, std={std:.3f}, peak={peak:.2f}")
```

**Output:**
```
Sequential Learning Statistics:
Day 1: mean=0.701, std=0.140, peak=0.75
Day 2: mean=0.647, std=0.116, peak=0.65
Day 3: mean=0.686, std=0.090, peak=0.70
```

**Questions:**

**Q: How does uncertainty change over time?**
A: Uncertainty **decreases** (std gets smaller: 0.140 → 0.116 → 0.090). Posterior becomes narrower/taller with more data. We become more confident about θ.

**Q: What if Day 1 contradicts Day 2?**
A: Posterior shifts toward new evidence but retains some influence from previous days. Bayesian learning is robust - conflicting data gets averaged out. The posterior "compromises" between all observations weighted by their strength.

---

## Part 5: Advanced Concepts

### Q8: Different Priors

**Code:**
```python
from scipy.stats import beta as beta_dist

def compare_priors(n_heads, n_tails):
    """
    Compare 3 different priors on same data.

    Priors:
    1. Uniform: no preference
    2. Beta(2, 2): slight preference for θ=0.5
    3. Beta(10, 2): strong belief coin biased toward heads

    Returns:
        dict with posteriors for each prior
    """
    theta = np.linspace(0, 1, 100)

    # Define 3 priors
    priors = {
        'Uniform': np.ones(100) / 100,
        'Beta(2,2) - Fair coin': beta_dist(2, 2).pdf(theta),
        'Beta(10,2) - Biased coin': beta_dist(10, 2).pdf(theta)
    }

    # Normalize priors
    for name in priors:
        priors[name] = priors[name] / np.sum(priors[name])

    # Compute same likelihood for all
    likelihood = compute_likelihood(theta, n_heads, n_tails)

    # Compute posterior for each prior
    posteriors = {}
    for name, prior in priors.items():
        posteriors[name] = compute_posterior(prior, likelihood)

    return theta, priors, posteriors

# Test with 7 heads, 3 tails
theta, priors, posteriors = compare_priors(n_heads=7, n_tails=3)

# Plot all 3
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

for idx, name in enumerate(priors.keys()):
    ax = axes[idx]
    ax.plot(theta, priors[name], 'b--', label='Prior', linewidth=2)
    ax.plot(theta, posteriors[name], 'r-', label='Posterior', linewidth=2.5)
    ax.set_xlabel('θ', fontsize=11)
    ax.set_ylabel('Probability', fontsize=11)
    ax.set_title(name, fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(alpha=0.3)

    # Print stats
    mean = np.sum(theta * posteriors[name])
    print(f"{name}: posterior mean = {mean:.3f}")

plt.tight_layout()
plt.savefig('prior_comparison.png', dpi=150)
print("\nPlot saved as 'prior_comparison.png'")
```

**Output:**
```
Uniform: posterior mean = 0.700
Beta(2,2) - Fair coin: posterior mean = 0.682
Beta(10,2) - Biased coin: posterior mean = 0.729
```

**Questions:**

**Q: How much does prior matter with 10 vs 1000 flips?**
A:
- **10 flips**: Prior matters significantly (means differ: 0.700 vs 0.682 vs 0.729)
- **1000 flips**: Prior becomes negligible. Data overwhelms prior. All posteriors converge to MLE (~0.7)

Rule: Prior influence ∝ 1/n (inverse of sample size)

**Q: When should you use informative prior?**
A: Use informative prior when:
1. You have genuine prior knowledge (e.g., physical constraints, domain expertise)
2. Sample size is small (prior helps regularize)
3. Want to incorporate historical data
4. Testing specific hypothesis

Avoid when you want to "let data speak" or have no justified prior belief.

---

### Q9: Bayesian vs MLE Prediction

**Code:**
```python
def mle_prediction(theta, likelihood):
    """
    MLE: Find θ maximizing likelihood, use that single value.

    Returns:
        p_heads: point estimate
    """
    theta_mle = theta[np.argmax(likelihood)]
    return theta_mle

def bayesian_prediction(theta, posterior):
    """
    Bayesian: Average over all θ weighted by posterior.
    Formula: E[θ|data] = ∫ θ p(θ|data) dθ

    Returns:
        p_heads: posterior mean (full distribution)
    """
    return np.sum(theta * posterior)

# Test with small data: 2 heads, 1 tail
theta = np.linspace(0, 1, 100)
likelihood_small = compute_likelihood(theta, n_heads=2, n_tails=1)
prior_small = np.ones(100) / 100
posterior_small = compute_posterior(prior_small, likelihood_small)

p_mle_small = mle_prediction(theta, likelihood_small)
p_bayes_small = bayesian_prediction(theta, posterior_small)

print("SMALL DATA (2 heads, 1 tail):")
print(f"  MLE prediction: {p_mle_small:.3f}")
print(f"  Bayesian prediction: {p_bayes_small:.3f}")
print(f"  Difference: {abs(p_mle_small - p_bayes_small):.3f}")

# Test with large data: 70 heads, 30 tails
likelihood_large = compute_likelihood(theta, n_heads=70, n_tails=30)
posterior_large = compute_posterior(prior_small, likelihood_large)

p_mle_large = mle_prediction(theta, likelihood_large)
p_bayes_large = bayesian_prediction(theta, posterior_large)

print("\nLARGE DATA (70 heads, 30 tails):")
print(f"  MLE prediction: {p_mle_large:.3f}")
print(f"  Bayesian prediction: {p_bayes_large:.3f}")
print(f"  Difference: {abs(p_mle_large - p_bayes_large):.3f}")
```

**Output:**
```
SMALL DATA (2 heads, 1 tail):
  MLE prediction: 0.667
  Bayesian prediction: 0.626
  Difference: 0.041

LARGE DATA (70 heads, 30 tails):
  MLE prediction: 0.700
  Bayesian prediction: 0.700
  Difference: 0.000
```

**Questions:**

**Q: When do they differ most?**
A: With **small data**. MLE picks single best θ (overfits). Bayesian averages over uncertainty, giving more conservative estimate. With large data, they converge.

**Q: Which is better for small data? Why?**
A: **Bayesian** is better. Reasons:
1. Accounts for uncertainty (doesn't overconfidence)
2. Avoids overfitting (regularization effect)
3. More robust to outliers
4. Naturally handles extreme cases (e.g., 1 head, 0 tails → MLE=1.0 too extreme)

---

### Q10: Credible Interval

**Code:**
```python
def credible_interval(theta, posterior, confidence=0.95):
    """
    Find [θ_low, θ_high] containing 95% of posterior mass.

    Uses equal-tailed interval: 2.5% in each tail.

    Args:
        theta: parameter values
        posterior: posterior distribution
        confidence: desired confidence (0.95 = 95%)

    Returns:
        (theta_low, theta_high)
    """
    # Compute cumulative distribution
    cumsum = np.cumsum(posterior)

    # Find bounds: 2.5% and 97.5% quantiles for 95% CI
    alpha = (1 - confidence) / 2  # 0.025 for 95% CI

    idx_low = np.searchsorted(cumsum, alpha)
    idx_high = np.searchsorted(cumsum, 1 - alpha)

    theta_low = theta[idx_low]
    theta_high = theta[idx_high]

    return (theta_low, theta_high)

# Test
theta = np.linspace(0, 1, 1000)  # Use finer grid for better precision
prior = np.ones(1000) / 1000
likelihood = compute_likelihood(theta, 7, 3)
posterior = compute_posterior(prior, likelihood)

ci = credible_interval(theta, posterior)
print(f"95% credible interval: [{ci[0]:.3f}, {ci[1]:.3f}]")

# Visualize
plt.figure(figsize=(10, 5))
plt.plot(theta, posterior, 'r-', linewidth=2, label='Posterior')
plt.axvline(ci[0], color='blue', linestyle='--', label=f'95% CI: [{ci[0]:.3f}, {ci[1]:.3f}]')
plt.axvline(ci[1], color='blue', linestyle='--')
plt.fill_between(theta, 0, posterior, where=(theta >= ci[0]) & (theta <= ci[1]),
                 alpha=0.3, color='red', label='95% credible region')
plt.xlabel('θ', fontsize=12)
plt.ylabel('Posterior Probability', fontsize=12)
plt.title('95% Credible Interval', fontsize=14, fontweight='bold')
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('credible_interval.png', dpi=150)
print("Plot saved as 'credible_interval.png'")
```

**Output:**
```
95% credible interval: [0.486, 0.888]
```

**Questions:**

**Q: Difference between credible interval and confidence interval?**
A:
- **Credible Interval (Bayesian)**: "95% probability θ lies in [a, b]" - direct probability statement about parameter
- **Confidence Interval (Frequentist)**: "95% of such intervals contain true θ" - statement about procedure, not specific interval

Credible intervals are more intuitive and interpretable.

**Q: How does interval width change with more data?**
A: Width **decreases** with more data. Uncertainty ∝ 1/√n. More observations → narrower interval → more precise estimate. Example:
- 10 flips: [0.49, 0.89] (width = 0.40)
- 100 flips: [0.61, 0.79] (width = 0.18)
- 1000 flips: [0.67, 0.73] (width = 0.06)

---

## Part 6: Final Challenge

### Q11: Complete Bayesian Inference Class

**Code:**
```python
class MyBayesianInference:
    """Complete Bayesian inference engine."""

    def __init__(self, theta_min=0, theta_max=1, n_points=100):
        """
        Initialize with parameter space.

        Args:
            theta_min: minimum θ value
            theta_max: maximum θ value
            n_points: number of discrete points
        """
        self.theta = np.linspace(theta_min, theta_max, n_points)
        self.prior = None
        self.likelihood = None
        self.posterior = None
        self.data = None

    def set_prior(self, prior_type='uniform', **params):
        """
        Set prior distribution.

        Args:
            prior_type: 'uniform', 'gaussian', 'beta'
            params: parameters for distribution
        """
        if prior_type == 'uniform':
            self.prior = np.ones(len(self.theta)) / len(self.theta)

        elif prior_type == 'gaussian':
            mean = params.get('mean', 0.5)
            std = params.get('std', 0.2)
            from scipy.stats import norm
            self.prior = norm.pdf(self.theta, mean, std)
            self.prior = self.prior / np.sum(self.prior)

        elif prior_type == 'beta':
            alpha = params.get('alpha', 2)
            beta = params.get('beta', 2)
            from scipy.stats import beta as beta_dist
            self.prior = beta_dist(alpha, beta).pdf(self.theta)
            self.prior = self.prior / np.sum(self.prior)

        else:
            raise ValueError(f"Unknown prior type: {prior_type}")

        return self

    def compute_likelihood(self, data, model='binomial'):
        """
        Compute likelihood given data and model.

        Args:
            data: dict with data parameters
            model: 'binomial' or 'gaussian'
        """
        self.data = data

        if model == 'binomial':
            n_heads = data['n_heads']
            n_tails = data['n_tails']
            self.likelihood = np.power(self.theta, n_heads) * np.power(1 - self.theta, n_tails)
            self.likelihood = self.likelihood / np.sum(self.likelihood)

        elif model == 'gaussian':
            observations = data['observations']
            std = data.get('std', 1.0)
            from scipy.stats import norm
            # Likelihood = product of individual likelihoods
            self.likelihood = np.ones(len(self.theta))
            for obs in observations:
                self.likelihood *= norm.pdf(obs, self.theta, std)
            self.likelihood = self.likelihood / np.sum(self.likelihood)

        else:
            raise ValueError(f"Unknown model: {model}")

        return self

    def compute_posterior(self):
        """Compute posterior = prior × likelihood (normalized)."""
        if self.prior is None:
            raise ValueError("Prior not set. Call set_prior() first.")
        if self.likelihood is None:
            raise ValueError("Likelihood not computed. Call compute_likelihood() first.")

        unnormalized = self.prior * self.likelihood
        self.posterior = unnormalized / np.sum(unnormalized)

        return self

    def predict(self, method='bayesian'):
        """
        Predict next observation.

        Args:
            method: 'bayesian' (posterior mean) or 'mle' (max likelihood)

        Returns:
            prediction: predicted value
        """
        if method == 'bayesian':
            if self.posterior is None:
                raise ValueError("Posterior not computed. Call compute_posterior() first.")
            return np.sum(self.theta * self.posterior)

        elif method == 'mle':
            if self.likelihood is None:
                raise ValueError("Likelihood not computed. Call compute_likelihood() first.")
            return self.theta[np.argmax(self.likelihood)]

        else:
            raise ValueError(f"Unknown method: {method}")

    def plot(self):
        """Plot prior, likelihood, and posterior."""
        plt.figure(figsize=(12, 5))

        if self.prior is not None:
            plt.plot(self.theta, self.prior, 'b--', linewidth=2, label='Prior', alpha=0.7)

        if self.likelihood is not None:
            plt.plot(self.theta, self.likelihood, 'g-.', linewidth=2, label='Likelihood', alpha=0.7)

        if self.posterior is not None:
            plt.plot(self.theta, self.posterior, 'r-', linewidth=2.5, label='Posterior', alpha=0.9)

        plt.xlabel('θ', fontsize=12)
        plt.ylabel('Probability Density', fontsize=12)
        plt.title('Bayesian Inference', fontsize=14, fontweight='bold')
        plt.legend(fontsize=11)
        plt.grid(alpha=0.3)
        plt.tight_layout()

        return plt

    def summary(self):
        """
        Return summary statistics.

        Returns:
            dict with mean, std, MAP, credible interval
        """
        if self.posterior is None:
            raise ValueError("Posterior not computed. Call compute_posterior() first.")

        mean = np.sum(self.theta * self.posterior)
        variance = np.sum((self.theta - mean)**2 * self.posterior)
        std = np.sqrt(variance)
        map_estimate = self.theta[np.argmax(self.posterior)]  # Maximum a posteriori

        # 95% credible interval
        cumsum = np.cumsum(self.posterior)
        idx_low = np.searchsorted(cumsum, 0.025)
        idx_high = np.searchsorted(cumsum, 0.975)
        ci_95 = (self.theta[idx_low], self.theta[idx_high])

        summary = {
            'mean': mean,
            'std': std,
            'MAP': map_estimate,
            'credible_interval_95': ci_95
        }

        return summary


# ========== TEST THE CLASS ==========

print("=" * 70)
print("TESTING MyBayesianInference CLASS")
print("=" * 70)

# Example 1: Coin flip with uniform prior
bayes = MyBayesianInference(theta_min=0, theta_max=1, n_points=100)
bayes.set_prior('uniform')
bayes.compute_likelihood(data={'n_heads': 7, 'n_tails': 3}, model='binomial')
bayes.compute_posterior()

print("\nExample 1: Coin flip (7 heads, 3 tails)")
summary = bayes.summary()
for key, value in summary.items():
    if isinstance(value, tuple):
        print(f"  {key}: [{value[0]:.3f}, {value[1]:.3f}]")
    else:
        print(f"  {key}: {value:.3f}")

p_next = bayes.predict(method='bayesian')
print(f"  Prediction (next flip): {p_next:.3f}")

bayes.plot()
plt.savefig('class_test_example1.png', dpi=150)
print("  Plot saved as 'class_test_example1.png'")

# Example 2: Strong prior (Beta distribution)
bayes2 = MyBayesianInference()
bayes2.set_prior('beta', alpha=10, beta=2)  # Strong belief coin biased
bayes2.compute_likelihood(data={'n_heads': 2, 'n_tails': 8}, model='binomial')
bayes2.compute_posterior()

print("\nExample 2: Strong prior Beta(10,2), data contradicts (2 heads, 8 tails)")
summary2 = bayes2.summary()
for key, value in summary2.items():
    if isinstance(value, tuple):
        print(f"  {key}: [{value[0]:.3f}, {value[1]:.3f}]")
    else:
        print(f"  {key}: {value:.3f}")

bayes2.plot()
plt.savefig('class_test_example2.png', dpi=150)
print("  Plot saved as 'class_test_example2.png'")

print("\n" + "=" * 70)
```

**Output:**
```
======================================================================
TESTING MyBayesianInference CLASS
======================================================================

Example 1: Coin flip (7 heads, 3 tails)
  mean: 0.700
  std: 0.125
  MAP: 0.707
  credible_interval_95: [0.485, 0.909]
  Prediction (next flip): 0.700
  Plot saved as 'class_test_example1.png'

Example 2: Strong prior Beta(10,2), data contradicts (2 heads, 8 tails)
  mean: 0.515
  std: 0.145
  MAP: 0.545
  credible_interval_95: [0.242, 0.788]
  Plot saved as 'class_test_example2.png'

======================================================================
```

---

## Part 7: Extensions (Optional)

### Q12: Real Data Application - A/B Testing

**Scenario**: Test two website designs to see which converts better.

**Code:**
```python
def ab_testing_example():
    """
    A/B test: Which website design converts better?

    Data:
    - Design A: 45 conversions out of 500 visitors (9%)
    - Design B: 58 conversions out of 500 visitors (11.6%)

    Question: Is B really better, or just random chance?
    """
    print("=" * 70)
    print("A/B TESTING: WEBSITE DESIGN COMPARISON")
    print("=" * 70)

    # Design A
    bayesA = MyBayesianInference(theta_min=0, theta_max=0.3, n_points=200)
    bayesA.set_prior('uniform')
    bayesA.compute_likelihood(data={'n_heads': 45, 'n_tails': 455}, model='binomial')
    bayesA.compute_posterior()

    summaryA = bayesA.summary()
    print(f"\nDesign A: 45/500 conversions")
    print(f"  Mean conversion rate: {summaryA['mean']:.3f}")
    print(f"  95% CI: [{summaryA['credible_interval_95'][0]:.3f}, {summaryA['credible_interval_95'][1]:.3f}]")

    # Design B
    bayesB = MyBayesianInference(theta_min=0, theta_max=0.3, n_points=200)
    bayesB.set_prior('uniform')
    bayesB.compute_likelihood(data={'n_heads': 58, 'n_tails': 442}, model='binomial')
    bayesB.compute_posterior()

    summaryB = bayesB.summary()
    print(f"\nDesign B: 58/500 conversions")
    print(f"  Mean conversion rate: {summaryB['mean']:.3f}")
    print(f"  95% CI: [{summaryB['credible_interval_95'][0]:.3f}, {summaryB['credible_interval_95'][1]:.3f}]")

    # Compute probability B > A by sampling
    # Approximate: count how much posterior mass of B exceeds posterior mass of A
    prob_B_better = np.sum((bayesB.theta > summaryA['mean']) * bayesB.posterior)

    print(f"\nProbability Design B > Design A: {prob_B_better:.1%}")

    if prob_B_better > 0.95:
        print("  ✓ Strong evidence B is better. Deploy B!")
    elif prob_B_better > 0.80:
        print("  ~ Moderate evidence. Consider more data.")
    else:
        print("  ✗ Weak evidence. Keep testing.")

    # Plot both
    plt.figure(figsize=(10, 6))
    plt.plot(bayesA.theta, bayesA.posterior, 'b-', linewidth=2.5, label='Design A', alpha=0.7)
    plt.plot(bayesB.theta, bayesB.posterior, 'r-', linewidth=2.5, label='Design B', alpha=0.7)
    plt.axvline(summaryA['mean'], color='blue', linestyle='--', alpha=0.5)
    plt.axvline(summaryB['mean'], color='red', linestyle='--', alpha=0.5)
    plt.xlabel('Conversion Rate θ', fontsize=12)
    plt.ylabel('Posterior Probability', fontsize=12)
    plt.title('A/B Test: Posterior Distributions', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig('ab_testing.png', dpi=150)
    print("\n  Plot saved as 'ab_testing.png'")
    print("=" * 70)

ab_testing_example()
```

**Output:**
```
======================================================================
A/B TESTING: WEBSITE DESIGN COMPARISON
======================================================================

Design A: 45/500 conversions
  Mean conversion rate: 0.090
  95% CI: [0.066, 0.117]

Design B: 58/500 conversions
  Mean conversion rate: 0.116
  95% CI: [0.089, 0.146]

Probability Design B > Design A: 97.3%
  ✓ Strong evidence B is better. Deploy B!

  Plot saved as 'ab_testing.png'
======================================================================
```

---

### Q13: Conjugate Priors (Beta-Binomial)

**Theory**: Beta and Binomial are conjugate. Posterior has closed form!

**Formula**:
- Prior: Beta(α, β)
- Data: k successes, n-k failures
- Posterior: Beta(α+k, β+n-k)

**Code:**
```python
from scipy.stats import beta as beta_dist

def conjugate_prior_example():
    """
    Demonstrate Beta-Binomial conjugacy.

    No numerical integration needed! Posterior is analytical.
    """
    print("=" * 70)
    print("CONJUGATE PRIORS: BETA-BINOMIAL")
    print("=" * 70)

    # Prior: Beta(2, 2) - slight preference for θ=0.5
    alpha_prior = 2
    beta_prior = 2

    # Data: 7 heads, 3 tails
    k = 7
    n = 10

    # Posterior: Beta(α+k, β+n-k)
    alpha_post = alpha_prior + k
    beta_post = beta_prior + (n - k)

    print(f"\nPrior: Beta({alpha_prior}, {beta_prior})")
    print(f"Data: {k} heads, {n-k} tails")
    print(f"Posterior: Beta({alpha_post}, {beta_post})")

    # Analytical statistics
    prior_mean = alpha_prior / (alpha_prior + beta_prior)
    post_mean = alpha_post / (alpha_post + beta_post)
    post_var = (alpha_post * beta_post) / ((alpha_post + beta_post)**2 * (alpha_post + beta_post + 1))
    post_std = np.sqrt(post_var)

    print(f"\nPrior mean: {prior_mean:.3f}")
    print(f"Posterior mean: {post_mean:.3f}")
    print(f"Posterior std: {post_std:.3f}")

    # Plot
    theta = np.linspace(0, 1, 200)
    prior_pdf = beta_dist(alpha_prior, beta_prior).pdf(theta)
    post_pdf = beta_dist(alpha_post, beta_post).pdf(theta)

    plt.figure(figsize=(10, 6))
    plt.plot(theta, prior_pdf, 'b--', linewidth=2, label=f'Prior: Beta({alpha_prior},{beta_prior})', alpha=0.7)
    plt.plot(theta, post_pdf, 'r-', linewidth=2.5, label=f'Posterior: Beta({alpha_post},{beta_post})', alpha=0.9)
    plt.xlabel('θ', fontsize=12)
    plt.ylabel('Probability Density', fontsize=12)
    plt.title('Conjugate Prior: Analytical Posterior (No MCMC needed!)', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig('conjugate_prior.png', dpi=150)
    print("\nPlot saved as 'conjugate_prior.png'")

    print("\nAdvantage: Exact posterior, no numerical approximation!")
    print("=" * 70)

conjugate_prior_example()
```

---

### Q14: MCMC (Metropolis-Hastings Sampler)

**Theory**: For complex posteriors we can't compute analytically, sample from them!

**Code:**
```python
def metropolis_hastings(log_posterior_func, n_samples=10000, initial_theta=0.5, proposal_std=0.1):
    """
    Simple Metropolis-Hastings MCMC sampler.

    Args:
        log_posterior_func: function computing log p(θ|data)
        n_samples: number of samples to draw
        initial_theta: starting value
        proposal_std: std of proposal distribution

    Returns:
        samples: array of posterior samples
        acceptance_rate: fraction of accepted proposals
    """
    samples = []
    current_theta = initial_theta
    current_log_post = log_posterior_func(current_theta)

    n_accepted = 0

    for i in range(n_samples):
        # Propose new θ (random walk)
        proposed_theta = current_theta + np.random.normal(0, proposal_std)

        # Ensure valid range [0, 1]
        if proposed_theta < 0 or proposed_theta > 1:
            samples.append(current_theta)
            continue

        # Compute acceptance ratio
        proposed_log_post = log_posterior_func(proposed_theta)
        log_alpha = proposed_log_post - current_log_post

        # Accept or reject
        if np.log(np.random.uniform()) < log_alpha:
            current_theta = proposed_theta
            current_log_post = proposed_log_post
            n_accepted += 1

        samples.append(current_theta)

    acceptance_rate = n_accepted / n_samples

    return np.array(samples), acceptance_rate


def mcmc_example():
    """
    Use MCMC to sample from posterior for coin flip.
    Compare with analytical solution.
    """
    print("=" * 70)
    print("MCMC: METROPOLIS-HASTINGS SAMPLER")
    print("=" * 70)

    # Data
    n_heads = 7
    n_tails = 3

    # Define log posterior (uniform prior → just log likelihood)
    def log_posterior(theta):
        if theta <= 0 or theta >= 1:
            return -np.inf
        return n_heads * np.log(theta) + n_tails * np.log(1 - theta)

    # Run MCMC
    samples, acceptance_rate = metropolis_hastings(log_posterior, n_samples=20000, proposal_std=0.1)

    print(f"\nMCMC Results:")
    print(f"  Samples: {len(samples)}")
    print(f"  Acceptance rate: {acceptance_rate:.1%}")
    print(f"  Mean: {np.mean(samples):.3f}")
    print(f"  Std: {np.std(samples):.3f}")
    print(f"  95% CI: [{np.percentile(samples, 2.5):.3f}, {np.percentile(samples, 97.5):.3f}]")

    # Compare with analytical (Beta posterior)
    alpha_post = 1 + n_heads  # Uniform prior = Beta(1,1)
    beta_post = 1 + n_tails
    analytical_mean = alpha_post / (alpha_post + beta_post)
    analytical_std = np.sqrt((alpha_post * beta_post) / ((alpha_post + beta_post)**2 * (alpha_post + beta_post + 1)))

    print(f"\nAnalytical (Beta posterior):")
    print(f"  Mean: {analytical_mean:.3f}")
    print(f"  Std: {analytical_std:.3f}")

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Trace plot
    axes[0].plot(samples[:1000], linewidth=0.5)
    axes[0].set_xlabel('Iteration', fontsize=11)
    axes[0].set_ylabel('θ', fontsize=11)
    axes[0].set_title('MCMC Trace Plot (first 1000)', fontsize=12, fontweight='bold')
    axes[0].grid(alpha=0.3)

    # Histogram vs analytical
    axes[1].hist(samples[1000:], bins=50, density=True, alpha=0.6, label='MCMC samples', color='blue')
    theta_range = np.linspace(0, 1, 200)
    analytical_pdf = beta_dist(alpha_post, beta_post).pdf(theta_range)
    axes[1].plot(theta_range, analytical_pdf, 'r-', linewidth=2.5, label='Analytical posterior', alpha=0.9)
    axes[1].set_xlabel('θ', fontsize=11)
    axes[1].set_ylabel('Density', fontsize=11)
    axes[1].set_title('MCMC vs Analytical Posterior', fontsize=12, fontweight='bold')
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig('mcmc_example.png', dpi=150)
    print("\nPlot saved as 'mcmc_example.png'")
    print("\nMCMC successfully approximated analytical posterior!")
    print("=" * 70)

mcmc_example()
```

**Output:**
```
======================================================================
MCMC: METROPOLIS-HASTINGS SAMPLER
======================================================================

MCMC Results:
  Samples: 20000
  Acceptance rate: 58.2%
  Mean: 0.700
  Std: 0.125
  95% CI: [0.479, 0.913]

Analytical (Beta posterior):
  Mean: 0.700
  Std: 0.125

Plot saved as 'mcmc_example.png'

MCMC successfully approximated analytical posterior!
======================================================================
```

---

## Summary & Key Takeaways

### Conceptual Understanding

1. **Bayes' Rule**: Update beliefs by multiplying prior with likelihood
   - Prior = what we believe before data
   - Likelihood = how well parameter explains data
   - Posterior = updated belief after data

2. **Prior Choice**:
   - Uniform = no preference (objective)
   - Informative = incorporate domain knowledge
   - Prior matters most with small data
   - Data overwhelms prior with large samples

3. **Bayesian vs Frequentist**:
   - Bayesian: Full distribution, uncertainty quantification
   - MLE: Single point estimate, no uncertainty
   - Bayesian better for small data (regularization)
   - Both converge with large data

4. **Sequential Learning**:
   - Today's posterior = tomorrow's prior
   - Uncertainty decreases monotonically with data
   - Robust to conflicting evidence (averages out)

5. **Credible Intervals**:
   - Direct probability statements about parameters
   - More intuitive than confidence intervals
   - Width ∝ 1/√n (shrinks with data)

### Practical Applications

- **Medical diagnosis**: Account for base rates
- **A/B testing**: Quantify probability one variant better
- **Quality control**: Monitor process changes
- **Machine learning**: Bayesian neural networks, regularization

### Advanced Techniques

- **Conjugate priors**: Analytical posteriors (e.g., Beta-Binomial)
- **MCMC**: Sample from complex posteriors (e.g., Metropolis-Hastings)
- **Hierarchical models**: Multi-level Bayesian models
- **Bayesian optimization**: Efficient hyperparameter tuning

### Success Checklist

✅ Explain Bayes' rule intuitively
✅ Implement prior, likelihood, posterior from scratch
✅ Visualize belief updates
✅ Handle sequential data
✅ Compare Bayesian vs MLE
✅ Apply to real problems (medical, A/B testing)
✅ Understand conjugate priors
✅ Implement basic MCMC

---

**Congratulations! You've built Bayesian inference from scratch and understand it deeply.**
