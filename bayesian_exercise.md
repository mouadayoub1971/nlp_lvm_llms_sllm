# Bayesian Inference - Build It Yourself Exercise

## Goal
Implement Bayesian inference from scratch, understanding each component deeply.

---

## Part 1: Foundations (Theory + Simple Code)

### Q1: Understanding Bayes' Rule
**Theory**: Write Bayes' rule in your own words. What does each component mean?
- Prior p(θ)
- Likelihood p(data|θ)
- Posterior p(θ|data)
- Evidence p(data)

**Code Task**: Write a function that prints the formula with explanations.

---

### Q2: Discrete Prior
**Task**: Create a simple discrete prior for a coin flip problem.

```python
# TODO: Implement this
def create_discrete_prior():
    """
    Create a discrete prior over coin bias θ = [0.0, 0.1, 0.2, ..., 1.0]
    Assume uniform prior (all values equally likely)

    Returns:
        theta_values: array of possible θ values
        prior_probs: array of prior probabilities (should sum to 1)
    """
    pass

# Test it
theta, prior = create_discrete_prior()
print(f"Number of θ values: {len(theta)}")
print(f"Prior sums to: {sum(prior)}")  # Should be 1.0
```

**Questions**:
- Why do we normalize (divide by sum)?
- What does "uniform prior" mean philosophically?

---

### Q3: Computing Likelihood
**Task**: Compute binomial likelihood for coin flips.

```python
# TODO: Implement this
def compute_likelihood(theta_values, n_heads, n_tails):
    """
    Compute p(data|θ) for each θ value.

    Formula: L(θ) = θ^n_heads * (1-θ)^n_tails

    Args:
        theta_values: array of possible θ
        n_heads: number of heads observed
        n_tails: number of tails observed

    Returns:
        likelihood: array of likelihood values (normalized)
    """
    pass

# Test with 7 heads, 3 tails
theta = np.linspace(0, 1, 100)
likelihood = compute_likelihood(theta, n_heads=7, n_tails=3)
print(f"Max likelihood at θ = {theta[np.argmax(likelihood)]:.2f}")  # Should be ~0.7
```

**Questions**:
- What θ maximizes the likelihood? Why?
- What happens if n_heads=0 and n_tails=0?

---

### Q4: Applying Bayes' Rule
**Task**: Combine prior and likelihood to get posterior.

```python
# TODO: Implement this
def compute_posterior(prior, likelihood):
    """
    Apply Bayes' rule: posterior ∝ likelihood × prior

    Args:
        prior: array of prior probabilities
        likelihood: array of likelihood values

    Returns:
        posterior: array of posterior probabilities (normalized)
    """
    pass

# Test
theta = np.linspace(0, 1, 100)
prior = np.ones(100) / 100  # Uniform
likelihood = compute_likelihood(theta, 7, 3)
posterior = compute_posterior(prior, likelihood)

print(f"Posterior sums to: {sum(posterior)}")  # Should be 1.0
print(f"Posterior mean: {np.sum(theta * posterior):.3f}")  # Should be ~0.7
```

**Questions**:
- Why do we multiply prior × likelihood?
- What is the "evidence" p(data)? Why can we ignore it?

---

## Part 2: Visualization

### Q5: Plot Prior, Likelihood, Posterior
**Task**: Create a visualization showing all three distributions.

```python
import matplotlib.pyplot as plt

# TODO: Implement this
def plot_bayesian_update(theta, prior, likelihood, posterior):
    """
    Plot prior, likelihood, and posterior on same graph.

    Use different colors/styles:
    - Prior: blue dashed
    - Likelihood: green dash-dot
    - Posterior: red solid
    """
    pass

# Test
# (use data from Q4)
plot_bayesian_update(theta, prior, likelihood, posterior)
plt.show()
```

**Questions**:
- Where is the posterior peak compared to likelihood peak?
- How would the posterior change if prior was not uniform?

---

## Part 3: Real-World Scenario

### Q6: Medical Diagnosis Problem
**Scenario**:
- Disease affects 1% of population (prior)
- Test is 95% accurate (likelihood)
- You test positive

**Tasks**:
1. Set up prior: p(disease) = 0.01
2. Compute likelihood:
   - p(positive test | disease) = 0.95
   - p(positive test | no disease) = 0.05
3. Use Bayes' rule to find: p(disease | positive test)

```python
# TODO: Implement this
def medical_diagnosis():
    """
    Calculate probability of having disease given positive test.

    Returns:
        p_disease_given_positive: probability
    """
    # Hint: Use discrete case with 2 states: [disease, no_disease]
    pass

result = medical_diagnosis()
print(f"Probability of disease given positive test: {result:.1%}")
```

**Questions**:
- Is the result surprising? Why?
- How does it change if disease affects 10% of population?

---

## Part 4: Sequential Learning

### Q7: Sequential Bayesian Updates
**Task**: Implement learning over multiple days.

```python
# TODO: Implement this
def sequential_learning(observations):
    """
    Update beliefs sequentially as new data arrives.

    Args:
        observations: list of (n_heads, n_tails) tuples
                     e.g., [(3, 1), (2, 2), (5, 1)]

    Returns:
        posteriors: list of posterior distributions after each day
    """
    pass

# Test
observations = [(3, 1), (2, 2), (5, 1)]
posteriors = sequential_learning(observations)

# Plot evolution
for i, post in enumerate(posteriors):
    plt.plot(theta, post, label=f'Day {i+1}')
plt.legend()
plt.show()
```

**Questions**:
- How does uncertainty change over time?
- What if Day 1 data contradicts Day 2 data?

---

## Part 5: Advanced Concepts

### Q8: Different Priors
**Task**: Compare 3 different priors on same data.

```python
# TODO: Implement this
def compare_priors(n_heads, n_tails):
    """
    Test 3 priors:
    1. Uniform: no preference
    2. Beta(2, 2): slight preference for θ=0.5 (fair coin)
    3. Beta(10, 2): strong belief in biased coin

    Returns:
        dict with posteriors for each prior
    """
    pass

results = compare_priors(n_heads=7, n_tails=3)
# Plot all 3 posteriors
```

**Questions**:
- How much does prior matter with 10 flips? With 1000 flips?
- When should you use an informative prior?

---

### Q9: Bayesian vs MLE Prediction
**Task**: Implement both prediction methods.

```python
# TODO: Implement both
def mle_prediction(theta, likelihood):
    """
    Find θ that maximizes likelihood.
    Predict next flip using only this θ.

    Returns:
        p_heads: probability of heads (point estimate)
    """
    pass

def bayesian_prediction(theta, posterior):
    """
    Average over all θ weighted by posterior.
    Formula: E[θ|data] = ∫ θ p(θ|data) dθ

    Returns:
        p_heads: probability of heads (full distribution)
    """
    pass

# Compare both
theta = np.linspace(0, 1, 100)
# (compute likelihood and posterior for 2 heads, 1 tail)

p_mle = mle_prediction(theta, likelihood)
p_bayes = bayesian_prediction(theta, posterior)

print(f"MLE prediction: {p_mle:.3f}")
print(f"Bayesian prediction: {p_bayes:.3f}")
print(f"Difference: {abs(p_mle - p_bayes):.3f}")
```

**Questions**:
- When do they differ most?
- Which is better for small data? Why?

---

### Q10: Credible Interval
**Task**: Compute 95% credible interval for θ.

```python
# TODO: Implement this
def credible_interval(theta, posterior, confidence=0.95):
    """
    Find interval [θ_low, θ_high] containing 95% of posterior mass.

    Args:
        theta: parameter values
        posterior: posterior distribution
        confidence: desired confidence level (0.95 for 95%)

    Returns:
        (theta_low, theta_high)
    """
    pass

# Test
ci = credible_interval(theta, posterior)
print(f"95% credible interval: [{ci[0]:.3f}, {ci[1]:.3f}]")
```

**Questions**:
- What's the difference between credible interval and confidence interval?
- How does interval width change with more data?

---

## Part 6: Final Challenge

### Q11: Build Complete Class
**Task**: Combine everything into a reusable class (like the original code).

```python
class MyBayesianInference:
    """Your own Bayesian inference engine."""

    def __init__(self):
        # TODO
        pass

    def set_prior(self, prior_type='uniform', **params):
        # TODO: support uniform, gaussian, beta
        pass

    def compute_likelihood(self, data, model='binomial'):
        # TODO: support binomial, gaussian
        pass

    def compute_posterior(self):
        # TODO
        pass

    def predict(self, method='bayesian'):
        # TODO: support 'mle' and 'bayesian'
        pass

    def plot(self):
        # TODO
        pass

    def summary(self):
        # TODO: return mean, std, MAP, credible interval
        pass
```

**Test Your Class**:
```python
# Should work like this:
bayes = MyBayesianInference()
bayes.set_prior('uniform', theta_min=0, theta_max=1)
bayes.compute_likelihood(data={'n_heads': 7, 'n_tails': 3}, model='binomial')
bayes.compute_posterior()

print(bayes.summary())
bayes.plot()

p_next = bayes.predict(method='bayesian')
print(f"Probability of heads: {p_next:.3f}")
```

---

## Part 7: Extensions (Optional)

### Q12: Real Data Application
Pick one:
- **A/B Testing**: Which website design converts better?
- **Spam Filter**: Is email spam given word frequencies?
- **Sports**: Estimate player skill from game results

### Q13: Conjugate Priors
Research and implement Beta-Binomial conjugacy:
- Prior: Beta(α, β)
- Likelihood: Binomial(n_heads, n_tails)
- Posterior: Beta(α + n_heads, β + n_tails)

No numerical integration needed!

### Q14: Markov Chain Monte Carlo (MCMC)
For complex posteriors that can't be computed analytically:
- Implement simple Metropolis-Hastings sampler
- Sample from posterior distribution
- Estimate statistics from samples

---

## Success Criteria

You've mastered Bayesian inference if you can:
- ✓ Explain Bayes' rule in plain English
- ✓ Implement prior, likelihood, posterior from scratch
- ✓ Visualize belief updates
- ✓ Handle sequential data
- ✓ Compare Bayesian vs frequentist approaches
- ✓ Apply to real problems
- ✓ Explain when/why Bayesian methods excel

---

## Tips

1. **Start simple**: Use discrete θ with 10 values first
2. **Check normalization**: Probabilities should always sum to 1
3. **Visualize everything**: Plots reveal bugs quickly
4. **Test edge cases**: What if n_heads=0? θ=0? θ=1?
5. **Compare with original**: Your results should match `bayesian.py`

---

## Resources

- Prior = "What I believe before seeing data"
- Likelihood = "How probable is my data given θ?"
- Posterior = "What I believe after seeing data"
- Evidence = "How probable is my data overall?" (normalization constant)

**Key Formula**:
```
              Likelihood × Prior
Posterior = ───────────────────────
                  Evidence
```

Good luck! Start with Q1 and work through sequentially. Each question builds on previous ones.
