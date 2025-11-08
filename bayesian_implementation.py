import numpy as np
def explain_bayesian_rule():
    """
    Explains Bayes' Rule in probability theory.

    Bayes' Rule describes the probability of an event, based on prior knowledge of conditions that might be related to the event.
    It is mathematically expressed as:
    P(teta|data) = (P(data|teta) * P(teta)) / P(data)
    where P(teta|data) represents our updated belief about teta after seeing the new data
    and P(data|teta) is the probability of seeing our data given a specific teta
    and P(data) is the total probability of the data under all possible thetas.
    and P(teta) initial beilief before seing in data 
    p(teta|data) called the posterior probability
    p(teta) called the prior 
    p(data) evidence 
    p(data|teta) likelyhood 
    """
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
    print("it's updatinging your belief about teta by multiplying how new evidence by you initial knowldege deviding by total probability")
    print("=" * 60)


explain_bayesian_rule()


def create_discrete_prior():
    """
    Create a discrete prior distribution over a set of hypotheses.
    """
    teta = np.linspace(0, 1, 11)
    prior = np.ones_like(teta) / len(teta)
    return teta, prior

theta, prior = create_discrete_prior()
print(f"Number of θ values: {len(theta)}")
print(f"Prior sums to: {sum(prior):.10f}")  # Should be 1.0
print(f"θ values: {theta}")
print(f"Prior probs: {prior}")

def create_likelihood_computation(teta_values , number_of_heads, number_of_tails):
    """
    Create a likelihood computation function for binomial data.
    p(data|teta) = teta^number_of_heads * (1 - teta)^number_of_tails
    return the likelihood normalized over all teta values.
    """
    likelihood = teta_values**number_of_heads * (1 - teta_values)**number_of_tails
    likelihood /= np.sum(likelihood)  # Normalize
    return likelihood

theta_values = np.linspace(0, 1, 100)
likelihood = create_likelihood_computation(theta_values, 7, 3)
print(f"Likelihood sums to: {np.sum(likelihood):.10f}")  # Should be 1.0
print(f"Max likelihood at teta =  : {theta_values[np.argmax(likelihood)]:.2f} ")


def compute_posterior(prior, likelihood):
    """
    Compute the posterior distribution using Bayes' Rule.
    P(teta|data) ∝ P(data|teta) * P(teta)
    """
    unnormalized_posterior = likelihood * prior
    posterior = unnormalized_posterior / np.sum(unnormalized_posterior)  # Normalize
    return posterior

teta_values = np.linspace(0, 1, 100)
prior = np.ones_like(teta_values) / len(teta_values)
likelihood = create_likelihood_computation(teta_values, 7, 3)
posterior = compute_posterior(prior, likelihood)
print(f"Posterior sums to: {np.sum(posterior):.10f}")  # Should be 1.0
print(f"Max posterior at teta =  : {teta_values[np.argmax(posterior)]:.2f} ")


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
likelihood = create_likelihood_computation(theta, 7, 3)
posterior = compute_posterior(prior, likelihood)

plot_bayesian_update(theta, prior, likelihood, posterior)
plt.savefig('bayesian_update.png', dpi=150)
print("Plot saved as 'bayesian_update.png'")