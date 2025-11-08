"""
Bayesian Inference Implementation
MIT-style demonstration of Bayesian updating with priors, likelihoods, and posteriors.

Author: MIT AI & Math Professor (simulated)
Date: 2025-11-04
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from typing import Tuple, Callable


class BayesianInference:
    """
    A class to demonstrate Bayesian inference with various priors and likelihoods.
    """

    def __init__(self):
        """Initialize the Bayesian inference engine."""
        self.prior = None
        self.likelihood = None
        self.posterior = None

    def set_uniform_prior(self, theta_min: float = 0.0, theta_max: float = 1.0,
                         n_points: int = 1000) -> Tuple[np.ndarray, np.ndarray]:
        """
        Set a uniform (maximum ignorance) prior.

        Args:
            theta_min: Minimum value of parameter space
            theta_max: Maximum value of parameter space
            n_points: Number of discretization points

        Returns:
            Tuple of (theta_values, prior_probabilities)
        """
        theta = np.linspace(theta_min, theta_max, n_points)
        prior = np.ones_like(theta) / n_points  # Uniform distribution

        self.theta = theta
        self.prior = prior

        return theta, prior

    def set_gaussian_prior(self, mean: float = 0.0, std: float = 1.0,
                          theta_min: float = -5.0, theta_max: float = 5.0,
                          n_points: int = 1000) -> Tuple[np.ndarray, np.ndarray]:
        """
        Set a Gaussian prior (common in regularization).

        Args:
            mean: Mean of the Gaussian
            std: Standard deviation
            theta_min: Minimum value of parameter space
            theta_max: Maximum value of parameter space
            n_points: Number of discretization points

        Returns:
            Tuple of (theta_values, prior_probabilities)
        """
        theta = np.linspace(theta_min, theta_max, n_points)
        prior = stats.norm.pdf(theta, loc=mean, scale=std)
        prior = prior / np.sum(prior)  # Normalize

        self.theta = theta
        self.prior = prior

        return theta, prior

    def compute_binomial_likelihood(self, n_heads: int, n_tails: int) -> np.ndarray:
        """
        Compute likelihood for coin flip data (Binomial).

        Args:
            n_heads: Number of heads observed
            n_tails: Number of tails observed

        Returns:
            Likelihood values for each theta
        """
        if self.theta is None:
            raise ValueError("Must set prior first!")

        # p(data|theta) = theta^n_heads * (1-theta)^n_tails
        likelihood = (self.theta ** n_heads) * ((1 - self.theta) ** n_tails)

        # Normalize
        likelihood = likelihood / np.sum(likelihood)

        self.likelihood = likelihood
        return likelihood

    def compute_gaussian_likelihood(self, data: np.ndarray, sigma: float = 1.0) -> np.ndarray:
        """
        Compute likelihood for Gaussian-distributed data.

        Args:
            data: Observed data points
            sigma: Known standard deviation

        Returns:
            Likelihood values for each theta (mean parameter)
        """
        if self.theta is None:
            raise ValueError("Must set prior first!")

        # p(data|theta) = product of N(x_i | theta, sigma^2)
        likelihood = np.ones_like(self.theta)

        for x in data:
            likelihood *= stats.norm.pdf(x, loc=self.theta, scale=sigma)

        # Normalize
        likelihood = likelihood / np.sum(likelihood)

        self.likelihood = likelihood
        return likelihood

    def compute_posterior(self) -> np.ndarray:
        """
        Compute posterior using Bayes' rule:
        p(theta|data) = p(data|theta) * p(theta) / p(data)

        Returns:
            Posterior distribution
        """
        if self.prior is None or self.likelihood is None:
            raise ValueError("Must set both prior and likelihood first!")

        # Bayes' rule (numerator)
        posterior = self.likelihood * self.prior

        # Evidence p(data) - normalizing constant
        evidence = np.sum(posterior)

        # Normalized posterior
        posterior = posterior / evidence

        self.posterior = posterior
        return posterior

    def plot_distributions(self, title: str = "Bayesian Inference"):
        """
        Visualize prior, likelihood, and posterior.

        Args:
            title: Plot title
        """
        if self.posterior is None:
            raise ValueError("Must compute posterior first!")

        plt.figure(figsize=(12, 6))

        plt.plot(self.theta, self.prior, 'b--', linewidth=2, label='Prior p(θ)')
        plt.plot(self.theta, self.likelihood, 'g-.', linewidth=2, label='Likelihood p(data|θ)')
        plt.plot(self.theta, self.posterior, 'r-', linewidth=3, label='Posterior p(θ|data)')

        plt.xlabel('θ (parameter)', fontsize=12)
        plt.ylabel('Probability Density', fontsize=12)
        plt.title(title, fontsize=14, fontweight='bold')
        plt.legend(fontsize=11)
        plt.grid(True, alpha=0.3)

        # Add MAP (Maximum A Posteriori) estimate
        map_estimate = self.theta[np.argmax(self.posterior)]
        plt.axvline(map_estimate, color='red', linestyle=':', linewidth=2,
                   label=f'MAP estimate: {map_estimate:.3f}')
        plt.legend(fontsize=11)

        plt.tight_layout()
        plt.show()

    def get_statistics(self) -> dict:
        """
        Compute statistics of the posterior distribution.

        Returns:
            Dictionary with mean, variance, MAP, and credible interval
        """
        if self.posterior is None:
            raise ValueError("Must compute posterior first!")

        # Mean
        posterior_mean = np.sum(self.theta * self.posterior)

        # Variance
        posterior_var = np.sum(((self.theta - posterior_mean) ** 2) * self.posterior)

        # MAP (Maximum A Posteriori)
        map_estimate = self.theta[np.argmax(self.posterior)]

        # 95% Credible Interval
        cumulative = np.cumsum(self.posterior)
        idx_lower = np.searchsorted(cumulative, 0.025)
        idx_upper = np.searchsorted(cumulative, 0.975)
        credible_interval = (self.theta[idx_lower], self.theta[idx_upper])

        return {
            'mean': posterior_mean,
            'variance': posterior_var,
            'std': np.sqrt(posterior_var),
            'MAP': map_estimate,
            'credible_interval_95': credible_interval
        }


def example_coin_flip():
    """
    Example 1: Estimating coin bias from flip data.
    """
    print("=" * 70)
    print("EXAMPLE 1: Coin Flip Bias Estimation")
    print("=" * 70)

    # Initialize
    bayes = BayesianInference()

    # Set uniform prior (we don't know the bias)
    bayes.set_uniform_prior(theta_min=0.0, theta_max=1.0, n_points=1000)
    print("\n[Prior] Uniform distribution over [0, 1]")

    # Observe data: 7 heads, 3 tails
    n_heads, n_tails = 7, 3
    print(f"[Data] Observed {n_heads} heads and {n_tails} tails")

    # Compute likelihood
    bayes.compute_binomial_likelihood(n_heads, n_tails)

    # Compute posterior
    bayes.compute_posterior()

    # Get statistics
    stats = bayes.get_statistics()
    print(f"\n[Posterior Statistics]")
    print(f"  Mean: {stats['mean']:.4f}")
    print(f"  Std:  {stats['std']:.4f}")
    print(f"  MAP:  {stats['MAP']:.4f}")
    print(f"  95% Credible Interval: [{stats['credible_interval_95'][0]:.4f}, "
          f"{stats['credible_interval_95'][1]:.4f}]")

    # Visualize
    bayes.plot_distributions(title="Coin Flip: Estimating Bias θ (7 Heads, 3 Tails)")

    print("\nInterpretation: We started with no knowledge (uniform prior),")
    print("but after seeing 7/10 heads, we believe θ ≈ 0.7 (but with uncertainty!)")
    print()


def example_sequential_learning():
    """
    Example 2: Sequential Bayesian updating (today's posterior = tomorrow's prior).
    """
    print("=" * 70)
    print("EXAMPLE 2: Sequential Learning")
    print("=" * 70)

    # Day 1: Observe 3 heads, 1 tail
    bayes = BayesianInference()
    bayes.set_uniform_prior(theta_min=0.0, theta_max=1.0, n_points=1000)
    bayes.compute_binomial_likelihood(n_heads=3, n_tails=1)
    bayes.compute_posterior()

    print("\n[Day 1] Observed 3 heads, 1 tail")
    stats1 = bayes.get_statistics()
    print(f"  Posterior Mean: {stats1['mean']:.4f}")

    # Day 2: Use yesterday's posterior as today's prior
    # Observe 2 more heads, 2 tails
    bayes.prior = bayes.posterior.copy()  # Update!
    bayes.compute_binomial_likelihood(n_heads=2, n_tails=2)
    bayes.compute_posterior()

    print("\n[Day 2] Observed 2 heads, 2 tails (using Day 1 posterior as prior)")
    stats2 = bayes.get_statistics()
    print(f"  Posterior Mean: {stats2['mean']:.4f}")

    bayes.plot_distributions(title="Sequential Learning: Day 2 Update")

    print("\nInterpretation: Our belief evolves as we see more data!")
    print()


def example_gaussian_prior():
    """
    Example 3: Gaussian prior for regularization (like Ridge regression).
    """
    print("=" * 70)
    print("EXAMPLE 3: Gaussian Prior (Regularization)")
    print("=" * 70)

    # Initialize
    bayes = BayesianInference()

    # Set Gaussian prior centered at 0 (prefer small weights)
    bayes.set_gaussian_prior(mean=0.0, std=0.5, theta_min=-3, theta_max=3, n_points=1000)
    print("\n[Prior] Gaussian N(0, 0.5²) - prefers small values (regularization)")

    # Observe data from a Gaussian with true mean = 1.0
    np.random.seed(42)
    data = np.random.normal(loc=1.0, scale=0.5, size=5)
    print(f"[Data] {len(data)} observations: {data}")

    # Compute likelihood
    bayes.compute_gaussian_likelihood(data, sigma=0.5)

    # Compute posterior
    bayes.compute_posterior()

    # Get statistics
    stats = bayes.get_statistics()
    print(f"\n[Posterior Statistics]")
    print(f"  Mean: {stats['mean']:.4f}")
    print(f"  MAP:  {stats['MAP']:.4f}")
    print(f"  95% Credible Interval: [{stats['credible_interval_95'][0]:.4f}, "
          f"{stats['credible_interval_95'][1]:.4f}]")

    # Visualize
    bayes.plot_distributions(title="Gaussian Prior: Regularization Effect")

    print("\nInterpretation: The prior pulls our estimate toward 0,")
    print("but data pulls it toward the true value (~1.0). Trade-off!")
    print()


def example_mle_vs_bayesian_prediction():
    """
    Example 4: MLE vs Bayesian Prediction - THE KEY DIFFERENCE!

    This demonstrates the formula from your text:
    p(x_m+1 | x_1,...,x_m) = ∫ p(x_m+1|θ) p(θ|x_1,...,x_m) dθ
    """
    print("=" * 70)
    print("EXAMPLE 4: MLE vs BAYESIAN PREDICTION (The Critical Difference)")
    print("=" * 70)

    # Scenario: We flip a coin 3 times and get 2 heads, 1 tail
    # Now we want to predict: what's the probability the NEXT flip is heads?

    n_heads, n_tails = 2, 1
    print(f"\n[Data] Observed {n_heads} heads, {n_tails} tails in {n_heads + n_tails} flips")
    print("Question: What's the probability the NEXT flip is heads?\n")

    # Initialize Bayesian inference
    bayes = BayesianInference()
    bayes.set_uniform_prior(theta_min=0.0, theta_max=1.0, n_points=1000)
    bayes.compute_binomial_likelihood(n_heads, n_tails)
    bayes.compute_posterior()

    # ==================== MLE APPROACH ====================
    print("─" * 70)
    print("APPROACH 1: Maximum Likelihood Estimation (Point Estimate)")
    print("─" * 70)

    # MLE: Find the single best θ
    theta_mle = bayes.theta[np.argmax(bayes.likelihood)]
    print(f"Step 1: Find best θ")
    print(f"        θ_MLE = argmax p(data|θ) = {theta_mle:.4f}")

    # MLE prediction: Use only this single θ
    p_heads_mle = theta_mle
    print(f"\nStep 2: Predict using ONLY θ_MLE")
    print(f"        p(next flip = heads) = θ_MLE = {p_heads_mle:.4f}")

    print(f"\n[MLE Prediction] Probability of heads = {p_heads_mle:.4f} (or {p_heads_mle*100:.1f}%)")
    print("⚠️  Problem: Assumes we KNOW θ exactly. Ignores uncertainty!")

    # ==================== BAYESIAN APPROACH ====================
    print("\n" + "─" * 70)
    print("APPROACH 2: Bayesian Prediction (Full Distribution)")
    print("─" * 70)

    print("Step 1: Compute full posterior distribution p(θ|data)")
    stats = bayes.get_statistics()
    print(f"        Posterior mean: {stats['mean']:.4f}")
    print(f"        Posterior std:  {stats['std']:.4f}")
    print(f"        95% CI: [{stats['credible_interval_95'][0]:.4f}, "
          f"{stats['credible_interval_95'][1]:.4f}]")

    print("\nStep 2: Average over ALL possible θ (marginalization)")
    print("        p(x_m+1|data) = ∫ p(x_m+1|θ) p(θ|data) dθ")

    # Bayesian prediction: Integrate over all θ weighted by posterior
    # For coin flip: p(heads|data) = ∫ θ · p(θ|data) dθ = E[θ|data]
    p_heads_bayesian = np.sum(bayes.theta * bayes.posterior)

    print(f"        p(x_m+1|data) = ∫ θ · p(θ|data) dθ")
    print(f"                       = E[θ|data]")
    print(f"                       = {p_heads_bayesian:.4f}")

    print(f"\n[Bayesian Prediction] Probability of heads = {p_heads_bayesian:.4f} "
          f"(or {p_heads_bayesian*100:.1f}%)")
    print("✓  Benefit: Accounts for uncertainty in θ!")

    # ==================== COMPARISON ====================
    print("\n" + "=" * 70)
    print("COMPARISON & INTERPRETATION")
    print("=" * 70)

    print(f"\nMLE Prediction:      {p_heads_mle:.4f}")
    print(f"Bayesian Prediction: {p_heads_bayesian:.4f}")
    print(f"Difference:          {abs(p_heads_bayesian - p_heads_mle):.4f}")

    print("\nWhy are they different?")
    print("  • MLE uses ONLY the peak of the likelihood (θ = 0.667)")
    print("  • Bayesian averages over ALL plausible θ values")
    print("  • With little data, uncertainty is HIGH → bigger difference")
    print("  • With lots of data, posterior concentrates → similar results")

    # Visualize the prediction process
    plt.figure(figsize=(14, 5))

    # Left plot: Posterior distribution
    plt.subplot(1, 2, 1)
    plt.plot(bayes.theta, bayes.posterior, 'b-', linewidth=2, label='Posterior p(θ|data)')
    plt.axvline(theta_mle, color='red', linestyle='--', linewidth=2,
                label=f'MLE: θ = {theta_mle:.3f}')
    plt.axvline(p_heads_bayesian, color='green', linestyle='--', linewidth=2,
                label=f'Bayesian: E[θ] = {p_heads_bayesian:.3f}')
    plt.fill_between(bayes.theta, 0, bayes.posterior, alpha=0.3)
    plt.xlabel('θ (coin bias)', fontsize=11)
    plt.ylabel('Posterior Probability', fontsize=11)
    plt.title('Posterior Distribution Over θ', fontsize=12, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)

    # Right plot: Prediction for next flip
    plt.subplot(1, 2, 2)
    predictions = ['Tails', 'Heads']

    # MLE predictions
    mle_probs = [1 - p_heads_mle, p_heads_mle]
    bars1 = plt.bar([0.2, 1.2], mle_probs, width=0.35, label='MLE',
                    color='red', alpha=0.7)

    # Bayesian predictions
    bayes_probs = [1 - p_heads_bayesian, p_heads_bayesian]
    bars2 = plt.bar([0.6, 1.6], bayes_probs, width=0.35, label='Bayesian',
                    color='green', alpha=0.7)

    plt.xticks([0.4, 1.4], predictions)
    plt.ylabel('Predicted Probability', fontsize=11)
    plt.title('Prediction for Next Flip', fontsize=12, fontweight='bold')
    plt.legend(fontsize=10)
    plt.ylim([0, 1])
    plt.grid(True, alpha=0.3, axis='y')

    # Add value labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.3f}',
                    ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    plt.show()

    # ==================== THE MARGINALIZATION FORMULA ====================
    print("\n" + "=" * 70)
    print("THE MARGINALIZATION FORMULA (From Your Text)")
    print("=" * 70)
    print("\nGeneral case:")
    print("  p(x_m+1 | x_1,...,x_m) = ∫ p(x_m+1|θ) p(θ|x_1,...,x_m) dθ")
    print("                            ↑            ↑")
    print("                            |            └─ Posterior (how likely is θ?)")
    print("                            └─ Likelihood (how likely is x given θ?)")
    print("\nIn our coin example:")
    print("  p(heads|data) = ∫ θ · p(θ|data) dθ")
    print("                = E[θ|data]")
    print(f"                = {p_heads_bayesian:.4f}")
    print("\n✓ This is the EXACT formula from your textbook!")
    print()


def example_small_vs_large_data():
    """
    Example 5: Show that MLE ≈ Bayesian with LARGE data, but differ with SMALL data.
    """
    print("=" * 70)
    print("EXAMPLE 5: Small Data vs Large Data")
    print("=" * 70)

    # True coin bias
    true_bias = 0.7
    print(f"\n[Ground Truth] True coin bias θ = {true_bias}")

    # Small data: 7 heads, 3 tails
    print("\n" + "─" * 70)
    print("SCENARIO A: SMALL DATA (10 flips)")
    print("─" * 70)

    bayes_small = BayesianInference()
    bayes_small.set_uniform_prior(theta_min=0.0, theta_max=1.0, n_points=1000)
    bayes_small.compute_binomial_likelihood(n_heads=7, n_tails=3)
    bayes_small.compute_posterior()

    mle_small = bayes_small.theta[np.argmax(bayes_small.likelihood)]
    bayesian_small = np.sum(bayes_small.theta * bayes_small.posterior)

    print(f"  MLE estimate:      {mle_small:.4f}")
    print(f"  Bayesian estimate: {bayesian_small:.4f}")
    print(f"  Difference:        {abs(bayesian_small - mle_small):.4f}")
    print(f"  Posterior std:     {bayes_small.get_statistics()['std']:.4f} (HIGH uncertainty)")

    # Large data: 700 heads, 300 tails
    print("\n" + "─" * 70)
    print("SCENARIO B: LARGE DATA (1000 flips)")
    print("─" * 70)

    bayes_large = BayesianInference()
    bayes_large.set_uniform_prior(theta_min=0.0, theta_max=1.0, n_points=1000)
    bayes_large.compute_binomial_likelihood(n_heads=700, n_tails=300)
    bayes_large.compute_posterior()

    mle_large = bayes_large.theta[np.argmax(bayes_large.likelihood)]
    bayesian_large = np.sum(bayes_large.theta * bayes_large.posterior)

    print(f"  MLE estimate:      {mle_large:.4f}")
    print(f"  Bayesian estimate: {bayesian_large:.4f}")
    print(f"  Difference:        {abs(bayesian_large - mle_large):.4f}")
    print(f"  Posterior std:     {bayes_large.get_statistics()['std']:.4f} (LOW uncertainty)")

    # Visualization
    plt.figure(figsize=(14, 5))

    # Small data
    plt.subplot(1, 2, 1)
    plt.plot(bayes_small.theta, bayes_small.posterior, 'b-', linewidth=2)
    plt.axvline(mle_small, color='red', linestyle='--', linewidth=2, label=f'MLE: {mle_small:.3f}')
    plt.axvline(bayesian_small, color='green', linestyle='--', linewidth=2,
                label=f'Bayesian: {bayesian_small:.3f}')
    plt.axvline(true_bias, color='black', linestyle=':', linewidth=2, label=f'True θ: {true_bias}')
    plt.fill_between(bayes_small.theta, 0, bayes_small.posterior, alpha=0.3)
    plt.xlabel('θ', fontsize=11)
    plt.ylabel('Posterior', fontsize=11)
    plt.title('Small Data (n=10): Wide Posterior', fontsize=12, fontweight='bold')
    plt.legend(fontsize=9)
    plt.grid(True, alpha=0.3)

    # Large data
    plt.subplot(1, 2, 2)
    plt.plot(bayes_large.theta, bayes_large.posterior, 'b-', linewidth=2)
    plt.axvline(mle_large, color='red', linestyle='--', linewidth=2, label=f'MLE: {mle_large:.3f}')
    plt.axvline(bayesian_large, color='green', linestyle='--', linewidth=2,
                label=f'Bayesian: {bayesian_large:.3f}')
    plt.axvline(true_bias, color='black', linestyle=':', linewidth=2, label=f'True θ: {true_bias}')
    plt.fill_between(bayes_large.theta, 0, bayes_large.posterior, alpha=0.3)
    plt.xlabel('θ', fontsize=11)
    plt.ylabel('Posterior', fontsize=11)
    plt.title('Large Data (n=1000): Narrow Posterior', fontsize=12, fontweight='bold')
    plt.legend(fontsize=9)
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    print("\n" + "=" * 70)
    print("KEY INSIGHT")
    print("=" * 70)
    print("\n  • SMALL DATA → High uncertainty → MLE ≠ Bayesian")
    print("  • LARGE DATA → Low uncertainty → MLE ≈ Bayesian")
    print("  • Bayesian prediction is MORE ROBUST with limited data!")
    print()


def main():
    """
    Run all examples demonstrating Bayesian inference.
    """
    print("\n")
    print("╔═══════════════════════════════════════════════════════════════════╗")
    print("║         BAYESIAN INFERENCE: From Theory to Practice              ║")
    print("║              MIT AI & Math Professor Demonstration                ║")
    print("╚═══════════════════════════════════════════════════════════════════╝")
    print("\n")

    # Run examples
    example_coin_flip()
    example_sequential_learning()
    example_gaussian_prior()

    # NEW: The critical MLE vs Bayesian comparison
    example_mle_vs_bayesian_prediction()
    example_small_vs_large_data()

    print("=" * 70)
    print("All examples completed!")
    print("=" * 70)
    print("\nKey Takeaways:")
    print("  1. Prior + Likelihood → Posterior (via Bayes' rule)")
    print("  2. Priors encode our initial beliefs/preferences")
    print("  3. Data updates beliefs through the likelihood")
    print("  4. Posterior captures uncertainty, not just point estimates")
    print("  5. Sequential learning: yesterday's posterior = today's prior")
    print("\n  ★★ CRITICAL DIFFERENCE ★★")
    print("  6. MLE uses POINT estimate → p(x_new) = p(x_new|θ_MLE)")
    print("  7. Bayesian uses DISTRIBUTION → p(x_new|data) = ∫ p(x_new|θ)p(θ|data)dθ")
    print("  8. Bayesian accounts for uncertainty in θ (better for small data!)")
    print("\n")


if __name__ == "__main__":
    main()
