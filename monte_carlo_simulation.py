"""
Monte Carlo Simulation for Infinite Slope Factor of Safety (Eq. 2.1)

Equation:
    F_s = [c' + {gamma_1 * h_1 + (gamma_sat - gamma_w) * h_2} * cos^2(beta) * tan(phi')]
          / [(gamma_1 * h_1 + gamma_sat * h_2) * sin(beta) * cos(beta)]

Random variables: c' (effective cohesion) and phi' (effective friction angle)
Uses numpy for vectorized computation on large arrays.
"""

import numpy as np


def factor_of_safety(c_prime, phi_prime_deg, gamma_1, h_1, gamma_sat, gamma_w, h_2, beta_deg):
    """
    Calculate the Factor of Safety for an infinite slope with seepage.

    Parameters
    ----------
    c_prime : float or np.ndarray
        Effective cohesion (kPa).
    phi_prime_deg : float or np.ndarray
        Effective friction angle (degrees).
    gamma_1 : float
        Unit weight of soil above water table (kN/m^3).
    h_1 : float
        Thickness of soil layer above water table (m).
    gamma_sat : float
        Saturated unit weight of soil (kN/m^3).
    gamma_w : float
        Unit weight of water (kN/m^3).
    h_2 : float
        Thickness of saturated soil layer (m).
    beta_deg : float
        Slope angle (degrees).

    Returns
    -------
    fs : float or np.ndarray
        Factor of Safety.
    """
    beta_rad = np.radians(beta_deg)
    phi_prime_rad = np.radians(phi_prime_deg)

    numerator = c_prime + (gamma_1 * h_1 + (gamma_sat - gamma_w) * h_2) * (np.cos(beta_rad) ** 2) * np.tan(phi_prime_rad)
    denominator = (gamma_1 * h_1 + gamma_sat * h_2) * np.sin(beta_rad) * np.cos(beta_rad)

    return numerator / denominator


def run_simulation(n_simulations=100_000,
                   c_prime_mean=10.0, c_prime_std=3.0,
                   phi_prime_mean=25.0, phi_prime_std=5.0,
                   gamma_1=17.0, h_1=2.0,
                   gamma_sat=20.0, gamma_w=9.81,
                   h_2=1.0, beta_deg=30.0,
                   seed=None):
    """
    Run Monte Carlo simulation for the Factor of Safety.

    Parameters
    ----------
    n_simulations : int
        Number of Monte Carlo samples.
    c_prime_mean : float
        Mean of effective cohesion c' (kPa).
    c_prime_std : float
        Standard deviation of effective cohesion c' (kPa).
    phi_prime_mean : float
        Mean of effective friction angle phi' (degrees).
    phi_prime_std : float
        Standard deviation of effective friction angle phi' (degrees).
    gamma_1 : float
        Unit weight of soil above water table (kN/m^3).
    h_1 : float
        Thickness of soil layer above water table (m).
    gamma_sat : float
        Saturated unit weight of soil (kN/m^3).
    gamma_w : float
        Unit weight of water (kN/m^3).
    h_2 : float
        Thickness of saturated soil layer (m).
    beta_deg : float
        Slope angle (degrees).
    seed : int or None
        Random seed for reproducibility.

    Returns
    -------
    results : dict
        Dictionary containing:
        - fs_values: array of FS values
        - c_prime_samples: array of c' samples
        - phi_prime_samples: array of phi' samples
        - mean_fs: mean Factor of Safety
        - std_fs: standard deviation of Factor of Safety
        - prob_failure: probability of failure (FS < 1.0)
        - n_simulations: number of simulations
    """
    rng = np.random.default_rng(seed)

    # Generate random samples for c' and phi'
    c_prime_samples = rng.normal(c_prime_mean, c_prime_std, n_simulations)
    phi_prime_samples = rng.normal(phi_prime_mean, phi_prime_std, n_simulations)

    # Calculate Factor of Safety for all samples (vectorized)
    fs_values = factor_of_safety(
        c_prime_samples, phi_prime_samples,
        gamma_1, h_1, gamma_sat, gamma_w, h_2, beta_deg
    )

    mean_fs = np.mean(fs_values)
    std_fs = np.std(fs_values)
    prob_failure = np.mean(fs_values < 1.0)

    return {
        "fs_values": fs_values,
        "c_prime_samples": c_prime_samples,
        "phi_prime_samples": phi_prime_samples,
        "mean_fs": float(mean_fs),
        "std_fs": float(std_fs),
        "prob_failure": float(prob_failure),
        "n_simulations": n_simulations,
    }


def plot_results(results, save_path=None):
    """
    Plot the Monte Carlo simulation results.

    Parameters
    ----------
    results : dict
        Output from run_simulation().
    save_path : str or None
        If provided, save the figure to this path instead of showing it.
    """
    import matplotlib.pyplot as plt

    fs_values = results["fs_values"]

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # Histogram of Factor of Safety
    axes[0].hist(fs_values, bins=100, density=True, alpha=0.7, color="steelblue", edgecolor="black")
    axes[0].axvline(x=1.0, color="red", linestyle="--", linewidth=2, label="FS = 1.0")
    axes[0].axvline(x=results["mean_fs"], color="green", linestyle="-", linewidth=2, label=f"Mean FS = {results['mean_fs']:.3f}")
    axes[0].set_xlabel("Factor of Safety (FS)")
    axes[0].set_ylabel("Probability Density")
    axes[0].set_title("Monte Carlo Simulation: Factor of Safety Distribution")
    axes[0].legend()

    # Histogram of c' samples
    axes[1].hist(results["c_prime_samples"], bins=80, density=True, alpha=0.7, color="orange", edgecolor="black")
    axes[1].set_xlabel("c' (kPa)")
    axes[1].set_ylabel("Probability Density")
    axes[1].set_title("c' (Effective Cohesion) Distribution")

    # Histogram of phi' samples
    axes[2].hist(results["phi_prime_samples"], bins=80, density=True, alpha=0.7, color="green", edgecolor="black")
    axes[2].set_xlabel("φ' (degrees)")
    axes[2].set_ylabel("Probability Density")
    axes[2].set_title("φ' (Effective Friction Angle) Distribution")

    fig.suptitle(
        f"N = {results['n_simulations']:,} | "
        f"P(failure) = {results['prob_failure']:.4f} | "
        f"Mean FS = {results['mean_fs']:.3f} ± {results['std_fs']:.3f}",
        fontsize=12, y=1.02,
    )
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Figure saved to {save_path}")
    else:
        plt.show()

    plt.close(fig)


def main():
    """Run the Monte Carlo simulation with default parameters and print results."""
    print("=" * 60)
    print("Monte Carlo Simulation: Infinite Slope Factor of Safety")
    print("=" * 60)
    print()
    print("Equation (2.1):")
    print("  F_s = [c' + {γ₁h₁ + (γ_sat - γ_w)h₂} cos²β tan φ']")
    print("        / [(γ₁h₁ + γ_sat·h₂) sin β cos β]")
    print()

    # Default parameters
    params = {
        "n_simulations": 100_000,
        "c_prime_mean": 10.0,
        "c_prime_std": 3.0,
        "phi_prime_mean": 25.0,
        "phi_prime_std": 5.0,
        "gamma_1": 17.0,
        "h_1": 2.0,
        "gamma_sat": 20.0,
        "gamma_w": 9.81,
        "h_2": 1.0,
        "beta_deg": 30.0,
        "seed": 42,
    }

    print("Input Parameters:")
    print(f"  Number of simulations: {params['n_simulations']:,}")
    print(f"  c' ~ Normal(μ={params['c_prime_mean']} kPa, σ={params['c_prime_std']} kPa)")
    print(f"  φ' ~ Normal(μ={params['phi_prime_mean']}°, σ={params['phi_prime_std']}°)")
    print(f"  γ₁ = {params['gamma_1']} kN/m³")
    print(f"  h₁ = {params['h_1']} m")
    print(f"  γ_sat = {params['gamma_sat']} kN/m³")
    print(f"  γ_w = {params['gamma_w']} kN/m³")
    print(f"  h₂ = {params['h_2']} m")
    print(f"  β = {params['beta_deg']}°")
    print()

    results = run_simulation(**params)

    print("Results:")
    print(f"  Mean Factor of Safety: {results['mean_fs']:.4f}")
    print(f"  Std Dev of FS:         {results['std_fs']:.4f}")
    print(f"  P(failure) [FS < 1.0]: {results['prob_failure']:.4f} ({results['prob_failure'] * 100:.2f}%)")
    print()

    plot_results(results, save_path="monte_carlo_results.png")


if __name__ == "__main__":
    main()
