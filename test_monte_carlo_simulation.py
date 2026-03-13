"""Tests for the Monte Carlo slope stability simulation."""

import numpy as np
import pytest

from monte_carlo_simulation import factor_of_safety, run_simulation


class TestFactorOfSafety:
    """Tests for the factor_of_safety function."""

    def test_scalar_inputs(self):
        """FS should return a scalar for scalar inputs."""
        fs = factor_of_safety(
            c_prime=10.0, phi_prime_deg=25.0,
            gamma_1=17.0, h_1=2.0,
            gamma_sat=20.0, gamma_w=9.81,
            h_2=1.0, beta_deg=30.0,
        )
        assert isinstance(fs, (float, np.floating))
        assert fs > 0

    def test_array_inputs(self):
        """FS should return an array for array inputs."""
        c = np.array([10.0, 12.0, 8.0])
        phi = np.array([25.0, 30.0, 20.0])
        fs = factor_of_safety(
            c_prime=c, phi_prime_deg=phi,
            gamma_1=17.0, h_1=2.0,
            gamma_sat=20.0, gamma_w=9.81,
            h_2=1.0, beta_deg=30.0,
        )
        assert isinstance(fs, np.ndarray)
        assert fs.shape == (3,)
        assert np.all(fs > 0)

    def test_higher_cohesion_increases_fs(self):
        """Increasing c' should increase FS, all else equal."""
        common = dict(
            phi_prime_deg=25.0, gamma_1=17.0, h_1=2.0,
            gamma_sat=20.0, gamma_w=9.81, h_2=1.0, beta_deg=30.0,
        )
        fs_low = factor_of_safety(c_prime=5.0, **common)
        fs_high = factor_of_safety(c_prime=15.0, **common)
        assert fs_high > fs_low

    def test_higher_friction_angle_increases_fs(self):
        """Increasing phi' should increase FS, all else equal."""
        common = dict(
            c_prime=10.0, gamma_1=17.0, h_1=2.0,
            gamma_sat=20.0, gamma_w=9.81, h_2=1.0, beta_deg=30.0,
        )
        fs_low = factor_of_safety(phi_prime_deg=15.0, **common)
        fs_high = factor_of_safety(phi_prime_deg=35.0, **common)
        assert fs_high > fs_low

    def test_known_value(self):
        """Verify FS against a hand-calculated value."""
        # c'=10, phi'=25°, gamma_1=17, h_1=2, gamma_sat=20, gamma_w=9.81, h_2=1, beta=30°
        beta_rad = np.radians(30.0)
        phi_rad = np.radians(25.0)
        numerator = 10.0 + (17.0 * 2.0 + (20.0 - 9.81) * 1.0) * np.cos(beta_rad) ** 2 * np.tan(phi_rad)
        denominator = (17.0 * 2.0 + 20.0 * 1.0) * np.sin(beta_rad) * np.cos(beta_rad)

        expected_fs = numerator / denominator
        computed_fs = factor_of_safety(
            c_prime=10.0, phi_prime_deg=25.0,
            gamma_1=17.0, h_1=2.0,
            gamma_sat=20.0, gamma_w=9.81,
            h_2=1.0, beta_deg=30.0,
        )
        assert abs(computed_fs - expected_fs) < 1e-10

    def test_zero_cohesion(self):
        """With c'=0, FS should still be positive for positive phi'."""
        fs = factor_of_safety(
            c_prime=0.0, phi_prime_deg=30.0,
            gamma_1=17.0, h_1=2.0,
            gamma_sat=20.0, gamma_w=9.81,
            h_2=1.0, beta_deg=30.0,
        )
        assert fs > 0


class TestRunSimulation:
    """Tests for the run_simulation function."""

    def test_returns_correct_keys(self):
        """Simulation result should contain all expected keys."""
        results = run_simulation(n_simulations=100, seed=42)
        expected_keys = {
            "fs_values", "c_prime_samples", "phi_prime_samples",
            "mean_fs", "std_fs", "prob_failure", "n_simulations",
        }
        assert set(results.keys()) == expected_keys

    def test_correct_number_of_samples(self):
        """Arrays should have the specified number of simulations."""
        n = 500
        results = run_simulation(n_simulations=n, seed=42)
        assert len(results["fs_values"]) == n
        assert len(results["c_prime_samples"]) == n
        assert len(results["phi_prime_samples"]) == n
        assert results["n_simulations"] == n

    def test_reproducibility_with_seed(self):
        """Same seed should produce identical results."""
        r1 = run_simulation(n_simulations=1000, seed=123)
        r2 = run_simulation(n_simulations=1000, seed=123)
        np.testing.assert_array_equal(r1["fs_values"], r2["fs_values"])
        assert r1["mean_fs"] == r2["mean_fs"]

    def test_different_seeds_produce_different_results(self):
        """Different seeds should produce different results."""
        r1 = run_simulation(n_simulations=1000, seed=1)
        r2 = run_simulation(n_simulations=1000, seed=2)
        assert not np.array_equal(r1["fs_values"], r2["fs_values"])

    def test_prob_failure_in_valid_range(self):
        """Probability of failure should be between 0 and 1."""
        results = run_simulation(n_simulations=1000, seed=42)
        assert 0.0 <= results["prob_failure"] <= 1.0

    def test_mean_fs_positive(self):
        """Mean FS should be positive under typical parameters."""
        results = run_simulation(n_simulations=1000, seed=42)
        assert results["mean_fs"] > 0

    def test_large_simulation_uses_numpy(self):
        """Large simulation should complete efficiently with numpy arrays."""
        results = run_simulation(n_simulations=100_000, seed=42)
        assert isinstance(results["fs_values"], np.ndarray)
        assert results["fs_values"].shape == (100_000,)
