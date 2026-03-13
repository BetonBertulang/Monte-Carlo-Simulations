# Monte Carlo Simulations

Monte Carlo simulation for the **Factor of Safety** of an infinite slope with seepage (Eq. 2.1).

## Equation

```
F_s = [c' + {γ₁h₁ + (γ_sat - γ_w)h₂} cos²β tan φ'] / [(γ₁h₁ + γ_sat·h₂) sin β cos β]
```

Where:
- **c'** — effective cohesion (kPa) *(random variable)*
- **φ'** — effective friction angle (degrees) *(random variable)*
- **γ₁** — unit weight of soil above water table (kN/m³)
- **h₁** — thickness of soil layer above water table (m)
- **γ_sat** — saturated unit weight of soil (kN/m³)
- **γ_w** — unit weight of water (kN/m³)
- **h₂** — thickness of saturated soil layer (m)
- **β** — slope angle (degrees)

## Requirements

- Python 3.8+
- numpy
- matplotlib (for plotting)

Install dependencies:

```bash
pip install numpy matplotlib
```

## Usage

Run the simulation with default parameters:

```bash
python monte_carlo_simulation.py
```

Or use as a library:

```python
from monte_carlo_simulation import run_simulation, plot_results

results = run_simulation(
    n_simulations=100_000,
    c_prime_mean=10.0, c_prime_std=3.0,    # c' distribution (kPa)
    phi_prime_mean=25.0, phi_prime_std=5.0, # φ' distribution (degrees)
    gamma_1=17.0, h_1=2.0,
    gamma_sat=20.0, gamma_w=9.81,
    h_2=1.0, beta_deg=30.0,
    seed=42,
)

print(f"Mean FS: {results['mean_fs']:.4f}")
print(f"P(failure): {results['prob_failure']:.4f}")

plot_results(results, save_path="monte_carlo_results.png")
```

## Testing

```bash
pip install pytest
python -m pytest test_monte_carlo_simulation.py -v
```