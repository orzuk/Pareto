# Pareto
Computing statistics and verifying correlation inequalities for Pareto-optimal vectors in the unit hypercube.

Companion code for the paper: *A correlation inequality for random points in a hypercube* by R. Jacobovic and O. Zuk ([arXiv:2209.00346](https://arxiv.org/abs/2209.00346)).

## Verifying the main inequality

The script `verify_inequality.py` checks the correlation inequality R(m,k) ≤ α(m,k) using exact rational arithmetic. Combined with the analytic proof for k ≥ 5 ln(m), this gives a rigorous verification for all k ≥ 1.

```bash
# Verify for all m ≤ 20 (default, ~1 second)
python verify_inequality.py

# Verify for all m ≤ 100 (~30 minutes)
python verify_inequality.py --max_m 100

# Verify for all m ≤ 50, all k ≤ 25 (fixed k range instead of 5*ln(m))
python verify_inequality.py --max_m 50 --max_k 25
```

Requires Python 3 (no external packages — uses only `fractions` for exact arithmetic).

## Other code

- `pareto.py` — exact computation of Pareto-optimal probabilities (E[Z₁Z₂], etc.)
- `PartialOrders.py` — verification of generalized DNF/CNF correlation inequalities
- `pareto_funcs.R` — R functions for Pareto statistics (older codebase)
