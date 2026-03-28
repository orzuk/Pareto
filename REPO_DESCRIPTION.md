# Pareto Repository — Description and Connection to the Correlation Inequality

## Overview

This repository computes statistics for **Pareto-optimal vectors** (non-dominated points under the coordinatewise product order) among n i.i.d. uniform random vectors in [0,1]^k. The main focus is on computing exact probabilities and verifying correlation inequalities.

## Connection to the Correlation Inequality

The main open problem (in the companion research directory `NewProof/`) is to prove:

$$P(X_2 \in W \mid X_1 \in W) \leq P(X_2 \in W \mid X_1 \in S)$$

which is equivalent to the variance inequality:

$$\text{Var}(V) \leq \mathbb{E}[V](1 - \mathbb{E}[V]) \left(1 - \left(\frac{n-1}{n}\right)^k\right)$$

where $V = \text{Vol}(\bigcup_{j=3}^n [0, X_j])$ and $m = n - 2$.

### How this repo relates

| Quantity in the inequality | Function in this repo |
|---|---|
| $P(X_1 \text{ is Pareto-optimal}) = P(\overline{B_{1j}} \text{ for all } j)$ | `pareto_P(n, k)` (R), `pareto_P_Bj1c(k, n)` (Python) |
| $P(X_1, X_2 \text{ both Pareto-optimal}) = \mathbb{E}[Z_1 Z_2]$ | `pareto_E_Z1Z2(k, n)` (R), `pareto_E_Z1Z2_python(k, n)` (Python) |
| $P(X_1, X_2 \text{ both non-dominated by } X_{3:n})$ | `pareto_P_Bj1c_and_Bj2c_python(k, n)` (Python) |
| $\text{Var}(\text{number of Pareto maxima})$ | `pareto_P_var(k, n)` (R) |
| Correlation coefficient of Pareto indicators | `pareto_P_corr(k, n)` (R) |

Note: The repo also studies a **generalized DNF/CNF inequality** for partial orders (see `PartialOrders.py`), which is a broader class of inequalities that includes the main correlation inequality as a special case (the "matrix inequality").

## File Descriptions

### Python Files

| File | Purpose | Key Functions |
|---|---|---|
| `pareto.py` | Core exact probability computations using arbitrary precision (mpmath) | `pareto_E_Z1Z2_python(k, n)` — computes P(both X_1, X_2 are Pareto maxima) via multinomial sum; `pareto_P_Bj1c_and_Bj2c_python(k, n)` — P(both non-dominated by X_{3:n}); `pareto_P_Bj1c(k, n)` — P(X_1 non-dominated by X_{3:n}) |
| `PartialOrders.py` | Inequality verification for general partial order constraints (DNF vs CNF conditioning) | `check_CNF_DNF_inequality()` — test inequality for given constraint sets; `find_random_counter_example_DNF_CNF_inequality()` — random search for counterexamples; `find_enumerate_counter_example_DNF_CNF_inequality()` — exhaustive search; `check_matrix_CNF_DNF_inequality_combinatorics(n, k)` — verify the specific matrix inequality |
| `RunPartialOrders.py` | Test script that exercises the inequality checking functions | Runs various counterexample searches and verifications |

### R Files

| File | Purpose | Status |
|---|---|---|
| `pareto_funcs.R` | Comprehensive R library: simulations, exact formulas, variance computation, visualization helpers | **Mostly superseded by Python** for the core computations. The R code calls `pareto_E_Z1Z2_python()` for the hard multinomial sum. Contains useful functions like `pareto_P_mat()` (fast recursive computation) and `pareto_P_approx()` (asymptotics) that have no Python equivalent yet. |
| `run_pareto.R` | Setup script sourcing other files | Minimal |
| `make_p_pareto_plots.R` | Publication-quality plots: log-log scaling, phase transitions, variance analysis | R-specific visualization, could be converted to matplotlib |

### C++ File

| File | Purpose |
|---|---|
| `chrom_funcs.cpp` | C++ helper for fast Pareto front extraction (used by R via Rcpp) |

## Key Mathematical Formulas Implemented

### P(X_1 is Pareto-optimal among n points)
```
p_{k,n} = sum_{r=0}^{n-1} C(n-1, r) * (-1)^r / (r+1)^k
```
Implemented in `pareto_P()` (R) and `pareto_P_Bj1c()` (Python).

### P(X_1 AND X_2 are both Pareto-optimal)
```
E[Z_1 Z_2] = sum_{a+b+c+d=n-2} (-1)^{a+b} * C(n-2; a,b,c,d)
             * [(a+b+2c+2)^k - (a+c+1)^k - (b+c+1)^k]
             / [(a+c+1)(b+c+1)(a+b+c+2)]^k
```
Implemented in `pareto_E_Z1Z2_python()` (Python, arbitrary precision) and `pareto_E_Z1Z2()` (R, multiple methods).

### P(X_1, X_2 both non-dominated by X_{3:n}) = E[(1-V)^2]
```
E[(1-V)^2] = sum_{a+b+c+d=n-2} (-1)^{a+b} * C(n-2; a,b,c,d)
             * [(a+b+2c+2) / ((a+c+1)(b+c+1)(a+b+c+2))]^k
```
Implemented in `pareto_P_Bj1c_and_Bj2c_python()` (Python).

### Variance of number of Pareto maxima
```
Var(M_n) = n * p * (1-p) + n*(n-1) * (E[Z_1 Z_2] - p^2)
```
where `p = p_{k,n}`. Implemented in `pareto_P_var()` (R).

## Relationship Between R and Python Code

The R codebase is the **older, more complete** implementation with visualization and simulation capabilities. However, for the core heavy computation (`E[Z_1 Z_2]`), R calls Python via `pareto_E_Z1Z2_python()` because Python's mpmath provides better arbitrary precision arithmetic than R's options.

**Recommended path forward**: Use Python for all computations. The R functions that are still uniquely useful are:
- `pareto_P_mat()` — fast recursive matrix computation of p_{k,n}
- `pareto_P_approx()` — asymptotic approximation
- `pareto_P_sim()` — Monte Carlo simulation
- Plotting code in `make_p_pareto_plots.R`

These should be ported to Python for a unified codebase.
