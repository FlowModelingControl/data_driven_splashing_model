"""
# Riboux & Gordillo splashing model

Equations for RG splashing model (Riboux & Gordillo 2014; Riboux & Gordillo 2015)

## References
* Riboux, Guillaume, and José Manuel Gordillo. "Experiments of drops impacting
  a smooth solid surface: a model of the critical impact speed for drop splashing."
  Physical review letters 113.2 (2014): 024507.
* Riboux, Guillaume, and José Manuel Gordillo. "The diameters and velocities of
  the droplets ejected after splashing." Journal of Fluid Mechanics 772 (2015): 630-648.
"""
import sympy as sp
import numpy as np
from scipy.optimize import fsolve

from tools.utils import validate_input

# create symbols
V0, R0, rho_l, mu_l, sigma_l, rho_g, mu_g, lambda_g, t_e, beta = sp.symbols("V0 R0 rho_l mu_l sigma_l rho_g mu_g lambda_g t_e beta")
state = [V0, R0, rho_l, mu_l, sigma_l, rho_g, mu_g, lambda_g]

# RG equations
H_t = R0 * sp.sqrt(12) * t_e**(3/2) / sp.pi
V_t = V0/2 * sp.sqrt(3/t_e)

alpha = 60 * sp.pi / 180
kl = -6/sp.tan(alpha)**2 * (sp.ln(19.2 * lambda_g / H_t) - sp.ln(1 + 19.2 * lambda_g / H_t))
ku = 0.3

beta_expr = sp.sqrt((kl * mu_g * V_t + ku * rho_g * H_t * V_t**2) / (2 * sigma_l))

# Ejection time. Use high Oh limit for mu > 10cp = 10e-3 Pa*s (Riboux & Gordillo 2015)
t_e_expr = sigma_l / (R0 * V0**2 * rho_l) + sp.sqrt(3) * mu_l / (2 * R0 * V0 * rho_l * sp.sqrt(t_e)) - 1.2 * t_e**(3/2)
t_e_high_Oh_expr = 2 * (rho_l * R0 * V0 / mu_l)**(-1/2)

# Convert equations into python usable functions
beta_func = sp.lambdify(
    state + [t_e],
    beta_expr,
    "numpy"
)

t_e_func = sp.lambdify(
    [t_e, ] + state,
    t_e_expr,
    "numpy"
)

t_e_high_Oh_func = sp.lambdify(
    state,
    t_e_high_Oh_expr,
    "numpy"
)


def solve_t_e(impact_vector):
    """
    Solve time equation numerically for given vector of impact conditions.

    :impact_vector: numpy.ndarry of shape (8, ) with drop impact conditions
                    (V0, R0, rho_l, mu_l, sigma_l, rho_g, mu_g, lambda_g)
    """
    # Array index of mu_l in state vector
    mu_l_idx = state.index(mu_l)

    # if mu_l > 10cp = 10e-3 Pa s, use high Oh limit
    if impact_vector[mu_l_idx] >= 10e-3:
        return t_e_high_Oh_func(*impact_vector)
    else:
        args = tuple(impact_vector.tolist())
        x0 = 1

        sol = fsolve(
            t_e_func,
            x0=x0,
            args=args,
        )

        return sol[0]


def calc_t_e(impact_matrix):
    """
    Calculate ejection time (Riboux & Gordillo 2014) either
    for single impact vector or for matrix of impact vectors.

    :impact_matrix: numpy.ndarry of shape (8, ) or (n, 8) with drop impact
                    conditions (V0, R0, rho_l, mu_l, sigma_l, rho_g, mu_g, lambda_g)
    """
    # Validate input and check if matrix or vector
    _, is_matrix = validate_input(impact_matrix)

    if is_matrix:
        # 2D matrix
        return np.array([
            solve_t_e(vec)
            for vec in impact_matrix
        ])

    # 1D matrix, i.e. vector
    return solve_t_e(impact_matrix)


def calc_beta(impact_matrix):
    """
    Calculate beta splashing factor (Riboux & Gordillo 2014)
    either for single impact vector or for matrix of impact vectors.

    :impact_matrix: numpy.ndarry of shape (8, ) or (n, 8) with drop impact
                    conditions (V0, R0, rho_l, mu_l, sigma_l, rho_g, mu_g, lambda_g)
    """
    # Validate input and check if matrix or vector
    _, is_matrix = validate_input(impact_matrix)

    # Calculate ejection times first
    t_e_result = calc_t_e(impact_matrix)

    if is_matrix:
        # 2D matrix
        impact_matrix = np.c_[impact_matrix, t_e_result]
    else:
        # 1D matrix, i.e. vector
        impact_matrix = np.r_[impact_matrix, t_e_result]

    return beta_func(*impact_matrix.T)
