"""
# Uncertainty magnification factors for RG splashing model

Equations for uncertainty magnification factors (UMFs) derived for the RG splashing
model (Riboux & Gordillo 2014; 2015) in Pierzyna et al. (2021).

## References
* Pierzyna, Maximilian, David A. Burzynski, Stephan E. Bansmer, and Richard Semaan.
  "Data-driven splashing threshold model for drop impact on dry smooth surfaces."
  Physical Review Fluids (2021, in review)
* Riboux, Guillaume, and José Manuel Gordillo. "Experiments of drops impacting
  a smooth solid surface: a model of the critical impact speed for drop splashing."
  Physical review letters 113.2 (2014): 024507.
* Riboux, Guillaume, and José Manuel Gordillo. "The diameters and velocities of
  the droplets ejected after splashing." Journal of Fluid Mechanics 772 (2015): 630-648.
"""
import sympy as sp
import numpy as np

from .rg2014 import calc_t_e, ll, ls
from ..tools.utils import validate_input

# create symbols
V0, R0, rho_l, mu_l, sigma_l, rho_g, mu_g, lambda_g, alpha, t_e, beta = sp.symbols("V0 R0 rho_l mu_l sigma_l rho_g mu_g lambda_g alpha t_e beta")
state = [V0, R0, rho_l, mu_l, sigma_l, rho_g, mu_g, lambda_g, alpha]

#####
# Implement uncertainty magnification factors for RG splashing model (TODO add reference to our manuscript)
#####
# Constants according to Riboux and Gordillo (2014, 2015)
c1 = sp.sqrt(3)/2
c2 = sp.sqrt(1.2)

# Define auxiliary functions
f1 = (19.2 * sp.pi * lambda_g) / (sp.sqrt(12) * R0 * t_e**(3/2))
f2 = c1 * mu_l + 3 * c2**2 * R0 * t_e**2 * V0 * rho_l
f3 = (c1 * V0 * mu_l + sp.sqrt(t_e) * sigma_l) / f2
f4 = c1 * V0 * mu_l + 2 * sp.sqrt(t_e) * sigma_l
f5 = sp.ln(f1) - sp.ln(1 + f1)

# Define UMF functions
umf_V0 = (
    3 * f4 * ll + (1 + f1) * (
        f4 * (ll - ls) + f2 * (ll + 2 * ls) * V0
    ) * f5
) / (
    2 * (1 + f1) * (ll + ls) * f5 * f2 * V0
)

umf_R0 = - (
    ll * (V0 - 3 * f3) - (
        f3 * (ll + f1 * ll - ls - f1 * ls) + (1 + f1) * ls * V0
    ) * f5
) / (
    2 * (1 + f1) * (ll + ls) * f5 * V0
)

umf_rho_l = (
    f3 * (
        3 * ll + (ll + f1 * ll - ls - f1 * ls) * f5
    )
) / (
    2 * (1 + f1) * (ll + ls) * f5 * V0
)

umf_mu_l = c1 / (2 * f2 * (ll + ls)) * (
    ls - ll * (
        1 + 3 / ((1 + f1) * f5)
    )
) * mu_l

umf_sigma_l = - (
    (ll + ls) * (1 + f1) * f2 * f5 * V0 + sp.sqrt(t_e) * sigma_l * (
        3 * ll + (ll + f1 * ll - ls - f1 * ls) * f5
    )
) / (
    2 * (1 + f1) * (ll + ls) * f5 * f2 * V0
)

umf_rho_g = ls / (2 * (ll + ls))

umf_mu_g = ll / (2 * (ll + ls))

umf_lambda_g = ll / (2 * (1 + f1) * (ll + ls) * f5)

# numpy does not define sec and csc -> use alternative representation: sec(x) csc(x) = 1/(sin(x) cos(x))
umf_alpha = - (
    ll * alpha / (sp.sin(alpha) * sp.cos(alpha))
) / (ll + ls)

# Substitute auxiliary functions f1, f2, f3, and RG's beta equation into UMF equations.
# Convert all UMF equations into array of python usabale functions.
umf_funcs = [
    sp.lambdify(
        state + [t_e],
        umf,
        "numpy"
    ) for umf in
    [umf_V0, umf_R0, umf_rho_l, umf_mu_l, umf_sigma_l, umf_rho_g, umf_mu_g, umf_lambda_g, umf_alpha]
]


def vectorize_constant(result, length):
    """
    If sympy expression simplifies to a constant, only this constant is returned regardless of dimensionality of input.
    E.g. Input: (n x m) matrix
         -> expected output: vector length m
         -> actual output: scalar constant
    This workaround will convert the constant scalar into a vector for further processing
    """
    if isinstance(result, (int, float)):
        return np.ones(length) * result
    return result


def calc_umfs(impact_matrix):
    """
    Calculate UMF factors for RG splashing model (Pierzyna et al. 2021)
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

        # Output has to be vectorized in case equations return only constant
        umfs = np.array([
            vectorize_constant(
                umf_f(*impact_matrix.T), impact_matrix.shape[0]
            )
            for umf_f in umf_funcs
        ]).T
    else:
        # 1D matrix, i.e. vector
        impact_matrix = np.r_[impact_matrix, t_e_result]

        umfs = np.array([
            umf_f(*impact_matrix.T)
            for umf_f in umf_funcs
        ]).T

    return umfs
