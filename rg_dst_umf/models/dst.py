"""
# Data-driven splashing threshold model

Implementation of the data-driven splashing threshold (DST) model
presented in Pierzyna et al. (2021) based on the splashing model proposed by
Riboux and Gordillo (2014; 2015).

## References
* Pierzyna, Maximilian, David A. Burzynski, Stephan E. Bansmer, and Richard Semaan.
  "Data-driven splashing threshold model for drop impact on dry smooth surfaces."
  International Journal of Multiphase Flow (2021, submitted)
* Riboux, Guillaume, and José Manuel Gordillo. "Experiments of drops impacting
  a smooth solid surface: a model of the critical impact speed for drop splashing."
  Physical review letters 113.2 (2014): 024507.
* Riboux, Guillaume, and José Manuel Gordillo. "The diameters and velocities of
  the droplets ejected after splashing." Journal of Fluid Mechanics 772 (2015): 630-648.
"""
import sympy as sp
from .rg2014 import calc_beta
from ..tools.utils import validate_input

# create symbols
V0, R0, rho_l, mu_l, sigma_l, rho_g, mu_g, lambda_g, alpha, t_e, beta = sp.symbols("V0 R0 rho_l mu_l sigma_l rho_g mu_g lambda_g alpha t_e beta")
state = [V0, R0, rho_l, mu_l, sigma_l, rho_g, mu_g, lambda_g, alpha]

# DST equation
c0 = 0.1081
c1 = 9.5836e-03
c2 = 1.7332e-05
c3 = -7.0521e-01
c4 = 1.661e-02

beta_dst_expr = c0 + c1 * V0 + c2 / mu_l + c3 * sigma_l + c4 * rho_g

# Transform equation into function usabale with python
beta_dst = sp.lambdify(
    state,
    beta_dst_expr,
    "numpy"
)


def dst_calc_threshold(X):
    """
    Calculates threshold value for `X` according to the data-driven splashing model
    :X: numpy array of shape (8, ) or (n, 8)
    """
    # Validate input and check if matrix or vector
    _, is_matrix = validate_input(X)

    if is_matrix:
        # Process matrix
        return beta_dst(*X.T)

    # Process single vector
    return beta_dst(*X)


def dst_predict_splashing(X):
    """
    Predict splashing according to RG splashing model and DST model.
    :X: numpy array of shape (8, ) or (n, 8)
    """
    return calc_beta(X) > dst_calc_threshold(X)
