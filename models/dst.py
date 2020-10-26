"""
Data-driven splashing threshold model
-----
Implementation of the data-driven splashing threshold (DST) model
presented in Pierzyna et al. (2020) based on the splashing model proposed by
Riboux and Gordillo (2014; 2015).
"""
import numpy as np
import sympy as sp
from models.rg2014 import calc_beta
from tools.utils import validate_input

# create symbols
U0, R0, rho_l, mu_l, sigma_l, rho_g, mu_g, lambda_g, t_e, beta = sp.symbols("U0 R0 rho_l mu_l sigma_l rho_g mu_g lambda_g t_e beta")
state = [U0, R0, rho_l, mu_l, sigma_l, rho_g, mu_g, lambda_g]

# DST equation
c0 = 0.1081
c1 = 9.6428e-03
c2 = 1.7455e-05
c3 = -7.0763e-01
c4 = 1.6841e-02

beta_dst_expr = c0 + c1 * U0 + c2 / mu_l + c3 * sigma_l + c4 * rho_g

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
