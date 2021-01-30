"""
# Data-driven Splashing Threshold Model (Pierzyna et al. 2020)

This example shows how to use the Python implementation of the
Data-driven Splashing Threshold (DST) model presented by Pierzyna et al. (2020)
based on the splashing model proposed by Riboux & Gordillo (2014; 2015).

## References
* Pierzyna, Maximilian, David A. Burzynski, Stephan E. Bansmer, and Richard Semaan.
  "Data-driven splashing threshold model for drop impact on dry smooth surfaces."
  Journal of Fluid Mechanics (2020, submitted)
* Riboux, Guillaume, and José Manuel Gordillo. "Experiments of drops impacting
  a smooth solid surface: a model of the critical impact speed for drop splashing."
  Physical review letters 113.2 (2014): 024507.
* Riboux, Guillaume, and José Manuel Gordillo. "The diameters and velocities of
  the droplets ejected after splashing." Journal of Fluid Mechanics 772 (2015): 630-648.
"""
import numpy as np
from rg_dst_umf import dst_calc_threshold, dst_predict_splashing
from rg_dst_umf import calc_beta

####
# Vector mode:
# Predict splashing for single measurement
####
# Drop 1: Water (10 m/s, R=2mm) in Air
x1 = np.array([
    10,         # V_0 in m/s
    2e-3,       # R_0 in m (Attention! Radius is used *not* diameter!)
    1000,       # rho_l in kg/m^3
    1e-3,       # mu_l in Pa s
    72e-3,      # sigma_l in N/m
    1.205,      # rho_g in kg/m^3
    1.8e-5,     # mu_g in Pa s
    68e-9,      # lambda_g in m
    np.pi/3,    # alpha in rad
])

# Optional: Calculate beta and threshold value of DST model manually
print("beta     = {:.4f}".format(calc_beta(x1)))
print("beta_DST = {:.4f}".format(dst_calc_threshold(x1)))

# Directly predit if drop will splash based on RG beta and DST model
print("Will splash?", dst_predict_splashing(x1))

####
# Matrix mode:
# Predict splashing for multiple measurements
####
# Drop 2: Water (5 m/s, R=2mm) in Air
x2 = np.array([
    5,
    2e-3,
    1000,
    1e-3,
    72e-3,
    1.205,
    1.8e-5,
    68e-9,
    np.pi/3,
])

# Drop 3: Water (5 m/s, R=2mm) in Helium
x3 = np.array([
    5,
    2e-3,
    1000,
    1e-3,
    72e-3,
    .22,
    2e-5,
    173e-9,
    np.pi/3,
])

# Bundle all three drops into matrix (numpy array)
# Note: You can also load this matrix from CSV file of measurements!
X = np.array([x1, x2, x3])

# Optional: Calculate beta and threshold value of DST model manually
print("beta     = ", calc_beta(X))
print("beta_DST = ", dst_calc_threshold(X))

# Directly predit if drop will splash based on RG beta and DST model
print("Will splash?", dst_predict_splashing(X))
