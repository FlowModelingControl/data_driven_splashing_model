"""
# Uncertainty quantifier for Riboux & Gordillo (2014; 2015) splashing model

This example shows how to use the Python implementation of the
uncertainty quantification method for the RG splashing model presented by
Pierzyna et al. (2021).

The propagator can be used to quantify the influence of uncertainties in the measured
drop impact parameters (impact velocity, impact diameter, liquid properties, and gas properties)
on the calculated splashing factor `beta` according to the RG theory.

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
import numpy as np
from rg_dst_umf.tools.taylor_propagator import TaylorPropagator
from rg_dst_umf import calc_beta

####
# Initialize two uncertainty propagators with different uncertainties
# representing e.g. different experimental setups.
####
experiment1 = {
    # Absolute uncertainty for V0 of 0.15 m/s.
    "V0": 0.15
}
uq1 = TaylorPropagator(
    experiment1,
    update_defaults=True    # Keep default uncertainties of 1% for liquid and gas parameters.
)

experiment2 = {
    # Absolute uncertainty for V0 of 0.15 m/s.
    "V0": 0.15,
    # Relative uncertainty of 5% for R0. Function will be applied to R0 column of incoming data.
    "R0": lambda val: val * 0.05,
}
uq2 = TaylorPropagator(
    experiment2,
    update_defaults=False   # DON'T keep default uncertainties of 1% for liquid and gas parameters.
)

####
# Vector mode:
# Propagate uncertainties of single measurement
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

# Calculate beta for reference. Not required.
print("For reference, beta(x1) = {:.3f}".format(calc_beta(x1)))
print("")

# Calculate absolute and relative (optional) uncertainties for both experimental setups
print("Absolute combined uncertainty:")
print("u_beta = {:.5f}".format(uq1.calc_beta_uncertainty(x1)))
print("u_beta = {:.5f}".format(uq2.calc_beta_uncertainty(x1)))
print("")

print("Optional: Relative combined uncertainty:")
print("u_beta/beta = {:.2f}%".format(uq1.calc_beta_uncertainty(x1, relative=True) * 100))
print("u_beta/beta = {:.2f}%".format(uq2.calc_beta_uncertainty(x1, relative=True) * 100))

print("")
print("-----")
print("")

####
# Matrix mode:
# Propagate uncertainties of matrix of measurements
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
    np.pi/3
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
    np.pi/3
])

# Bundle all three drops into matrix (numpy array)
# Note: You can also load this matrix from CSV file of measurements!
X = np.array([x1, x2, x3])

# Calculate beta for reference. Not required.
print("For reference, beta(X) = ", calc_beta(X))
print("")

# Calculate absolute and relative (optional) uncertainties for both experimental setups
print("Absolute combined uncertainty:")
print("u_beta = ", uq1.calc_beta_uncertainty(X))
print("u_beta = ", uq2.calc_beta_uncertainty(X))
print("")

print("Optional: Relative combined uncertainty in %:")
print("u_beta/beta = ", uq1.calc_beta_uncertainty(X, relative=True) * 100)
print("u_beta/beta = ", uq2.calc_beta_uncertainty(X, relative=True) * 100)
