import numpy as np
from tools.taylor_propagator import TaylorPropagator
from models.rg2014 import calc_beta

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
    68e-9       # lambda_g in m
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
    68e-9
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
    173e-9
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
