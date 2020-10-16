import numpy as np
from RG2015_UMFs import RG2015_UMFs

# Define two example drop impacts as numpy array
# Water in Air, V0 = 10 m/s, R0 = 1.5 mm
x1 = np.array([
    10,
    1.5e-3,
    1000,
    1e-3,
    72e-3,
    1.225,
    1.82e-5,
    68e-9
])

# Ethanol in Air, V0 = 3 m/s, R0 = 1 mm
x2 = np.array([
    3,
    1e-3,
    789.3,
    1.36e-3,
    21.97e-3,
    1.225,
    1.82e-5,
    68e-9
])

#####
# Initialize the calculator
#####
umfs = RG2015_UMFs()

#####
# Way 1: Evaluate single vector, i.e. one drop impact at a time.
#####
# Optional: Calculate ejection time
print(umfs.calc_t_e(x1))

# Optional: Calculate beta
print(umfs.calc_beta(x1))

# Calculate UMFs:
print(umfs.calc_umfs(x1))

#####
# Way 2: Evaluate single vector, i.e. one drop impact at a time.
#####
# Define matrix of impact conditions
x_matrix = np.array([x1, x2])

# Optional: Calculate ejection time
print(umfs.calc_t_e(x_matrix))

# Optional: Calculate beta
print(umfs.calc_beta(x_matrix))

# Calculate UMFs:
print(umfs.calc_umfs(x_matrix))


