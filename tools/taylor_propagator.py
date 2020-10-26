import numpy as np

from tools.utils import validate_input

from models.rg_umfs import calc_umfs
from models.rg2014 import calc_beta


class TaylorPropagator:
    """
    Propagate combined uncertainties through the RG model
    based on the Taylor expansion method.
    """
    VALID_VARIABLES = (
        "V0", "R0", "rho_l", "mu_l", "sigma_l", "rho_g", "mu_g", "lambda_g"
    )

    # Assume a relative uncertainty of 1% for all liquid parameters (???).
    DEFAULT_UNCERTAINTIES = {
        "rho_l": lambda val: val * 0.01,
        "mu_l": lambda val: val * 0.01,
        "sigma_l": lambda val: val * 0.01,
        "rho_g": lambda val: val * 0.01,
        "mu_g": lambda val: val * 0.01,
        "lambda_g": lambda val: val * 0.01,
    }

    def __init__(self, uncertainty_dict=None, update_defaults=False):
        """
        Initialize propagator with combined measurment uncertainties of experiment.

        :uncertainty_dict:  Dict with variables and corresponding uncertainties.
                            E.g.: {"V0": 0.15}
        :update_defaults:   True: default uncertainties (1% for liquid and gas)
                            will be kept and updated with provided `uncertainty_dict`.
                            False: Only use provided `uncertainty_dict`.
        """
        if uncertainty_dict is None:
            # Use default uncertainties for propagation
            self.uncertainty_dict = self.DEFAULT_UNCERTAINTIES
        else:
            if update_defaults:
                # Keep default uncertainties but update them provided uncertainty dict
                self.uncertainty_dict = self.DEFAULT_UNCERTAINTIES
                self.uncertainty_dict.update(uncertainty_dict)
            else:
                # Only use provided uncertainty dict (default)
                self.uncertainty_dict = uncertainty_dict

        # Check that all variables for which uncertainties are provided are valid.
        for var in uncertainty_dict:
            if var not in self.VALID_VARIABLES:
                raise ValueError(
                    f"Variable '{var}' is invalid. Please provide uncertainty "
                    f"for {self.VALID_VARIABLES}"
                )

    def calc_beta_uncertainty(self, X, relative=False):
        """
        Propagate uncertainties according to Eq. (2.2) shown in Pierzyna et al. (2020)
        to obtain combined uncertainty of splashing factor beta for provided state X.

        :relative:  Optional. If true, resulting uncertainty will be relative to `beta`.
        """
        # Validate input and check if matrix or vector
        _, is_matrix = validate_input(X)

        # Calculated squared UMFs
        umfs_sq = np.square(calc_umfs(X))

        # Result variable for squared propagated uncertainty of beta
        if is_matrix:
            u_beta_sq = np.zeros(X.shape[0])
        else:
            u_beta_sq = 0

        # Apply uncertainty of every valid variable
        for (i, var) in enumerate(self.VALID_VARIABLES):
            if var in self.uncertainty_dict:
                # Get uncertainty corresponding to variable `var`
                u_var = self.uncertainty_dict[var]

                # Select column from input `X` corresponding to `var`
                if is_matrix:
                    # Matrix
                    x_i = X[:, i]
                    umf_i_sq = umfs_sq[:, i]
                else:
                    # Vector
                    x_i = X[i]
                    umf_i_sq = umfs_sq[i]

                if callable(u_var):
                    # Relative uncertainty
                    # Appy function `u_var` to respective measurements `x_i`
                    u_beta_sq += umf_i_sq * np.square(u_var(x_i) / x_i)
                else:
                    # Absolute uncertainty
                    u_beta_sq += umf_i_sq * np.square(u_var / x_i)

        if relative:
            # Calculation of `beta` for relative result not required.
            return np.sqrt(u_beta_sq)

        # Calculate `beta` for state `X` and multiply with propagation result.
        return np.sqrt(u_beta_sq) * calc_beta(X)
