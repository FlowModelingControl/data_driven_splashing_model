import sympy as sp
import numpy as np
from scipy.optimize import fsolve

# create symbols
U0, R0, rho_l, mu_l, sigma_l, rho_g, mu_g, lambda_g, t_e, beta, f1, f2, f3 = sp.symbols("U0 R0 rho_l mu_l sigma_l rho_g mu_g lambda_g t_e beta f1 f2 f3")

#####
# Implement RG splashing model (Riboux & Gordillo 2014; Riboux & Gordillo 2015)
#####
H_t = R0 * sp.sqrt(12) * t_e**(3/2) / sp.pi
V_t = U0/2 * sp.sqrt(3/t_e)

alpha = 60 * sp.pi / 180
kl = -6/sp.tan(alpha)**2 * (sp.ln(19.2 * lambda_g / H_t) - sp.ln(1 + 19.2 * lambda_g / H_t))
ku = 0.3

beta_expr = sp.sqrt((kl * mu_g * V_t + ku * rho_g * H_t * V_t**2) / (2 * sigma_l))

# Ejection time. Use high Oh limit for mu > 10cp = 10e-3 Pa*s (Riboux & Gordillo 2015)
t_e_expr = sigma_l / (R0 * U0**2 * rho_l) + sp.sqrt(3) * mu_l / (2 * R0 * U0 * rho_l * sp.sqrt(t_e)) - 1.2 * t_e**(3/2)
t_e_high_Oh_expr = 2 * (rho_l * R0 * U0 / mu_l)**(-1/2)

#####
# Implement uncertainty magnification factors for RG splashing model (???)
#####
# Time derivative functions
# update constants
f1_expr = sp.sqrt(3)/7.2 * mu_l + R0 * t_e**2 * U0 * rho_l
f2_expr = sp.sqrt(3)/3.6 * t_e * U0 * mu_l
f3_expr = 2/3.6 * t_e**(3/2) * sigma_l

# Common functions
a1 = f1 * t_e * U0 * beta**2 * sigma_l * (R0 * t_e**(3/2) + 17.4125 * lambda_g)
a2 = 3/8 * sp.sqrt(3) * R0 * t_e * U0 * mu_g
a3 = R0 * sp.sqrt(t_e) * U0**2 * rho_g * (0.0620245 * R0 * t_e**(3/2) + 1.08 * lambda_g)
a4 = -1/4 * beta**2 * sigma_l * (R0 * t_e**(3/2) + 17.4125 * lambda_g)

umf_U0 = (
    a2 * (-f2 - 2*f3) +
    a3 * (f1 * t_e * U0 - f2 - 2*f3) -
    a4 * (2*f1 * t_e * U0 + f2 + 2*f3)
) / a1


umf_R0 = (
    a2 * (2/3 * f1 * t_e * U0 - f2 - f3) +
    a3 * (f1 * t_e * U0 - f2 - f3) -
    a4 * (f2 + f3)
) / a1

umf_rho_l = - (f2 + f3) / a1 * (a2 + a3 + a4)

umf_mu_l = f2 / a1 * (
    a2 + a3 + a4
)

umf_sigma_l = (
    a2 * f3 +
    a3 * f3 +
    a4 * (2 * t_e * U0 * f1 + f3)
) / a1

umf_rho_g = (0.0620245 * R0 * sp.sqrt(t_e) * U0**2 * rho_g) / (beta**2 * sigma_l)

umf_mu_g = 0.5 - umf_rho_g

umf_lambda_g = -(sp.sqrt(3) * R0 * t_e * U0 * mu_g) / (4 * beta**2 * sigma_l * (R0 * t_e**(3/2) + 17.4125 * lambda_g))


class RG2015_UMFs:
    """
    Calculate UMFs based on equations from ???.
    Provide drop impact condtions as vector or as matrix containing
    - V0:       Impact velocity in m/s
    - R0:       Drop radius in m
    - rho_l:    Liquid density in kg/m^3
    - mu_l:     Liquid viscosity in Pa s
    - sigma_l:  Liquid surface tension in N/m
    - rho_g:    Gas density in kg/m^3
    - mu_g:     Gas viscosity in Pa s
    - lambda_g: Gas mean free path in m
    """
    def __init__(self):
        umfs = {
            "U0": umf_U0,
            "R0": umf_R0,
            "rho_l": umf_rho_l,
            "mu_l": umf_mu_l,
            "sigma_l": umf_sigma_l,
            "rho_g": umf_rho_g,
            "mu_g": umf_mu_g,
            "lambda_g": umf_lambda_g,
        }

        self.state = [U0, R0, rho_l, mu_l, sigma_l, rho_g, mu_g, lambda_g]

        # Convert umf dict into compiled lambda expressions with numpy backend for fast evaluation
        self._umfs = {
            k: sp.lambdify(
                self.state + [t_e],
                umf.subs([
                    (f1, f1_expr),
                    (f2, f2_expr),
                    (f3, f3_expr),
                    (beta, beta_expr),
                ]),
                "numpy"
            )
            for k, umf in umfs.items()
        }

        self._beta = sp.lambdify(
            self.state + [t_e],
            beta_expr,
            "numpy"
        )

        self._t_e = sp.lambdify(
            [t_e, ] + self.state,
            t_e_expr,
            "numpy"
        )

        self._t_e_high_Oh = sp.lambdify(
            self.state,
            t_e_high_Oh_expr,
            "numpy"
        )

    def _solve_t_e(self, impact_vector):
        """
        Solve time equation numerically for given vector of impact conditions.

        :impact_vector: numpy.ndarry of shape (8, ) with drop impact conditions
                        (V0, R0, rho_l, mu_l, sigma_l, rho_g, mu_g, lambda_g)
        """
        # Array index of mu_l in state vector
        mu_l_idx = self.state.index(mu_l)

        # if mu_l > 10cp = 10e-3 Pa s, use high Oh limit
        if impact_vector[mu_l_idx] >= 10e-3:
            return self._t_e_high_Oh(*impact_vector)
        else:
            args = tuple(impact_vector.tolist())
            x0 = 1

            sol = fsolve(
                self._t_e,
                x0=x0,
                args=args,
            )

            return sol[0]

    def calc_t_e(self, impact_matrix):
        """
        Calculate ejection time (Riboux & Gordillo 2014) either
        for single impact vector or for matrix of impact vectors.
        This is not explicitly required to calculate the UMFs.

        :impact_matrix: numpy.ndarry of shape (8, ) or (n, 8) with drop impact
                        conditions (V0, R0, rho_l, mu_l, sigma_l, rho_g, mu_g, lambda_g)
        """
        if len(impact_matrix.shape) == 1:
            # 1D matrix, i.e. vector
            return self._solve_t_e(impact_matrix)
        elif len(impact_matrix.shape) == 2:
            # 2D matrix
            return np.array([
                self._solve_t_e(vec)
                for vec in impact_matrix
            ])
        else:
            raise KeyError("Wrong shape of array")

    def calc_beta(self, impact_matrix, t_e=None):
        """
        Calculate beta splashing factor (Riboux & Gordillo 2014)
        either for single impact vector or for matrix of impact vectors.
        This is not explicitly required to calculate the UMFs.

        :impact_matrix: numpy.ndarry of shape (8, ) or (n, 8) with drop impact
                        conditions (V0, R0, rho_l, mu_l, sigma_l, rho_g, mu_g, lambda_g)
        :t_e:           Optional: Precalculated vector of ejection time.
                        If left blank, t_e will be calculated automatically
        """
        # Calculate ejection times first if not given from previous calculation
        if t_e is None:
            t_e = self.calc_t_e(impact_matrix)

        if len(impact_matrix.shape) == 1:
            # 1D matrix, i.e. vector
            impact_matrix = np.r_[impact_matrix, t_e]
        elif len(impact_matrix.shape) == 2:
            # 2D matrix
            impact_matrix = np.c_[impact_matrix, t_e]
        else:
            raise KeyError("Wrong shape of array")

        return self._beta(*impact_matrix.T)

    def calc_umfs(self, impact_matrix, t_e=None):
        """
        Calculate UMF factors for RG splashing model (???)
        either for single impact vector or for matrix of impact vectors.

        :impact_matrix: numpy.ndarry of shape (8, ) or (n, 8) with drop impact
                        conditions (V0, R0, rho_l, mu_l, sigma_l, rho_g, mu_g, lambda_g)
        :t_e:           Optional: Precalculated vector of ejection time.
                        If left blank, t_e will be calculated automatically
        """
        # Calculate ejection times first if not given from previous calculation
        if t_e is None:
            t_e = self.calc_t_e(impact_matrix)

        if len(impact_matrix.shape) == 1:
            # 1D matrix, i.e. vector
            impact_matrix = np.r_[impact_matrix, t_e]
        elif len(impact_matrix.shape) == 2:
            # 2D matrix
            impact_matrix = np.c_[impact_matrix, t_e]
        else:
            raise KeyError("Wrong shape of array")

        return {
            k: umf(*impact_matrix.T)
            for k, umf in self._umfs.items()
        }

# References
# - Riboux, Guillaume, and José Manuel Gordillo. "Experiments of drops impacting a smooth solid surface: a model of the critical impact speed for drop splashing." Physical review letters 113.2 (2014): 024507.
# - Riboux, Guillaume, and José Manuel Gordillo. "The diameters and velocities of the droplets ejected after splashing." Journal of Fluid Mechanics 772 (2015): 630-648.
# - ???
