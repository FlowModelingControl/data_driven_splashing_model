import unittest
import numpy as np

from tests.base_test import BaseTest
from tools.taylor_propagator import TaylorPropagator
from models.rg2014 import calc_beta


class UQFunctionalTestCase(unittest.TestCase):
    """
    Test functionality of `TaylorPropagator` but not calculations
    """
    def test_fail_on_invalid_var(self):
        """ Test that propagator fails on invalid variable name """
        with self.assertRaises(ValueError):
            _ = TaylorPropagator({"invalid": 0})

    def test_update_defaults(self):
        """ Tests that passed uncertainties gets added to defaults """
        uq = TaylorPropagator({
            "V0": 0.15,
            "R0": 0.5e-3
        }, update_defaults=True)

        # New uncertainties added
        self.assertIn("V0", uq.uncertainty_dict)
        self.assertIn("R0", uq.uncertainty_dict)

        # Old uncertainties remain
        self.assertIn("rho_l", uq.uncertainty_dict)
        self.assertIn("mu_l", uq.uncertainty_dict)
        self.assertIn("sigma_l", uq.uncertainty_dict)
        self.assertIn("rho_g", uq.uncertainty_dict)
        self.assertIn("mu_g", uq.uncertainty_dict)
        self.assertIn("lambda_g", uq.uncertainty_dict)

    def test_discard_defaults(self):
        """ Test that default uncertainties are discarded by default """
        uq = TaylorPropagator({
            "V0": 0.15,
            "R0": 0.5e-3
        })

        # Only new uncertainties should be in uncertainty_dict
        self.assertListEqual(["V0", "R0"], list(uq.uncertainty_dict.keys()))


class UQCalculationTest(BaseTest):
    """
    Tests calculations of uncertainty quantifier. No functionality tests are performed.
    Also, UMFs are assumed to be correct since they are tested in another TestCase.
    """
    def test_absolute_u_relative_result(self):
        """ Test to propagate absolute uncertainty and obtain relative result (u_beta/beta) """
        # Only one uncertainty for V0
        uq = TaylorPropagator({
            "V0": 0.1
        }, update_defaults=False)

        # Calculated with Mathematica
        umf_V0 = 0.6250379165

        V_0 = self.x_vector[0]
        u_V_0 = 0.1

        np.testing.assert_almost_equal(
            uq.calc_beta_uncertainty(self.x_vector, relative=True),
            np.sqrt(
                np.square(umf_V0) * np.square(u_V_0 / V_0)
            )
        )

    def test_relative_u_relative_result(self):
        """ Test to propagate relative uncertainty and obtain relative result (u_beta/beta) """
        # Only one uncertainty for V0
        uq = TaylorPropagator({
            "V0": lambda val: val * 0.05
        }, update_defaults=False)

        # Calculated with Mathematica
        umf_V0 = 0.6250379165

        np.testing.assert_almost_equal(
            uq.calc_beta_uncertainty(self.x_vector, relative=True),
            np.sqrt(
                np.square(umf_V0) * np.square(0.05)
            )
        )

    def test_multiple_u_relative_result(self):
        """ Test to propagate two uncertainties and obtain relative result (u_beta/beta) """
        # Two uncertainties, relative for simplicity
        uq = TaylorPropagator({
            "V0": lambda val: val * 0.05,
            "R0": lambda val: val * 0.01
        }, update_defaults=False)

        # Calculated with Mathematica
        umf_V0 = 0.6250379165
        umf_R0 = 0.3040998342

        np.testing.assert_almost_equal(
            uq.calc_beta_uncertainty(self.x_vector, relative=True),
            np.sqrt(
                np.square(umf_V0) * np.square(0.05) +
                np.square(umf_R0) * np.square(0.01)
            )
        )

    def test_relative_u_absolute_result(self):
        """ Test to propagate relative uncertainty and obtain absolute result (u_beta) """
        # Only one uncertainty for V0
        uq = TaylorPropagator({
            "V0": lambda val: val * 0.05,
        }, update_defaults=False)

        # Calculated with Mathematica
        umf_V0 = 0.6250379165

        # Calculated beta for example drop impact
        beta = calc_beta(self.x_vector)

        np.testing.assert_almost_equal(
            # Default is absolute result
            uq.calc_beta_uncertainty(self.x_vector, relative=False),
            # Multiple relative result with beta to obtain absolute result
            np.sqrt(
                np.square(umf_V0) * np.square(0.05)
            ) * beta
        )
