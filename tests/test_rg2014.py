import numpy as np

from tests.base_test import BaseTest
from models.rg2014 import calc_beta, calc_t_e


class RG2014TestCase(BaseTest):
    """
    Tests to ensure correct implementation of RG splashing model.
    Reference values where calculated with great care using
    analytical expressions and Mathematica.

    Test examples are found in `BaseTest` class.
    """
    def test_t_e(self):
        """ Test that ejection time is calculated correctly and respects high Oh limit """
        # Test single drop impact
        np.testing.assert_almost_equal(calc_t_e(self.x_vector), 0.00792398)

        # Test list of drop impacts
        np.testing.assert_array_almost_equal(
            calc_t_e(self.x_matrix),
            np.array([
                0.00792398,
                0.0147641,
                0.00792398,
                0.0147641,
                0.0467184,  # High Oh limit
                0.0660698,  # High Oh limit
            ])
        )

    def test_beta(self):
        """ Test that splashing factor beta is calculated correctly """
        # Test single drop impact
        np.testing.assert_almost_equal(calc_beta(self.x_vector), 0.2367353239)

        # Test list of drop impacts
        np.testing.assert_array_almost_equal(
            calc_beta(self.x_matrix),
            np.array([
                0.2367353239,
                0.158211258,
                0.1308765051,
                0.1003368678,
                0.3683646504,
                0.216364554,
            ])
        )
