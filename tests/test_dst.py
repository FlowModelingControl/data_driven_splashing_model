import numpy as np

from tests.base_test import BaseTest
from rg_dst_umf import dst_calc_threshold, dst_predict_splashing


class DSTModelTestCase(BaseTest):
    """
    Tests to ensure correct implementation of DST model.
    Reference values where calculated with great care using
    analytical expressions and Mathematica.

    Test examples are found in `BaseTest` class.
    """
    def test_dst_threshold(self):
        """ Test that threshold value according to DST model is calculated correctly """
        # Test single drop impact
        np.testing.assert_almost_equal(dst_calc_threshold(self.x_vector), 0.191327045)

        # Test list of drop impacts
        np.testing.assert_array_almost_equal(
            dst_calc_threshold(self.x_matrix),
            np.array([
                0.191327045,
                0.143113045,
                0.17473866,
                0.12652466,
                0.1816315517,  # High Oh limit
                0.1334175517,  # High Oh limit
            ])
        )

    def test_dst_predict(self):
        """ Test that DST model predicts splashing (True or False) correctly """
        # Test single drop impact
        np.testing.assert_equal(dst_predict_splashing(self.x_vector), True)

        # Test list of drop impacts
        np.testing.assert_equal(
            dst_predict_splashing(self.x_matrix),
            np.array([
                True,
                True,
                False,
                False,
                True,  # High Oh limit
                True,  # High Oh limit
            ])
        )
