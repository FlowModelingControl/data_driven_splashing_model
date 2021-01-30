import unittest
import numpy as np


class BaseTest(unittest.TestCase):
    """
    BaseTest class containing example drop impacts.
    Inherit this class to have same examples in all tests.
    """
    def setUp(self):
        """ Set up example drop impacts for testing """
        self.x_matrix = np.array([
            [10, 2e-3, 1000, 1e-3, 72e-3, 1.205, 1.8e-5, 68e-9, np.pi/3],        # Water in Air
            [5, 2e-3, 1000, 1e-3, 72e-3, 1.205, 1.8e-5, 68e-9, np.pi/3],         # Water in Air
            [10, 2e-3, 1000, 1e-3, 72e-3, 0.22, 2e-5, 173e-9, np.pi/3],          # Water in He
            [5, 2e-3, 1000, 1e-3, 72e-3, 0.22, 2e-5, 173e-9, np.pi/3],           # Water in He
            [10, 2e-3, 1150, 12.55e-3, 63e-3, 1.205, 1.8e-5, 68e-9, np.pi/3],    # 3:2 g/w in Air
            [5, 2e-3, 1150, 12.55e-3, 63e-3, 1.205, 1.8e-5, 68e-9, np.pi/3],     # 3:2 g/w in Air
        ])
        self.x_vector = self.x_matrix[0]

