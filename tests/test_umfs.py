import numpy as np

from tests.base_test import BaseTest
from models.rg_umfs import calc_umfs


class DSTModelTestCase(BaseTest):
    """
    Tests to ensure correct implementation of uncertainty magnification factors
    of the RG splashing model. Reference values where calculated with great care using
    analytical expressions and Mathematica.

    Test examples are found in `BaseTest` class.
    """
    def test_umfs(self):
        """ Test that UMF values are calculated correctly """
        # Test single drop impact
        np.testing.assert_array_almost_equal(
            np.array([
                0.6250379165,
                0.3040998342,
                -0.1436307931,
                0.08254305852,
                -0.4389122654,
                0.3297564441,
                0.1702435559,
                -0.1179741832
            ]),
            calc_umfs(self.x_vector)
        )

        # Test list of drop impacts
        np.testing.assert_array_almost_equal(
            np.array([
                [0.6250379164568434, 0.3040998342050456, -0.1436307930791466, 0.0825430585178547, -0.4389122654387081, 0.3297564440972821, 0.17024355590271792, -0.11797418318691001],
                [0.5488795956111323, 0.2640789180321515, -0.12167857924266112, 0.040285736263372715, -0.41860715702071155, 0.2519510178330818, 0.2480489821669182, -0.13380647944173074],
                [0.43829739648239757, 0.26709586411474395, -0.18149509793091737, 0.10430326372256558, -0.42280816579164826, 0.19698432862166657, 0.30301567137833335, -0.25160663342399464],
                [0.34858270617370946, 0.22236511139311566, -0.15925631400281878, 0.05272709382478582, -0.39347077982196693, 0.1143682403545613, 0.38563175964543855, -0.2672531850413732],
                # High Oh UMFs
                [0.8513100310346184, 0.39382478397996024, -0.023965968128786173, 0.02129762684395915, -0.49733165871517304, 0.37794434044823155, 0.12205565955176843, -0.03984641166051478],
                [0.7984944893767064, 0.3532489036538266, -0.02212059183003851, 0.017042226052714927, -0.49492163422267643, 0.32569344698406855, 0.17430655301593137, -0.04967604849979651],
            ]),
            calc_umfs(self.x_matrix)
        )
