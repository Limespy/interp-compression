import numpy as np
import pytest
from limesqueezer.poly import diff
# ======================================================================
parametrize = pytest.mark.parametrize

class Test_to_out:
    @parametrize(('coeffs_shape',), (((1,),),
                                     ((2,),),
                                     ((2,2),),
                                     ((2,2,2),)))
    def test_shape(self, coeffs_shape: tuple [int, ...]):
        shape_out = list(coeffs_shape)
        shape_out[0] -= 1
        out = np.zeros(shape_out, dtype = np.float64)
        diff.to_out(np.ones(coeffs_shape, dtype = np.float64), out)
# ======================================================================

class Test_in_place_coeffs_vars:
    # ------------------------------------------------------------------
    def test_1d(self):
        coeffs = np.ones((4,1), dtype = np.float64)
        diff.in_place_coeffs_vars(coeffs, 3)
        print(coeffs)
        assert np.all(coeffs == np.array(((3,), (2,), (1,), (1,)), dtype = np.float64))
    # ------------------------------------------------------------------
    def test_2d(self):
        coeffs = np.ones((4,2), dtype = np.float64)
        diff.in_place_coeffs_vars(coeffs, 3)
        print(coeffs)
        assert np.all(coeffs == np.array(((3, 3), (2, 2), (1, 1), (1, 1)),
                                         dtype = np.float64))
# ======================================================================
class Test_in_place_vars_coeffs:
    # ------------------------------------------------------------------
    def test_1d(self):
        coeffs = np.ones((1,4), dtype = np.float64)
        diff.in_place_vars_coeffs(coeffs, 3)
        print(coeffs)
        assert np.all(coeffs == np.array(((3, 2, 1, 1,)), dtype = np.float64))
    # ------------------------------------------------------------------
    def test_2d(self):
        coeffs = np.ones((2,4), dtype = np.float64)
        diff.in_place_vars_coeffs(coeffs, 3)
        print(coeffs)
        assert np.all(coeffs == np.array(((3, 2, 1, 1),
                                          (3, 2, 1, 1)), dtype = np.float64))
    # ------------------------------------------------------------------
