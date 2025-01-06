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
class Test_in_place:
    # ------------------------------------------------------------------
    def test_in_place(self):
        coeffs = np.ones((4,1), dtype = np.float64)
        diff.in_place(coeffs, 1)
        print(coeffs)
        assert np.all(coeffs == np.array(((3,), (2,), (1,), (1,)), dtype = np.float64))
        assert False
