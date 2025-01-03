import numpy as np
import pytest
from limesqueezer import poly
# ======================================================================
parametrize = pytest.mark.parametrize

class Test_diff:
    @parametrize(('coeffs_shape',), (((1,),),
                                     ((2,),),
                                     ((2,2),),
                                     ((2,2,2),)))
    def test_shape(self, coeffs_shape: tuple [int, ...]):
        shape_out = list(coeffs_shape)
        shape_out[0] -= 1
        out = np.zeros(shape_out, dtype = np.float64)
        poly.diff(np.ones(coeffs_shape, dtype = np.float64), out)

class Test_make:
    def test_shape(self, y_shape: tuple [int, ...]) -> None:

        poly.makers[]
