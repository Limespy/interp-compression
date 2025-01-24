import numpy as np
import pytest
from limesqueezer.poly import interpolate
# ======================================================================
parametrize = pytest.mark.parametrize
# ======================================================================
class Test_single:
    def test_single(self):
        coeffs = np.arange(1., 5., dtype = np.float64)
        x = 2.
        result = coeffs[0] * x**2 + coeffs[1] * x + coeffs[2]
        assert coeffs[0] == 1.
        assert interpolate.single(x, coeffs, 2) == result
# ======================================================================
