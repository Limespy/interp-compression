import numpy as np
import pytest
from limesqueezer.poly import diff
from limesqueezer.poly import interpolate
from limesqueezer.poly import make
# ======================================================================
parametrize = pytest.mark.parametrize
# ======================================================================
def make_ya_yb(Dx: float, coeffs):
    coeffs = coeffs.copy()
    n_coeffs = coeffs.shape[1]
    ya = np.empty((n_coeffs // 2, coeffs.shape[0]), np.float64)
    yb = ya.copy()

    for i_diff in range(len(ya)):
        n_coeffs -= 1
        ya[i_diff, 0] = coeffs[0, n_coeffs]
        yb[i_diff, 0] = interpolate.single(Dx, coeffs[0], n_coeffs)
        diff.in_place_vars_coeffs(coeffs, n_coeffs)
    return ya, yb
# ----------------------------------------------------------------------
def test_make_ya_yb():
    xa = 0.
    xb = 1.
    coeffs = np.ones((1, 6), np.float64)
    ya, yb = make_ya_yb(xb - xa, coeffs)
    print(ya, yb)
    assert ya.shape == (3, 1)
    assert np.all(ya == np.array(((1.,), (1.,), (2.,))))
    assert np.all(yb == np.array(((6.,), (15.,), (40.,))))
# ======================================================================
class Test_individuals:
    def test_make7(self):
        xa = 0.
        xb = 1.
        Dx = xb - xa
        coeffs = np.ones((1, 8), np.float64)
        ya, yb = make_ya_yb(Dx, coeffs)

        print(ya[:, 0], yb[:, 0])
        coeffs_out = coeffs.copy()

        make.make7(xb-xa, ya, yb, coeffs_out)
        print(coeffs[0], coeffs_out[0])
        assert np.all(coeffs == coeffs_out)

    def test_make7o(self):
        xa = 0.
        xb = 1.
        Dx = xb - xa
        coeffs = np.ones((8, 1), np.float64)
        ya, yb = make_ya_yb(Dx, coeffs.T)
        print(ya[:, 0], yb[:, 0])
        coeffs_out = coeffs.copy()
        make.omake7(Dx, ya, yb, coeffs_out)
        print(coeffs[:, 0], coeffs_out[:, 0])
        assert np.all(coeffs == coeffs_out)
