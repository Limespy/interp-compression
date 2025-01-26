import numpy as np
import pytest
from limesqueezer.poly import diff
from limesqueezer.poly import interpolate
from limesqueezer.poly import make
from limesqueezer.poly._experimental import omake7
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
    print(ya, yb)
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
    def test_make1(self):
        xa = 0.
        xb = 1.
        Dx = xb - xa
        coeffs = np.ones((1, 2), np.float64)
        ya, yb = make_ya_yb(Dx, coeffs)

        print(ya[:, 0], yb[:, 0])
        coeffs_out = coeffs.copy()

        make.make1_64(Dx, ya, yb, coeffs_out)
        print(coeffs[0], coeffs_out[0])
        assert np.all(coeffs == coeffs_out)
    # ------------------------------------------------------------------
    def test_make3(self):
        xa = 0.
        xb = 1.
        Dx = xb - xa
        coeffs = np.ones((1, 4), np.float64)
        ya, yb = make_ya_yb(Dx, coeffs)

        print(ya[:, 0], yb[:, 0])
        coeffs_out = coeffs.copy()

        make.make3_64(Dx, ya, yb, coeffs_out)
        print(coeffs[0], coeffs_out[0])
        assert np.all(coeffs == coeffs_out)
    # ------------------------------------------------------------------
    def test_make5(self):
        xa = 0.
        xb = 1.
        Dx = xb - xa
        coeffs = np.ones((1, 6), np.float64)
        ya, yb = make_ya_yb(Dx, coeffs)

        print(ya[:, 0], yb[:, 0])
        coeffs_out = coeffs.copy()

        make.make5_64(Dx, ya, yb, coeffs_out)
        print(coeffs[0], coeffs_out[0])
        assert np.all(coeffs == coeffs_out)
    # ------------------------------------------------------------------
    def test_make7(self):
        xa = 0.
        xb = 1.
        Dx = xb - xa
        coeffs = np.ones((1, 8), np.float64)
        ya, yb = make_ya_yb(Dx, coeffs)

        print(ya[:, 0], yb[:, 0])
        coeffs_out = coeffs.copy()

        make.make7_64(Dx, ya, yb, coeffs_out)
        print(coeffs[0], coeffs_out[0])
        assert np.all(coeffs == coeffs_out)
    # ------------------------------------------------------------------
    def test_make7o(self):
        xa = 0.
        xb = 1.
        Dx = xb - xa
        coeffs = np.ones((8, 1), np.float64)
        ya, yb = make_ya_yb(Dx, coeffs.T)
        print(ya[:, 0], yb[:, 0])
        coeffs_out = coeffs.copy()
        omake7(Dx, ya, yb, coeffs_out)
        print(coeffs[:, 0], coeffs_out[:, 0])
        assert np.all(coeffs == coeffs_out)
