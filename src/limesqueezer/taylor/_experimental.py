from typing import Literal as L

from .. import _lnumba as nb
from .._lnumpy import F64Array
from .._types import N_Diffs
from .._types import N_Vars
from .._types import XSingle
from .sequential import _Sequential
# ======================================================================
@nb.jitclass
@nb.clean
class Diff3Direct(_Sequential[L[4], N_Vars]):
    # ------------------------------------------------------------------
    def _prepare_coeffs(self, ya: F64Array[N_Diffs, N_Vars]) -> None:
        coeffs = self.coeffs
        for i in range(ya.shape[1]):
            coeffs[i, 4] = ya[3, i]
            coeffs[i, 5] = ya[2, i]
            coeffs[i, 6] = ya[1, i]
            coeffs[i, 7] = ya[0, i]
    # ------------------------------------------------------------------
    def _finish_coeffs(self, Dx: XSingle, yb: F64Array[N_Diffs, N_Vars]
                       ) -> None:
        coeffs = self.coeffs
        _1_6 = 0.16666666666666666666666666666666666666666666667
        Dx2 = Dx * Dx
        Dx3 = Dx2 * Dx

        _Dx1 = 1./Dx
        _Dx2 = _Dx1 * _Dx1

        _Dx4 = _Dx2 * _Dx2
        _Dx5 = _Dx4 * _Dx1
        _Dx6 = _Dx5 * _Dx1
        _Dx7 = _Dx6 * _Dx1

        a00 = a01 = a02 = a03 = _Dx7
        a00 *= -20.
        a01 *= 10.
        a02 *= -2.
        a03 *= _1_6

        a10 = a11 = a12 = a13 = _Dx6
        a10 *= 70.
        a11 *= -34.
        a12 *= 6.5
        a13 *= -0.5

        a20 = a21 = a22 = a23 = _Dx5
        a20 *= -84.
        a21 *= 39.
        a22 *= -7.
        a23 *= 0.5

        a30 = a31 = a32 = a33 = _Dx4
        a30 *= 35.
        a31 *= -15.
        a32 *= 2.5
        a33 *= -_1_6

        for i in range(yb.shape[1]):
            ya0 = coeffs[i, 7]
            ya1Dx = coeffs[i, 6] * Dx
            ya2Dx2 = coeffs[i, 5] * Dx2
            ya3Dx3 = coeffs[i, 4] * Dx3

            Dy0 = yb[0, i] - ya0 - ya1Dx - 0.5 * ya2Dx2 - _1_6 * ya3Dx3
            Dy1 = yb[1, i] * Dx - ya1Dx - ya2Dx2 - 0.5 * ya3Dx3
            Dy2 = yb[2, i] * Dx2 - ya2Dx2 - ya3Dx3
            Dy3 = yb[3, i] * Dx3 - ya3Dx3

            coeffs[i, 0] = a00 * Dy0 + a01 * Dy1 + a02 * Dy2 + a03 * Dy3
            coeffs[i, 1] = a10 * Dy0 + a11 * Dy1 + a12 * Dy2 + a13 * Dy3
            coeffs[i, 2] = a20 * Dy0 + a21 * Dy1 + a22 * Dy2 + a23 * Dy3
            coeffs[i, 3] = a30 * Dy0 + a31 * Dy1 + a32 * Dy2 + a33 * Dy3
            coeffs[i, 4] *= _1_6
            coeffs[i, 5] *= 0.5
