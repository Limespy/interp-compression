from .. import _lnumba as nb
from .._lnumpy import F64Array
from .._types import N_Coeffs
from .._types import N_Vars
# ----------------------------------------------------------------------
@nb.njit
def oprepare7(ya, coeffs):
    coeffs[4] = ya[3]
    coeffs[5] = ya[2]
    coeffs[6] = ya[1]
    coeffs[7] = ya[0]
# ----------------------------------------------------------------------
@nb.njit
def ofinish7(Dx, yb, coeffs: F64Array[N_Coeffs, N_Vars]):

    _1_6 = 0.16666666666666666666666666666666666666666666667
    Dx2 = Dx * Dx
    Dx3_6 = Dx2 * Dx * _1_6

    _Dx = 1./Dx
    _Dx2 = _Dx * _Dx
    _Dx3 = _Dx2 * _Dx
    _Dx4 = _Dx2 * _Dx2
    _Dx5 = _Dx3 * _Dx2
    _Dx6 = _Dx3 * _Dx3
    _Dx7 = _Dx4 * _Dx3

    ya1Dx = coeffs[6] * Dx
    ya2Dx2 = coeffs[5] * Dx2
    ya3Dx3_6 = coeffs[4] * Dx3_6

    Dy0 = yb[0] - coeffs[7] - ya1Dx - 0.5 * ya2Dx2 - ya3Dx3_6
    Dy1 = yb[1] * Dx - ya1Dx - ya2Dx2 - 3. * ya3Dx3_6
    Dy2 = yb[2] * Dx2 - ya2Dx2 - 6. * ya3Dx3_6
    Dy3 = yb[3] * Dx3_6 - ya3Dx3_6

    coeffs[0] = (-20. * Dy0 + 10. * Dy1 - 2.0 * Dy2 +  Dy3) * _Dx7
    coeffs[1] = ( 70. * Dy0 - 34. * Dy1 + 6.5 * Dy2 - 3. * Dy3) * _Dx6
    coeffs[2] = (-84. * Dy0 + 39. * Dy1 - 7.0 * Dy2 + 3. * Dy3) * _Dx5
    coeffs[3] = ( 35. * Dy0 - 15. * Dy1 + 2.5 * Dy2 - Dy3) * _Dx4
    coeffs[4] *= _1_6
    coeffs[5] *= 0.5
# ----------------------------------------------------------------------
@nb.njit
def omake7(Dx, ya, yb, coeffs):
    oprepare7(ya, coeffs)
    ofinish7(Dx, yb, coeffs)
