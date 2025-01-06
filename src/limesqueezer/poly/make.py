from typing import cast
from typing import TYPE_CHECKING

import numpy as np

from .. import _lnumba as nb
# ======================================================================
# Hinting types
if TYPE_CHECKING:
    from typing import Literal as L
    from typing import Protocol
    from typing import TypeAlias

    from .._typing import Cast
    from .._typing import F64Array
    from .._typing import N_Coeffs
    from .._typing import N_Diffs
    from .._typing import N_Vars

    class Make(Protocol[N_Diffs, N_Coeffs, N_Vars]):
        def __call__(self, Dx: float,
                      ya: F64Array[N_Diffs, N_Vars],
                      yb: F64Array[N_Diffs, N_Vars],
                      coeffs: F64Array[N_Coeffs, N_Vars]) -> None:
            ...
    class Prepare(Protocol[N_Diffs, N_Coeffs, N_Vars]):
        def __call__(self, ya: F64Array[N_Diffs, N_Vars],
                      coeffs: F64Array[N_Coeffs, N_Vars]) -> None:
            ...
    class Finish(Protocol[N_Diffs, N_Coeffs, N_Vars]):
        def __call__(self, Dx: float,
                     yb: F64Array[N_Diffs, N_Vars],
                      coeffs: F64Array[N_Coeffs, N_Vars]) -> None:
            ...
    Make1: TypeAlias = Make[L[1], L[2], int]
    Prepare1: TypeAlias = Prepare[L[1], L[2], int]
    Finish1: TypeAlias = Finish[L[1], L[2], int]
    Split1: TypeAlias = tuple[Prepare1, Finish1]

    Make3: TypeAlias = Make[L[2], L[4], int]
    Prepare3: TypeAlias = Prepare[L[2], L[4], int]
    Finish3: TypeAlias = Finish[L[2], L[4], int]
    Split3: TypeAlias = tuple[Prepare3, Finish3]

    Make5: TypeAlias = Make[L[3], L[6], int]
    Prepare5: TypeAlias = Prepare[L[3], L[6], int]
    Finish5: TypeAlias = Finish[L[3], L[6], int]
    Split5: TypeAlias = tuple[Prepare5, Finish5]

    Make7: TypeAlias = Make[L[4], L[8], int]
    Prepare7: TypeAlias = Prepare[L[4], L[8], int]
    Finish7: TypeAlias = Finish[L[4], L[8], int]
    Split7: TypeAlias = tuple[Prepare7, Finish7]
else:
    L = F64Array = Make = Prepare = Finish = tuple
    N_Coeffs = N_Diffs = N_Vars = int
    Make1 = Prepare1 = Finish1 = Split1 = object
    Make3 = Prepare3 = Finish3 = Split3 = object
    Make5 = Prepare5 = Finish5 = Split5 = object
    Make7 = Prepare7 = Finish7 = Split7 = object
# ======================================================================
nbdec = nb.njit # (nb.void(nb.f64, nb.ARO(2), nb.ARO(2), nb.List[nb.A(2)]))
# ======================================================================
@nbdec
def make1(Dx, ya, yb, coeffs):
    coeffs[0] = (yb[0] - ya[0]) / Dx
    coeffs[1] = ya[0]
make1 = cast(Make1, make1)
# ----------------------------------------------------------------------
@nbdec
def prepare1(ya, coeffs):
    coeffs[1] = ya[0]
prepare1 = cast(Prepare1, prepare1)
# ----------------------------------------------------------------------
@nbdec
def finish1(Dx, yb, coeffs):
    coeffs[0] = yb[0] - coeffs[1]
    coeffs[0] /= Dx
finish1 = cast(Finish1, finish1)
# ----------------------------------------------------------------------
split1 = (prepare1, finish1)
# ======================================================================
# 3
@nbdec
def prepare3(ya, coeffs):
    coeffs[2] = ya[1]
    coeffs[3] = ya[0]
prepare3 = cast(Prepare3, prepare3)
# ----------------------------------------------------------------------
@nbdec
def finish3(Dx, yb, coeffs):

    _Dx = 1./Dx
    _Dx2 = _Dx * _Dx
    _Dx3 = _Dx2 * _Dx

    ya1Dx = coeffs[2] * Dx

    Dy0 = yb[0] - (coeffs[3] + ya1Dx)
    Dy1 = yb[1] * Dx - ya1Dx

    coeffs[0] = (-2. * Dy0 + Dy1) * _Dx3
    coeffs[1] = (3. * Dy0 - Dy1) * _Dx2
finish3 = cast(Finish3, finish3)
# ----------------------------------------------------------------------
split3 = (prepare3, finish3)
# ----------------------------------------------------------------------
@nbdec
def make3(Dx, ya, yb, coeffs):
    prepare3(ya, coeffs)
    finish3(Dx, yb, coeffs)
make3 = cast(Make3, make3)
# ======================================================================
# 5
@nbdec
def prepare5(ya, coeffs):
    coeffs[3] = ya[2]
    coeffs[4] = ya[1]
    coeffs[5] = ya[0]
prepare5 = cast(Prepare5, prepare5)
# ----------------------------------------------------------------------
@nbdec
def finish5(Dx, yb, coeffs):
    Dx2_2 = Dx * Dx * 0.5

    _Dx = 1./Dx
    _Dx2 = _Dx * _Dx
    _Dx3 = _Dx2 * _Dx
    _Dx4 = _Dx2 * _Dx2
    _Dx5 = _Dx3 * _Dx2

    ya1Dx = coeffs[4] * Dx
    ya2Dx2_2 = coeffs[3] * Dx2_2

    Dy0 = yb[0] - coeffs[5] - ya1Dx - ya2Dx2_2
    # Dy0 = yb[0]
    # Dy0 -= coeffs[0]
    # Dy0 -= ya1Dx
    # Dy0 -= ya2Dx2_2

    Dy1 = yb[1] * Dx - ya1Dx - 2 * ya2Dx2_2
    # Dy1 = yb[1]
    # Dy1 *= Dx
    # Dy1 -= ya1Dx
    # Dy1 -= 2. * ya2Dx2_2
    Dy2 = yb[2] * Dx2_2 - ya2Dx2_2

    coeffs[0] = (  6. * Dy0 - 3. * Dy1 +      Dy2) * _Dx5
    coeffs[1] = (-15. * Dy0 + 7. * Dy1 - 2. * Dy2) * _Dx4
    coeffs[2] = ( 10. * Dy0 - 4. * Dy1 +      Dy2) * _Dx3
    coeffs[3] *= 0.5
finish5 = cast(Finish5, finish5)
# ----------------------------------------------------------------------
split5 = (prepare5, finish5)
# ----------------------------------------------------------------------
@nbdec
def make5(Dx, ya, yb, coeffs):
    prepare5(ya, coeffs)
    finish5(Dx, yb, coeffs)
make5 = cast(Make5, make5)
# ======================================================================
# 7
# ======================================================================
# @nbdec
# def make7_incremental(Dx: float, ya: F64Array[L[1], N_Vars], yb: F64Array[L[1], N_Vars], coeffs: F64Array[L[2], N_Vars]) -> None:

#     _1_6 = 0.16666666666666666666666666666666666667

#     Dx2 = Dx * Dx
#     Dx3 = Dx2 * Dx

#     _Dx = 1./Dx
#     _Dx2 = _Dx * _Dx
#     _Dx3 = _Dx2 * _Dx
#     _Dx4 = _Dx2 * _Dx2
#     _Dx5 = _Dx3 * _Dx2
#     _Dx6 = _Dx3 * _Dx3
#     _Dx7 = _Dx4 * _Dx3

#     ya1Dx = ya[1] * Dx
#     ya2Dx2 = ya[2] * Dx2
#     ya3Dx3 = ya[3] * Dx3

#     # Dy0 = yb[0] -      (ya[0] + ya1Dx + 0.5 * ya2Dx2 + _1_6 * ya3Dx3)
#     Dy0 = yb[0]
#     Dy0 -= ya[0]
#     Dy0 -=  ya1Dx
#     Dy0 -= 0.5 * ya2Dx2
#     Dy0 -= _1_6 * ya3Dx3

#     # Dy1 = yb[1] * Dx - (ya1Dx + ya2Dx2 + 0.5 * ya3Dx3)
#     Dy1 = yb[1]
#     Dy1 *= Dx
#     Dy1 -= ya1Dx
#     Dy1 -= ya2Dx2
#     Dy1 -= 0.5 * ya3Dx3

#     # Dy2 = 0.5 * (yb[2] * Dx2 - (ya2Dx2 + ya3Dx3))
#     Dy2 = yb[2]
#     Dy2 *= Dx2
#     Dy2 -= ya2Dx2
#     Dy2 -= ya3Dx3
#     Dy2 *= 0.5

#     # Dy3 = _1_6 * (yb[3] * Dx3 - ya3Dx3)
#     Dy3 = yb[3]
#     Dy3 *= Dx3
#     Dy3 -= ya3Dx3
#     Dy3 *= _1_6

#     coeffs[0] = (-20. * Dy0 + 10. * Dy1 - 2.0 * Dy2 +      Dy3) * _Dx7
#     # coeffs[0] = -20. * Dy0
#     # coeffs[0] += 10. * Dy1
#     # coeffs[0] -= 2.0 * Dy2
#     # coeffs[0] += Dy3
#     # coeffs[0] *= _Dx7

#     coeffs[1] = ( 70. * Dy0 - 34. * Dy1 + 6.5 * Dy2 - 3. * Dy3) * _Dx6
#     # coeffs[1] = 70. * Dy0
#     # coeffs[1] -= 34. * Dy1
#     # coeffs[1] += 6.5 * Dy2
#     # coeffs[1] -= 3. * Dy3
#     # coeffs[1] *= _Dx6

#     coeffs[2] = (-84. * Dy0 + 39. * Dy1 - 7.0 * Dy2 + 3. * Dy3) * _Dx5
#     # coeffs[2] = -84. * Dy0
#     # coeffs[2] += 39. * Dy1
#     # coeffs[2] -= 7.0 * Dy2
#     # coeffs[2] += 3. * Dy3
#     # coeffs[2] *= _Dx5

#     coeffs[3] = ( 35. * Dy0 - 15. * Dy1 + 2.5 * Dy2 -      Dy3) * _Dx4
#     # coeffs[3] = 35. * Dy0
#     # coeffs[3] -= 15. * Dy1
#     # coeffs[3] += 2.5 * Dy2
#     # coeffs[3] -= Dy3
#     # coeffs[3] *= _Dx4

#     coeffs[4] = ya[3] * _1_6
#     coeffs[5] = ya[2] * 0.5
#     coeffs[6] = ya[1]
#     coeffs[7] = ya[0]
# ----------------------------------------------------------------------
@nbdec
def prepare7(ya, coeffs):
    coeffs[4] = ya[3]
    coeffs[5] = ya[2]
    coeffs[6] = ya[1]
    coeffs[7] = ya[0]
prepare7 = cast(Prepare7, prepare7)
# ----------------------------------------------------------------------
@nbdec
def finish7(Dx, yb, coeffs):

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
    # Dy0 = yb[0]
    # Dy0 -= coeffs[0]
    # Dy0 -=  ya1Dx
    # Dy0 -= ya2Dx2_2
    # Dy0 -= ya3Dx3_6
    # print(yb[0,0], coeffs[7,0], ya1Dx[0], ya2Dx2[0], ya3Dx3_6[0])
    # print(yb[0], coeffs[7], ya1Dx, ya2Dx2, ya3Dx3_6)
    Dy0 = yb[0] - coeffs[7] - ya1Dx - 0.5 * ya2Dx2 - ya3Dx3_6
    # print('Dy0', Dy0[0])
    # Dy1 = yb[1]
    # Dy1 *= Dx
    # Dy1 -= ya1Dx
    # Dy1 -= 2. * ya2Dx2_2
    # Dy1 -= 3. * ya3Dx3_6
    Dy1 = yb[1] * Dx - ya1Dx - ya2Dx2 - 3. * ya3Dx3_6
    # print(Dy1)
    # Dy2 = yb[2]
    # Dy2 *= Dx2_2
    # Dy2 -= ya2Dx2_2
    # Dy2 -= 3. * ya3Dx3_6
    Dy2 = yb[2] * Dx2 - ya2Dx2 - 6. * ya3Dx3_6
    # print('Dy2', Dy2)
    # Dy3 = yb[3]
    # Dy3 *= Dx3_6
    # Dy3 -= ya3Dx3_6
    Dy3 = yb[3] * Dx3_6 - ya3Dx3_6
    # print('Dy3', Dy3)
    coeffs[0] = (-20. * Dy0 + 10. * Dy1 - 2.0 * Dy2 +  Dy3) * _Dx7
    coeffs[1] = ( 70. * Dy0 - 34. * Dy1 + 6.5 * Dy2 - 3. * Dy3) * _Dx6
    coeffs[2] = (-84. * Dy0 + 39. * Dy1 - 7.0 * Dy2 + 3. * Dy3) * _Dx5
    coeffs[3] = ( 35. * Dy0 - 15. * Dy1 + 2.5 * Dy2 - Dy3) * _Dx4
    coeffs[4] *= _1_6
    coeffs[5] *= 0.5
finish7 = cast(Finish7, finish7)
# ----------------------------------------------------------------------
split7 = (prepare7, finish7)
# ----------------------------------------------------------------------
@nbdec
def make7(Dx, ya, yb, coeffs):
    prepare7(ya, coeffs)
    finish7(Dx, yb, coeffs)
make7 = cast(Make7, make7)
# ======================================================================
makers: tuple[Make1, Make3, Make5, Make7] = (make1, make3, make5, make7)
# ----------------------------------------------------------------------
@nb.njit
def get_maker(n_y: int) -> Make1 | Make3 | Make5 | Make7:
    if n_y == 1:
        return make1
    if n_y == 2:
        return make3
    if n_y == 3:
        return make5
    return make7 # 4
# ======================================================================

split_makers: tuple[Split1, Split3, Split5, Split7] = (
    split1, split3, split5, split7)
# ----------------------------------------------------------------------
# @nb.njit
def get_split_maker(n_y: int) -> (Split1 | Split3 | Split5 | Split7):
    if n_y == 1:
        return split1
    if n_y == 2:
        return split3
    if n_y == 3:
        return split5
    return split7 # 4
# ----------------------------------------------------------------------
def make(Dx: float,
         ya: F64Array[N_Diffs, N_Vars],
         yb: F64Array[N_Diffs, N_Vars],
         coeffs: F64Array[N_Coeffs, N_Vars]) -> None:
    return get_maker(len(ya))(Dx, yb, coeffs)
# ----------------------------------------------------------------------
