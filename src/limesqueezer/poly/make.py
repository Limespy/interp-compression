from typing import cast
from typing import TYPE_CHECKING

from .. import _lnumba as nb
# ======================================================================
# Hinting types
if TYPE_CHECKING:
    from typing import Literal as L
    from typing import Protocol
    from typing import TypeAlias
    from typing import TypeVar
    from .._lnumpy import up

    from .._typing import F64Array

    N_CoeffsTV = TypeVar('N_CoeffsTV', bound = int, contravariant = True)
    N_DiffsTV = TypeVar('N_DiffsTV', bound = int, contravariant = True)
    N_VarsTV = TypeVar('N_VarsTV', bound = int, contravariant = True)


    class Make(Protocol[N_DiffsTV, N_CoeffsTV, N_VarsTV]):
        def __call__(self, Dx: float,
                      ya: F64Array[N_DiffsTV, N_VarsTV],
                      yb: F64Array[N_DiffsTV, N_VarsTV],
                      coeffs: F64Array[N_CoeffsTV, N_VarsTV]) -> None:
            ...
    class Prepare(Protocol[N_DiffsTV, N_CoeffsTV, N_VarsTV]):
        def __call__(self, ya: F64Array[N_DiffsTV, N_VarsTV],
                      coeffs: F64Array[N_CoeffsTV, N_VarsTV]) -> None:
            ...
    class Finish(Protocol[N_DiffsTV, N_CoeffsTV, N_VarsTV]):
        def __call__(self, Dx: float,
                     yb: F64Array[N_DiffsTV, N_VarsTV],
                      coeffs: F64Array[N_CoeffsTV, N_VarsTV]) -> None:
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

    Split_Maker: TypeAlias = Split1 | Split3 | Split5 | Split7
    Split_Makers: TypeAlias = tuple[Split1, Split3, Split5, Split7]
else:

    nbPrepareSignature: nb.Signature = nb.void(nb.ARO(2), nb.A(2))

    nbFinishSinature: nb.Signature = nb.void(nb.f64, nb.ARO(2), nb.A(2))

    nbMakeSignature: nb.Signature = nb.void(nb.f64, nb.ARO(2), nb.ARO(2), nb.A(2))

    L = F64Array = tuple
    Make: nb.Type = nbMakeSignature.as_type()
    Prepare: nb.Type = nbPrepareSignature.as_type()
    Finish: nb.Type = nbFinishSinature.as_type()
    N_CoeffsTV = N_DiffsTV = N_VarsTV = int
    Make1 = Prepare1 = Finish1 = Split1 = object
    Make3 = Prepare3 = Finish3 = Split3 = object
    Make5 = Prepare5 = Finish5 = Split5 = object
    Make7 = Prepare7 = Finish7 = Split7 = object
    Split_Maker = nb.types.Tuple((Prepare, Finish))
    Split_Makers = nb.types.UniTuple(Split_Maker, 4)
# ======================================================================
nbdec = nb.njitC
# ======================================================================
@nbdec
def make1(Dx, ya, yb, coeffs):
    coeffs[0] = (yb[0] - ya[0]) / Dx
    coeffs[1] = ya[0]
make1 = cast(Make1, make1)
# ----------------------------------------------------------------------
@nbdec
def prepare1(ya, coeffs):
    for i in range(ya.shape[1]):
        coeffs[i, 1] = ya[0, i]
# ----------------------------------------------------------------------
prepare1 = cast(Prepare1, prepare1)
# ----------------------------------------------------------------------
@nbdec
def finish1(Dx, yb, coeffs):
    _Dx = 1. / Dx
    for i in range(yb.shape[1]):
        coeffs[i, 0] = (yb[0, i] - coeffs[i, 1]) * _Dx
# ----------------------------------------------------------------------
finish1 = cast(Finish1, finish1)
# ----------------------------------------------------------------------
split1 = (prepare1, finish1)
# ======================================================================
# 3
@nbdec
def prepare3(ya, coeffs):
    for i in range(ya.shape[1]):
        coeffs[i, 2] = ya[1, i]
        coeffs[i, 3] = ya[0, i]
# ----------------------------------------------------------------------
prepare3 = cast(Prepare3, prepare3)
# ----------------------------------------------------------------------
@nbdec
def finish3(Dx, yb, coeffs):

    _Dx = 1./Dx
    _Dx2 = _Dx * _Dx

    a00 = a01 = _Dx2 * _Dx
    a00 *= -2.

    a10 = a11 = _Dx2
    a10 *= 3.
    a11 *= -1.

    for i in range(yb.shape[1]):
        ya1Dx = coeffs[i, 2] * Dx

        Dy0 = yb[0, i] - (coeffs[i, 3] + ya1Dx)
        Dy1 = yb[1, i] * Dx - ya1Dx

        coeffs[i, 0] = a00 * Dy0 + a01 * Dy1
        coeffs[i, 1] = a10 * Dy0 + a11 * Dy1
# ----------------------------------------------------------------------
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
    for i in range(ya.shape[1]):
        coeffs[i, 3] = ya[2, i]
        coeffs[i, 4] = ya[1, i]
        coeffs[i, 5] = ya[0, i]
prepare5 = cast(Prepare5, prepare5)
# ----------------------------------------------------------------------
@nbdec
def finish5(Dx, yb, coeffs):
    Dx2 = Dx * Dx

    _Dx = 1./Dx
    _Dx2 = _Dx * _Dx
    _Dx3 = _Dx2 * _Dx

    a00 = a01 = a02 = _Dx3 * _Dx2
    a00 *= 6.
    a01 *= -3.
    a02 *= 0.5

    a10 = a11 = a12 = _Dx3 * _Dx
    a10 *= -15.
    a11 *= 7.
    a12 *= -1.

    a20 = a21 = a22 = _Dx3
    a20 *= 10.
    a21 *= -4.
    a22 *= 0.5

    for i in range(yb.shape[1]):
        ya1Dx = coeffs[i, 4] * Dx
        ya2Dx2 = coeffs[i, 3] * Dx2

        Dy0 = yb[0, i] - coeffs[i, 5] - ya1Dx - 0.5 * ya2Dx2
        Dy1 = yb[1, i] * Dx - ya1Dx - ya2Dx2
        Dy2 = yb[2, i] * Dx2 - ya2Dx2

        coeffs[i, 0] = a00 * Dy0 + a01 * Dy1 + a02 * Dy2
        coeffs[i, 1] = a10 * Dy0 + a11 * Dy1 + a12 * Dy2
        coeffs[i, 2] = a20 * Dy0 + a21 * Dy1 + a22 * Dy2
        coeffs[i, 3] *= 0.5
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
# ----------------------------------------------------------------------
@nbdec
def oprepare7(ya, coeffs):
    coeffs[4] = ya[3]
    coeffs[5] = ya[2]
    coeffs[6] = ya[1]
    coeffs[7] = ya[0]
# ----------------------------------------------------------------------
@nbdec
def prepare7(ya, coeffs):
    for i in range(ya.shape[1]):
        coeffs[i, 4] = ya[3, i]
        coeffs[i, 5] = ya[2, i]
        coeffs[i, 6] = ya[1, i]
        coeffs[i, 7] = ya[0, i]
prepare7 = cast(Prepare7, prepare7)
# ----------------------------------------------------------------------
@nbdec
def ofinish7(Dx, yb, coeffs):

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
@nbdec
def finish7(Dx, yb, coeffs: F64Array[N_VarsTV, N_CoeffsTV]):

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
finish7 = cast(Finish7, finish7)
# ----------------------------------------------------------------------
split7 = (prepare7, finish7)
# ----------------------------------------------------------------------
@nbdec
def omake7(Dx, ya, yb, coeffs):
    oprepare7(ya, coeffs)
    ofinish7(Dx, yb, coeffs)
# ----------------------------------------------------------------------
@nbdec
def make7(Dx, ya, yb, coeffs):
    prepare7(ya, coeffs)
    finish7(Dx, yb, coeffs)
make7 = cast(Make7, make7)
# ======================================================================
makers: tuple[Make1, Make3, Make5, Make7] = (make1, make3, make5, make7)
# ----------------------------------------------------------------------
split_makers: Split_Makers = (split1, split3, split5, split7)
