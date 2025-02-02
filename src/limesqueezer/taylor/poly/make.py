from typing import cast
from typing import TYPE_CHECKING

from .. import _lnumba as nb

f32 = nb.f32
f64 = nb.f64
# ======================================================================
# Hinting types
if TYPE_CHECKING:
    from typing import Literal as L
    from typing import Protocol
    from typing import TypeAlias

    # from .._lnumpy import f32
    # from .._lnumpy import f64
    from .._lnumpy import Array
    from .._lnumpy import F32Array
    from .._lnumpy import F64Array

    from .._types import N_Coeffs
    from .._types import N_Diffs
    from .._types import N_Vars

    class Make[N_DiffsTV: N_Diffs,
               N_CoeffsTV: N_Coeffs,
               N_VarsTV: N_Vars,
               DType: (f32, f64)](Protocol):
        def __call__(self, Dx: DType,
                      ya: Array[tuple[N_DiffsTV, N_VarsTV], DType],
                      yb: Array[tuple[N_DiffsTV, N_VarsTV], DType],
                      coeffs: Array[tuple[N_VarsTV, N_CoeffsTV], DType]
                      ) -> None:
            ...
    class Prepare[N_DiffsTV: N_Diffs,
                  N_CoeffsTV: N_Coeffs,
                  N_VarsTV: N_Vars,
                  DType: (f32, f64)](Protocol):
        def __call__(self, ya: Array[tuple[N_DiffsTV, N_VarsTV], DType],
                      coeffs: Array[tuple[N_VarsTV, N_CoeffsTV], DType]) -> None:
            ...
    class Finish[N_DiffsTV: N_Diffs,
                 N_CoeffsTV: N_Coeffs,
                 N_VarsTV: N_Vars,
                 DType: (f32, f64)](Protocol):
        def __call__(self, Dx: DType,
                     yb: Array[tuple[N_DiffsTV, N_VarsTV], DType],
                     coeffs: Array[tuple[N_VarsTV, N_CoeffsTV], DType]) -> None:
            ...

    Make1_64: TypeAlias = Make[L[1], L[2], N_Vars, f64]
    Prepare1_64: TypeAlias = Prepare[L[1], L[2], N_Vars, f64]
    Finish1_64: TypeAlias = Finish[L[1], L[2], N_Vars, f64]
    Split1_64: TypeAlias = tuple[Prepare1_64, Finish1_64]

    Make3_64: TypeAlias = Make[L[2], L[4], N_Vars, f64]
    Prepare3_64: TypeAlias = Prepare[L[2], L[4], N_Vars, f64]
    Finish3_64: TypeAlias = Finish[L[2], L[4], N_Vars, f64]
    Split3_64: TypeAlias = tuple[Prepare3_64, Finish3_64]

    Make5_64: TypeAlias = Make[L[3], L[6], N_Vars, f64]
    Prepare5_64: TypeAlias = Prepare[L[3], L[6], N_Vars, f64]
    Finish5_64: TypeAlias = Finish[L[3], L[6], N_Vars, f64]
    Split5_64: TypeAlias = tuple[Prepare5_64, Finish5_64]

    Make7_64: TypeAlias = Make[L[4], L[8], N_Vars, f64]
    Prepare7_64: TypeAlias = Prepare[L[4], L[8], N_Vars, f64]
    Finish7_64: TypeAlias = Finish[L[4], L[8], N_Vars, f64]
    Split7_64: TypeAlias = tuple[Prepare7_64, Finish7_64]

    Split_Maker_64: TypeAlias = Split1_64 | Split3_64 | Split5_64 | Split7_64
    Split_Makers_64: TypeAlias = tuple[Split1_64, Split3_64, Split5_64, Split7_64]
    Makers_64: TypeAlias = tuple[Make1_64, Make3_64, Make5_64, Make7_64]
else:

    nbPrepareSignature_64 = nb.void(nb.ARO(2, f64), f64[:, ::1])

    nbFinishSinature_64 = nb.void(f64, nb.ARO(2, f64), f64[:, ::1])

    nbMakeSignature_64 = nb.void(f64, nb.ARO(2, f64), nb.ARO(2, f64), f64[:, ::1])

    F64Array = tuple

    Make_64: nb.Type = nbMakeSignature_64.as_type()
    Prepare_64: nb.Type = nbPrepareSignature_64.as_type()
    Finish_64: nb.Type = nbFinishSinature_64.as_type()

    N_CoeffsTV = N_DiffsTV = N_VarsTV = int
    Make1_64 = Prepare1_64 = Finish1_64 = Split1_64 = object
    Make3_64 = Prepare3_64 = Finish3_64 = Split3_64 = object
    Make5_64 = Prepare5_64 = Finish5_64 = Split5_64 = object
    Make7_64 = Prepare7_64 = Finish7_64 = Split7_64 = object
    Split_Maker_64 = nb.types.Tuple((Prepare_64, Finish_64))
    Split_Makers_64 = nb.types.UniTuple(Split_Maker_64, 4)
    Makers_64 = object
# ======================================================================
nbdec = nb.njitC
# ======================================================================
@nbdec
def make1_64(Dx, ya, yb, coeffs):
    _Dx = 1. / Dx
    for i in range(ya.shape[1]):
        coeffs[i, 0] = (yb[0, i] - ya[0, i]) * _Dx
        coeffs[i, 1] = ya[0, i]
make1_64 = cast(Make1_64, make1_64)
# ----------------------------------------------------------------------
@nbdec
def prepare1_64(ya, coeffs):
    for i in range(ya.shape[1]):
        coeffs[i, 1] = ya[0, i]
# ----------------------------------------------------------------------
prepare1_64 = cast(Prepare1_64, prepare1_64)
# ----------------------------------------------------------------------
@nbdec
def finish1_64(Dx, yb, coeffs):
    _Dx = 1. / Dx
    for i in range(yb.shape[1]):
        coeffs[i, 0] = (yb[0, i] - coeffs[i, 1]) * _Dx
# ----------------------------------------------------------------------
finish1_64 = cast(Finish1_64, finish1_64)
# ----------------------------------------------------------------------
split1_64 = (prepare1_64, finish1_64)
# ======================================================================
# 3
@nbdec
def prepare3_64(ya, coeffs):
    for i in range(ya.shape[1]):
        coeffs[i, 2] = ya[1, i]
        coeffs[i, 3] = ya[0, i]
# ----------------------------------------------------------------------
prepare3_64 = cast(Prepare3_64, prepare3_64)
# ----------------------------------------------------------------------
@nbdec
def finish3_64(Dx, yb, coeffs):

    _Dx = f64(1.)/Dx
    _Dx2 = _Dx * _Dx

    a00 = a01 = _Dx2 * _Dx
    a00 *= f64(-2.)

    a10 = a11 = _Dx2
    a10 *= f64(3.)
    a11 *= f64(-1.)

    for i in range(yb.shape[1]):
        ya1Dx = coeffs[i, 2] * Dx

        Dy0 = yb[0, i] - (coeffs[i, 3] + ya1Dx)
        Dy1 = yb[1, i] * Dx - ya1Dx

        coeffs[i, 0] = a00 * Dy0 + a01 * Dy1
        coeffs[i, 1] = a10 * Dy0 + a11 * Dy1
# ----------------------------------------------------------------------
finish3_64 = cast(Finish3_64, finish3_64)
# ----------------------------------------------------------------------
split3_64 = (prepare3_64, finish3_64)
# ----------------------------------------------------------------------
@nbdec
def make3_64(Dx, ya, yb, coeffs):
    prepare3_64(ya, coeffs)
    finish3_64(Dx, yb, coeffs)
make3_64 = cast(Make3_64, make3_64)
# ======================================================================
# 5
@nbdec
def prepare5_64(ya, coeffs):
    for i in range(ya.shape[1]):
        coeffs[i, 3] = ya[2, i]
        coeffs[i, 4] = ya[1, i]
        coeffs[i, 5] = ya[0, i]
prepare5_64 = cast(Prepare5_64, prepare5_64)
# ----------------------------------------------------------------------
@nbdec
def finish5_64(Dx, yb, coeffs):

    _1_2 = f64(0.5)

    Dx2 = Dx * Dx

    _Dx = f64(1.)/Dx
    _Dx2 = _Dx * _Dx
    _Dx3 = _Dx2 * _Dx

    a00 = a01 = a02 = _Dx3 * _Dx2
    a00 *= f64(6.)
    a01 *= f64(-3.)
    a02 *= _1_2

    a10 = a11 = a12 = _Dx3 * _Dx
    a10 *= f64(-15.)
    a11 *= f64(7.)
    a12 *= f64(-1.)

    a20 = a21 = a22 = _Dx3
    a20 *= f64(10.)
    a21 *= f64(-4.)
    a22 *= _1_2

    for i in range(yb.shape[1]):
        ya1Dx = coeffs[i, 4] * Dx
        ya2Dx2 = coeffs[i, 3] * Dx2

        Dy0 = yb[0, i] - coeffs[i, 5] - ya1Dx - _1_2 * ya2Dx2
        Dy1 = yb[1, i] * Dx - ya1Dx - ya2Dx2
        Dy2 = yb[2, i] * Dx2 - ya2Dx2

        coeffs[i, 0] = a00 * Dy0 + a01 * Dy1 + a02 * Dy2
        coeffs[i, 1] = a10 * Dy0 + a11 * Dy1 + a12 * Dy2
        coeffs[i, 2] = a20 * Dy0 + a21 * Dy1 + a22 * Dy2
        coeffs[i, 3] *= _1_2
finish5_64 = cast(Finish5_64, finish5_64)
# ----------------------------------------------------------------------
split5_64 = (prepare5_64, finish5_64)
# ----------------------------------------------------------------------
@nbdec
def make5_64(Dx, ya, yb, coeffs):
    prepare5_64(ya, coeffs)
    finish5_64(Dx, yb, coeffs)
make5_64 = cast(Make5_64, make5_64)
# ======================================================================
# 7
# ----------------------------------------------------------------------
@nbdec
def prepare7_64(ya, coeffs):
    for i in range(ya.shape[1]):
        coeffs[i, 4] = ya[3, i]
        coeffs[i, 5] = ya[2, i]
        coeffs[i, 6] = ya[1, i]
        coeffs[i, 7] = ya[0, i]
prepare7_64 = cast(Prepare7_64, prepare7_64)
# ----------------------------------------------------------------------
@nbdec
def finish7_64(Dx, yb, coeffs):

    _1_2 = f64(0.5)
    _1_6 = f64(0.16666666666666666666666666666666666666666666667)

    Dx2 = Dx * Dx
    Dx3 = Dx2 * Dx

    _Dx1 = 1./Dx
    _Dx2 = _Dx1 * _Dx1

    _Dx4 = _Dx2 * _Dx2
    _Dx5 = _Dx4 * _Dx1
    _Dx6 = _Dx5 * _Dx1
    _Dx7 = _Dx6 * _Dx1

    a00 = a01 = a02 = a03 = _Dx7
    a00 *= f64(-20.)
    a01 *= f64(10.)
    a02 *= f64(-2.)
    a03 *= _1_6

    a10 = a11 = a12 = a13 = _Dx6
    a10 *= f64(70.)
    a11 *= f64(-34.)
    a12 *= f64(6.5)
    a13 *= - _1_2

    a20 = a21 = a22 = a23 = _Dx5
    a20 *= f64(-84.)
    a21 *= f64(39.)
    a22 *= f64(-7.)
    a23 *= _1_2

    a30 = a31 = a32 = a33 = _Dx4
    a30 *= f64(35.)
    a31 *= f64(-15.)
    a32 *= f64(2.5)
    a33 *= -_1_6

    for i in range(yb.shape[1]):
        ya0 = coeffs[i, 7]
        ya1Dx = coeffs[i, 6] * Dx
        ya2Dx2 = coeffs[i, 5] * Dx2
        ya3Dx3 = coeffs[i, 4] * Dx3

        Dy0 = yb[0, i] - ya0 - ya1Dx - _1_2 * ya2Dx2 - _1_6 * ya3Dx3
        Dy1 = yb[1, i] * Dx - ya1Dx - ya2Dx2 - _1_2 * ya3Dx3
        Dy2 = yb[2, i] * Dx2 - ya2Dx2 - ya3Dx3
        Dy3 = yb[3, i] * Dx3 - ya3Dx3

        coeffs[i, 0] = a00 * Dy0 + a01 * Dy1 + a02 * Dy2 + a03 * Dy3
        coeffs[i, 1] = a10 * Dy0 + a11 * Dy1 + a12 * Dy2 + a13 * Dy3
        coeffs[i, 2] = a20 * Dy0 + a21 * Dy1 + a22 * Dy2 + a23 * Dy3
        coeffs[i, 3] = a30 * Dy0 + a31 * Dy1 + a32 * Dy2 + a33 * Dy3
        coeffs[i, 4] *= _1_6
        coeffs[i, 5] *= _1_2
finish7_64 = cast(Finish7_64, finish7_64)
# ----------------------------------------------------------------------
split7_64 = (prepare7_64, finish7_64)
# ----------------------------------------------------------------------
@nbdec
def make7_64(Dx, ya, yb, coeffs):
    prepare7_64(ya, coeffs)
    finish7_64(Dx, yb, coeffs)
make7_64 = cast(Make7_64, make7_64)
# ======================================================================
makers_64: Makers_64 = (make1_64, make3_64, make5_64, make7_64)
# ----------------------------------------------------------------------
split_makers: Split_Makers_64 = (split1_64, split3_64, split5_64, split7_64)
