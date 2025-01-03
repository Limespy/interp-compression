from typing import overload
from typing import Protocol
from typing import TYPE_CHECKING
from typing import TypeVar

from numba.typed import List
from numpy import uintp

from .. import _lnumba as nb
# ======================================================================
# Hinting types
if TYPE_CHECKING:
    from collections.abc import Callable

    from typing import TypeAlias
    from typing import Literal as L


    from .._lnumpy import F64Array

    InterpSingle: TypeAlias = Callable[[F64Array | float, F64Array], F64Array]

else:
    F64Array = L = tuple
# ----------------------------------------------------------------------
N_Coeffs = TypeVar('N_Coeffs', bound = int)
N_Diffs = TypeVar('N_Diffs', bound = int)
N_Points = TypeVar('N_Points', bound = int)
N_Samples = TypeVar('N_Samples', bound = int)
N_Vars = TypeVar('N_Vars', bound = int)
# ----------------------------------------------------------------------
class Interpolator(Protocol[N_Coeffs, N_Vars]):
    @overload
    def __call__(self, Dx: float, coefficients: F64Array[N_Coeffs, N_Vars]
                 ) -> F64Array[N_Vars]:
        ...
    @overload
    def __call__(self, Dx: F64Array[N_Points, L[1]],
                 coefficients: F64Array[N_Coeffs, N_Vars]
                 ) -> F64Array[N_Points, N_Vars]:
        ...

# ======================================================================
nbInterpSingleType = nb.f64(nb.float64, nb.float64[:]).as_type()
nbdec = nb.njit
# ======================================================================
@nbdec
def poly1(Dx: F64Array | float, coefficients: F64Array) -> F64Array:
    out = Dx * coefficients[0]
    out += coefficients[1]
    return out
# ======================================================================
@nbdec
def poly2(Dx: F64Array | float, coefficients: F64Array) -> F64Array:
    out = Dx * coefficients[0]
    out += coefficients[1]
    out *= Dx
    out += coefficients[2]
    return out
# ======================================================================
@nbdec
def poly3(Dx: F64Array | float, coefficients: F64Array) -> F64Array:
    out = Dx * coefficients[0]
    out += coefficients[1]
    out *= Dx
    out += coefficients[2]
    out *= Dx
    out += coefficients[3]
    return out
# ======================================================================
@nbdec
def poly4(Dx: F64Array | float, coefficients: F64Array) -> F64Array:
    out = Dx * coefficients[0]
    out += coefficients[1]
    out *= Dx
    out += coefficients[2]
    out *= Dx
    out += coefficients[3]
    out *= Dx
    out += coefficients[4]
    return out
# ======================================================================
@nbdec
def poly5(Dx: F64Array | float, coefficients: F64Array) -> F64Array:
    out = Dx * coefficients[0]
    out += coefficients[1]
    out *= Dx
    out += coefficients[2]
    out *= Dx
    out += coefficients[3]
    out *= Dx
    out += coefficients[4]
    out *= Dx
    out += coefficients[5]
    return out
# ======================================================================
@nbdec
def poly6(Dx: F64Array | float, coefficients: F64Array) -> F64Array:
    out = Dx * coefficients[0]
    out += coefficients[1]
    out *= Dx
    out += coefficients[2]
    out *= Dx
    out += coefficients[3]
    out *= Dx
    out += coefficients[4]
    out *= Dx
    out += coefficients[5]
    out *= Dx
    out += coefficients[6]
    return out
# ======================================================================
@nbdec
def poly7(Dx: F64Array | float, coefficients: F64Array) -> F64Array:
    out = Dx * coefficients[0]
    out += coefficients[1]
    out *= Dx
    out += coefficients[2]
    out *= Dx
    out += coefficients[3]
    out *= Dx
    out += coefficients[4]
    out *= Dx
    out += coefficients[5]
    out *= Dx
    out += coefficients[6]
    out *= Dx
    out += coefficients[7]
    return out
# ======================================================================
interpolators = (None, poly1, poly2, poly3, poly4, poly5, poly6, poly7)
nbInterpolatorsList = nb.List[nbInterpSingleType]

@nb.njit
def get_interpolators(n_diffs: int) -> list[Interpolator]:
    out = List.empty_list(nbInterpSingleType)
    if n_diffs == uintp(1):
        out.append(poly1)
    elif n_diffs == uintp(2):
        out.append(poly3)
        out.append(poly2)
    elif n_diffs == uintp(3):
        out.append(poly5)
        out.append(poly4)
        out.append(poly3)
    elif n_diffs == uintp(4):
        out.append(poly7)
        out.append(poly6)
        out.append(poly5)
        out.append(poly4)
    return out
# ======================================================================
def interpolate(Dx: F64Array | float, coefficients: F64Array) -> F64Array:
    return interpolators[len(coefficients) - 2](Dx, coefficients)
