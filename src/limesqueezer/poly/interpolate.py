# from typing import Protocol
from typing import TYPE_CHECKING
from typing import TypeVar

from .. import _lnumba as nb
# import numpy as np
# from .._typing import XSingle
# from .._typing import N_Coeffs
# ======================================================================
# Hinting types
if TYPE_CHECKING:
    from typing import Literal as L
    from typing import overload

    from .._lnumpy import f64
    from .._lnumpy import F64Array
    from .._lnumpy import Floaty
    from .._lnumpy import Inty
    from .._lnumpy import UInty

else:
    f64 = Floaty = Inty = UInty = object
    F64Array = L = tuple
    overload = lambda x: x
# ----------------------------------------------------------------------
N_CoeffsTV = TypeVar('N_CoeffsTV', bound = int)
N_PointsTV = TypeVar('N_PointsTV', bound = int)
N_VarsTV = TypeVar('N_VarsTV', bound = int)
# ----------------------------------------------------------------------
# class Interpolator(Protocol[N_CoeffsTV, N_VarsTV]):
#     @overload
#     def __call__(self, Dx: float, coefficients: F64Array[N_CoeffsTV, N_VarsTV]
#                  ) -> F64Array[N_VarsTV]:
#         ...
#     @overload
#     def __call__(self, Dx: F64Array[N_PointsTV, L[1]],
#                  coefficients: F64Array[N_CoeffsTV, N_VarsTV]
#                  ) -> F64Array[N_PointsTV, N_VarsTV]:
#         ...

# ======================================================================
# nbInterpSingleType = nb.f64[:](XSingle, N_Coeffs).as_type()
nbdec = nb.njit
# ======================================================================
# @overload
# def interpolate(Dx: float, coefficients: F64Array[N_CoeffsTV], n: int
#                 ) -> float:
#     ...
# @overload
# def group(Dx: float,
#                 coefficients: F64Array[N_CoeffsTV, N_VarsTV],
#                 n: int,
#                 out: F64Array[N_VarsTV]) -> None:
#     ...
# @overload
# def group(Dx: F64Array[N_PointsTV],
#                 coefficients: F64Array[N_CoeffsTV],
#                 n: int,
#                 out: F64Array[N_VarsTV]) -> None:
#     ...
# @overload
# def group(Dx: F64Array[N_PointsTV, L[1]],
#                 coefficients: F64Array[N_CoeffsTV],
#                 n: int,
#                 out: F64Array[N_PointsTV, N_VarsTV]) -> None:
#     ...
# @nb.njit
# def group(Dx, coefficients, n, out):
#     np.multiply(Dx, coefficients[0], out)
#     for i in range(1, n):
#         out += coefficients[i]
#         out *= Dx
#     out += coefficients[n]

@overload
def single(Dx: Floaty, coefficients: F64Array[N_CoeffsTV], n: Inty | UInty
           ) -> f64:
    ...
@overload
def single(Dx: F64Array[N_PointsTV], coefficients: F64Array[N_CoeffsTV], n: int
           ) -> F64Array[N_PointsTV]:
    ...
@nb.njit
def single(Dx, coefficients, n):
    """Interpolates degree n polynomial i.e. n+ 1 parameters."""
    out = Dx * coefficients[0]
    for i in range(1, n):
        out += coefficients[i]
        out *= Dx
    out += coefficients[n]
    return out
# ======================================================================
# @nbdec
# def poly1(Dx: F64Array | float, coefficients: F64Array) -> F64Array:
#     out = Dx * coefficients[0]
#     out += coefficients[1]
#     return out
# # ======================================================================
# @nbdec
# def poly2(Dx: F64Array | float, coefficients: F64Array) -> F64Array:
#     out = Dx * coefficients[0]
#     out += coefficients[1]
#     out *= Dx
#     out += coefficients[2]
#     return out
# # ======================================================================
# @nbdec
# def poly3(Dx: F64Array | float, coefficients: F64Array) -> F64Array:
#     out = Dx * coefficients[0]
#     out += coefficients[1]
#     out *= Dx
#     out += coefficients[2]
#     out *= Dx
#     out += coefficients[3]
#     return out
# # ======================================================================
# @nbdec
# def poly4(Dx: F64Array | float, coefficients: F64Array) -> F64Array:
#     out = Dx * coefficients[0]
#     out += coefficients[1]
#     out *= Dx
#     out += coefficients[2]
#     out *= Dx
#     out += coefficients[3]
#     out *= Dx
#     out += coefficients[4]
#     return out
# # ======================================================================
# @nbdec
# def poly5(Dx: F64Array | float, coefficients: F64Array) -> F64Array:
#     out = Dx * coefficients[0]
#     out += coefficients[1]
#     out *= Dx
#     out += coefficients[2]
#     out *= Dx
#     out += coefficients[3]
#     out *= Dx
#     out += coefficients[4]
#     out *= Dx
#     out += coefficients[5]
#     return out
# # ======================================================================
# @nbdec
# def poly6(Dx: F64Array | float, coefficients: F64Array) -> F64Array:
#     out = Dx * coefficients[0]
#     out += coefficients[1]
#     out *= Dx
#     out += coefficients[2]
#     out *= Dx
#     out += coefficients[3]
#     out *= Dx
#     out += coefficients[4]
#     out *= Dx
#     out += coefficients[5]
#     out *= Dx
#     out += coefficients[6]
#     return out
# # ======================================================================
# @nbdec
# def poly7(Dx: F64Array | float, coefficients: F64Array) -> F64Array:
#     out = Dx * coefficients[0]
#     out += coefficients[1]
#     out *= Dx
#     out += coefficients[2]
#     out *= Dx
#     out += coefficients[3]
#     out *= Dx
#     out += coefficients[4]
#     out *= Dx
#     out += coefficients[5]
#     out *= Dx
#     out += coefficients[6]
#     out *= Dx
#     out += coefficients[7]
#     return out
# # ======================================================================
# interpolators = (poly1, poly2, poly3, poly4, poly5, poly6, poly7)
# nbInterpolatorsList = nb.ListType(nbInterpSingleType)

# @nb.njit
# def get_interpolators(n_diffs: int) -> list[Interpolator]:
#     out = List.empty_list(nbInterpSingleType)
#     if n_diffs == uintp(1):
#         out.append(poly1)
#     elif n_diffs == uintp(2):
#         out.append(poly3)
#         out.append(poly2)
#     elif n_diffs == uintp(3):
#         out.append(poly5)
#         out.append(poly4)
#         out.append(poly3)
#     elif n_diffs == uintp(4):
#         out.append(poly7)
#         out.append(poly6)
#         out.append(poly5)
#         out.append(poly4)
#     return out
