# from typing import Protocol
from typing import overload

from .. import _lnumba as nb
from .._lnumpy import f32
from .._lnumpy import F32Array
from .._lnumpy import f64
from .._lnumpy import F64Array
from .._types import Index
from .._types import N_Coeffs
from .._types import N_Points
# from .._lnumpy import Inty
# from .._lnumpy import UInty
# ======================================================================
_0_Index = Index(0)
_1_Index = Index(1)
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
def single(Dx: f32, coefficients: F32Array[N_Coeffs], n: Index) -> f32:
    ...
@overload
def single[N_PointsTV: N_Points]( # type: ignore[overload-cannot-match]
    Dx: F32Array[N_PointsTV], coefficients: F32Array[N_Coeffs], n: Index
           ) -> F32Array[N_PointsTV]:
    ...
@overload
def single(Dx: f64, coefficients: F64Array[N_Coeffs], n: Index) -> f64: # type: ignore[overload-cannot-match]
    ...
@overload
def single[N_PointsTV: N_Points]( # type: ignore[overload-cannot-match]
    Dx: F64Array[N_PointsTV], coefficients: F64Array[N_Coeffs], n: Index
           ) -> F64Array[N_PointsTV]:
    ...
@nb.njit
def single(Dx, coefficients, n_coeffs):
    """Interpolates degree n polynomial i.e. n + 1 parameters. Computes
    polynomial c[0] * x^{n-1} + c[1] x^{n-2} + ... + c[n]

    Parameters
    ----------
    Dx : x
        _description_
    coefficients : _type_
        _description_
    n_coeffs : _type_
        Number of coefficients to be used

    Returns
    -------
    _type_
        _description_
    """
    # Method is:
    #
    # - P_0(x) = c[0]
    # - P_n(x) = P_{n-1}(x) * x + c[n]

    out = Dx * coefficients[_0_Index]
    i_coeff = _1_Index
    while i_coeff != n_coeffs:
        out += coefficients[i_coeff]
        out *= Dx
        i_coeff += _1_Index
    out += coefficients[i_coeff]
    return out
single
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
