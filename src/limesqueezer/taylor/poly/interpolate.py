# from typing import Protocol
from math import log2
from typing import overload

import numpy as np

from .. import _lnumba as nb
from ..._lnumpy import Array
from ..._lnumpy import f32
from ..._lnumpy import f64
from ..._lnumpy import u64
from ..._root import linear
from ..._types import Index
from ..._types import N_Coeffs
from ..._types import N_Points
from ..._types import N_Vars
# from .._lnumpy import Inty
# from .._lnumpy import UInty
# ======================================================================
_0_Index = Index(0)
_1_Index = Index(1)
_2_Index = Index(2)
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
def single[DType: (f32, f64)
           ](Dx: DType, coefficients: Array[tuple[N_Coeffs], DType], n: Index
             ) -> DType:
    ...
@overload
def single[N_PointsTV: N_Points, DType: (f32, f64)
           ]( # type: ignore[overload-cannot-match]
             Dx: Array[tuple[N_PointsTV], DType],
             coefficients: Array[tuple[N_Coeffs], DType],
             n: Index
             ) -> Array[tuple[N_PointsTV], DType]:
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
# ======================================================================
@nb.njit
def _find_i_poly[N_PointsTV: N_Points,
                 DType: (f32, f64)
                 ](x: DType,
                   x_all: Array[tuple[N_PointsTV], DType],
                   index_low: Index,
                   index_high: Index):
    """Searches polynomial index."""

    # Simple binomial search for now

    while index_high - index_low > _1_Index:
        index_try = index_low + (index_high - index_low) >> _1_Index
        if x_all[index_try] < x:
            index_low = index_try
        else:
            index_high = index_try

    return index_low
# ======================================================================
@nb.njit
def _update_vars[N_X: int,
                 N_Polys: int,
                 N_VarsTV: N_Vars,
                 N_CoeffsTV: N_Coeffs,
                 DType: (f32, f64)
                 ](i_point: Index,
                   x: Array[tuple[N_X], DType],
                   x_coeffs: Array[tuple[int], DType],
                   coefficients: Array[tuple[N_Polys, N_VarsTV, N_CoeffsTV], DType],
                   i_poly_low: Index,
                   i_poly_high: Index,
                   out: Array[tuple[N_X, N_VarsTV], DType]) -> Index:

    x_point = x[i_point]
    i_poly = _find_i_poly(x_point, x_coeffs, i_poly_low, i_poly_high)
    Dx = x_point - x_coeffs[i_poly]
    for i_var in range(coefficients.shape[1]):
        out[i_point, i_var] = single(Dx, coefficients[i_poly, i_var])
    return i_poly
# ======================================================================
# _log2_tab64 = np.array((
#     63,  0, 58,  1, 59, 47, 53,  2,
#     60, 39, 48, 27, 54, 33, 42,  3,
#     61, 51, 37, 40, 49, 18, 28, 20,
#     55, 30, 34, 11, 43, 14, 22,  4,
#     62, 57, 46, 52, 38, 26, 32, 41,
#     50, 36, 17, 19, 29, 10, 13, 21,
#     56, 45, 25, 31, 35, 16,  9, 12,
#     44, 24, 15,  8, 23,  7,  6,  5), dtype = Index)
# ----------------------------------------------------------------------
# def log2_64(value: u64):
#     value |= value >> 1;
#     value |= value >> 2;
#     value |= value >> 4;
#     value |= value >> 8;
#     value |= value >> 16;
#     value |= value >> 32;
#     return _log2_tab64[((value - (value >> 1))*0x07EDD5E59A4E28C2)) >> 58];
# ======================================================================
@nb.njit
def _calc_interval[N_X: int,
                   N_Polys: int,
                   N_VarsTV: N_Vars,
                   N_CoeffsTV: N_Coeffs,
                   DType: (f32, f64)
                   ](i_x_low: Index,
                     i_x_high: Index,
                     x: Array[tuple[N_X], DType],
                     x_poly: DType,
                     coefficients: Array[tuple[N_Polys, N_VarsTV, N_CoeffsTV], DType],
                     i_poly: Index,
                     out: Array[tuple[N_X, N_VarsTV], DType]) -> None:

    for i_point in range(i_x_low, i_x_high):
        Dx = x[i_point] - x_poly,
        for i_var in range(coefficients.shape[1]):
            out[i_point, i_var] = single(Dx, coefficients[i_poly, i_var])
    return
# ======================================================================
@nb.njit
def batch_linear[N_X: int,
          N_Polys: int,
          N_VarsTV: N_Vars,
          N_CoeffsTV: N_Coeffs,
          DType: (f32, f64)
          ](x: Array[tuple[N_X], DType],
            x_coeffs: Array[tuple[int], DType],
            coefficients: Array[tuple[N_Polys, N_VarsTV, N_CoeffsTV], DType],
            out: Array[tuple[N_X, N_VarsTV], DType]) -> None:
    """Sorted x from smallest to largest."""
    i_poly_low = _0_Index
    i_poly_high = Index(len(coefficients))
    for i_x in range(len(x)):
        i_poly_low = _update_vars(i_x, x, x_coeffs, coefficients,
                              i_poly_low, i_poly_high, out)
# ======================================================================
@nb.njit
def batch[N_X: int,
          N_Polys: int,
          N_VarsTV: N_Vars,
          N_CoeffsTV: N_Coeffs,
          DType: (f32, f64)
          ](x: Array[tuple[N_X], DType],
            x_coeffs: Array[tuple[int], DType],
            coefficients: Array[tuple[N_Polys, N_VarsTV, N_CoeffsTV], DType],
            out: Array[tuple[N_X, N_VarsTV], DType]) -> None:

    i_poly_low = _0_Index
    i_poly_high = Index(len(coefficients))

    i_x_low = _0_Index
    # Find first x
    i_poly_low = _update_vars(i_x_low, x, x_coeffs, coefficients,
                              i_poly_low, i_poly_high)

    i_x_high = Index(len(x) - 1)
    if i_x_high == _0_Index:
        return

    i_poly_high = _update_vars(i_x_high, x, x_coeffs, coefficients,
                               i_poly_low, i_poly_high) + _1_Index
    if i_x_high == _1_Index:
        return

    # All in same interval
    if  i_poly_high - i_poly_low == _1_Index:
        _calc_interval(1,
                       i_x_high,
                       x,
                       x_coeffs[i_poly_low],
                       coefficients,
                       i_poly_low,
                       out)

    length_i_x_low = int(log2(i_x_high - i_x_low)) + 1
    indices_x_low = np.full(length_i_x_low, i_x_low, Index)
    indices_poly_low = np.full(length_i_x_low, i_poly_low, Index)
    i_x_max = i_x_high
    i_x_low += (i_x_high - i_x_low) >> 1
    i_i_x_low = _1_Index
    while i_x_high < i_x_max:
        while i_x_high:
            if  i_poly_high - i_poly_low == _1_Index: # All in same interval
                _calc_interval(i_x_low,
                               i_x_high,
                               x,
                               x_coeffs[i_poly_low],
                               coefficients,
                               i_poly_low,
                               out)
                i_i_x_low -= _1_Index
                i_x_low = indices_x_low[i_i_x_low]
                i_poly_low = indices_poly_low[i_i_x_low]
                i_poly_high = i_poly_low
            else:

                i_poly_low = _update_vars(i_x_high, x, x_coeffs, coefficients,
                               i_poly_low, i_poly_high)
                i_i_x_low += _1_Index
                indices_x_low[i_i_x_low] = i_x_low
                indices_poly_low[i_i_x_low] =


    i_x_mean = i_x_high // _2_Index

    i_poly_mean = _update_vars(i_x_mean, x, x_coeffs, coefficients,
                               i_poly_low, i_poly_high)
    if i_poly_high == _2_Index:
        return
