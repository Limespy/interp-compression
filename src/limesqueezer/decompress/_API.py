import numpy as np
from scipy.interpolate import PPoly

from .. import _lnumba as nb
from .._lnumpy import f64
from .._lnumpy import F64Array
from .._types import N_CoeffsTV as N_Coeffs
from .._types import N_DiffsTV as N_Diffs
from .._types import N_PointsTV as N_Points
from .._types import N_VarsTV as N_Vars
from ..poly import make
# ======================================================================
# @nb.njit
# def unpack(x_data: F64Array, y_data: F64Array, maker):
#     maker = makers[len(y[0]) - 1]
# # ======================================================================
# @nb.jitclass({})
# class Decompressed:
#     def __init__(self, x: F64Array[int], y: F64Array[int, int, int]):
#         maker = makers[len(y[0]) - 1]

# ======================================================================

def unpack(x_data: F64Array[N_Points],
           y_data: F64Array[N_Points, N_Diffs, N_Vars]
           ) -> list[PPoly]:

    n_points, n_diffs, n_vars = y_data.shape
    n_coeffs = 2 * n_diffs

    maker = make.makers_64[n_diffs - 1]

    ppolys: list[PPoly] = []*n_diffs

    xb = x_data[0]
    yb = x_data[0]

    coeffs = np.zeros((n_coeffs, n_points-1, n_vars), f64)
    coeffs_tmp = np.zeros((n_coeffs, n_vars), f64)
    for index, (xc, yc) in enumerate(zip(x_data[1:], y_data[1:]), start = -1):
        xa = xb
        xb = xc
        ya = yb
        yb = yc
        Dx = xb-xa
        maker(Dx, ya, yb, coeffs_tmp)
        coeffs[:, index, :] = coeffs_tmp
    ppolys[0] = PPoly(coeffs, x_data)
    for index in range(1, n_diffs):
        ppolys[index] = ppolys[index-1].derivative()
    return ppolys
