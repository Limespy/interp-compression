from typing import TypeAlias

import numpy as np
from scipy.interpolate import PPoly

from .. import _lnumba as nb
from .._lnumpy import Array
from .._lnumpy import f32
from .._lnumpy import f64
from .._lnumpy import F64Array
from .._types import N_Coeffs
from .._types import N_Diffs
from .._types import N_Points
from .._types import N_Vars
from .poly import make
# ======================================================================
N_Polys: TypeAlias = int
# ======================================================================
@nb.njit
def _unpack[N_PointsTV: N_Points,
            N_DiffsTV: N_Diffs,
            N_CoeffsTV: N_Coeffs,
            N_VarsTV: N_Vars,
            DType: (f32, f64)
            ](x: Array[tuple[N_PointsTV], DType],
              y: Array[tuple[N_PointsTV, N_DiffsTV, N_VarsTV], DType],
              maker: make.Make[N_DiffsTV, N_CoeffsTV, N_VarsTV, DType],
              coeffs: Array[tuple[N_Coeffs, N_Polys, N_Vars], DType],
              coeffs_tmp: Array[tuple[N_Vars, N_Coeffs], DType]
              ) -> None:
    n_vars, n_coeffs = coeffs_tmp.shape
    xb = x[0]
    yb = y[0]
    for index in range(1, len(x)):
        xa = xb
        ya = yb
        xb = x[index]
        yb = y[index]
        Dx = xb - xa
        maker(Dx, ya, yb, coeffs_tmp)
        # insert coeffs
        for i_var in range(n_vars):
            for i_coeff in range(n_coeffs):
                coeffs[i_coeff, index, i_var] = coeffs_tmp[i_var, i_coeff]
# ======================================================================
def unpack[N_PointsTV: N_Points,
           DType: (f32, f64)
           ](x_data: Array[tuple[N_PointsTV], DType],
           y_data: Array[tuple[N_PointsTV, N_Diffs, N_Vars], DType],
           dtype: DType) -> list[PPoly]:

    n_points, n_diffs, n_vars = y_data.shape
    n_coeffs = 2 * n_diffs

    maker = make.makers_64[n_diffs - 1]

    ppolys: list[PPoly] = []*n_diffs

    coeffs = np.zeros((n_coeffs, n_points-1, n_vars), dtype)
    coeffs_tmp = np.zeros((n_vars, n_coeffs), dtype)
    _unpack(x_data, y_data, maker, coeffs, coeffs_tmp)
    ppolys[0] = PPoly(coeffs, x_data)
    for index in range(1, n_diffs):
        ppolys[index] = ppolys[index-1].derivative()
    return ppolys
