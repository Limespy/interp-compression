from typing import Literal as L

from .. import _lnumba as nb
from .._lnumpy import F64Array
from .._types import N_Points
from .._types import N_Vars
from .sequential import Sequential0_64
from .sequential import Sequential1_64
from .sequential import Sequential2_64
from .sequential import Sequential3_64
# ======================================================================
# def _make_batch_compressor(i_diff):
#     Compressor = sequentials_64[i_diff]
#     @nb.njitC
#     def inner[N_VarsTV: N_Vars,
#               N_Points_In: N_Points,
#               N_Points_Out: N_Points
#               ](x: F64Array[N_Points_In],
#                 y: F64Array[N_Points_In, L[4], N_VarsTV],
#                 rtol: F64Array[L[4], N_VarsTV],
#                 atol: F64Array[L[4], N_VarsTV]
#                 ) -> tuple[F64Array[N_Points_Out],
#                             F64Array[N_Points_Out, L[4], N_VarsTV]]:
#         len_x = len(x)
#         stream = Compressor(rtol, atol, len_x // 2)
#         stream.open(x[0], y[0], 5)
#         for index in range(1, len_x):
#             stream.append(x[index], y[index])
#         stream.close()
#         return stream.x, stream.y # type: ignore[return-value]
#     inner.__name__ = f'batch{i_diff}_64'
#     return inner
# ======================================================================
@nb.njitC
def batch0_64[N_VarsTV: N_Vars,
                      N_Points_In: N_Points,
                      N_Points_Out: N_Points
                      ](x: F64Array[N_Points_In],
             y: F64Array[N_Points_In, L[1], N_VarsTV],
             rtol: F64Array[L[1], N_VarsTV],
             atol: F64Array[L[1], N_VarsTV]
             ) -> tuple[F64Array[N_Points_Out],
                        F64Array[N_Points_Out, L[1], N_VarsTV]]:
    len_x = len(x)
    stream = Sequential0_64(rtol, atol, len_x // 2)
    stream.open(x[0], y[0], 5)
    for index in range(1, len_x):
        stream.append(x[index], y[index])
    stream.close()
    return stream.x, stream.y # type: ignore[return-value]
# ======================================================================
@nb.njitC
def batch1_64[N_VarsTV: N_Vars,
                      N_Points_In: N_Points,
                      N_Points_Out: N_Points
                      ](x: F64Array[N_Points_In],
             y: F64Array[N_Points_In, L[2], N_VarsTV],
             rtol: F64Array[L[2], N_VarsTV],
             atol: F64Array[L[2], N_VarsTV]
             ) -> tuple[F64Array[N_Points_Out],
                        F64Array[N_Points_Out, L[2], N_VarsTV]]:
    len_x = len(x)
    stream = Sequential1_64(rtol, atol, len_x // 2)
    stream.open(x[0], y[0], 5)
    for index in range(1, len_x):
        stream.append(x[index], y[index])
    stream.close()
    return stream.x, stream.y # type: ignore[return-value]
# ======================================================================
@nb.njitC
def batch2_64[N_VarsTV: N_Vars,
                      N_Points_In: N_Points,
                      N_Points_Out: N_Points
                      ](x: F64Array[N_Points_In],
             y: F64Array[N_Points_In, L[3], N_VarsTV],
             rtol: F64Array[L[3], N_VarsTV],
             atol: F64Array[L[3], N_VarsTV]
             ) -> tuple[F64Array[N_Points_Out],
                        F64Array[N_Points_Out, L[3], N_VarsTV]]:
    len_x = len(x)
    stream = Sequential2_64(rtol, atol, len_x // 2)
    stream.open(x[0], y[0], 5)
    for index in range(1, len_x):
        stream.append(x[index], y[index])
    stream.close()
    return stream.x, stream.y # type: ignore[return-value]
# ======================================================================
@nb.njitC
def batch3_64[N_VarsTV: N_Vars,
                      N_Points_In: N_Points,
                      N_Points_Out: N_Points
                      ](x: F64Array[N_Points_In],
             y: F64Array[N_Points_In, L[4], N_VarsTV],
             rtol: F64Array[L[4], N_VarsTV],
             atol: F64Array[L[4], N_VarsTV]
             ) -> tuple[F64Array[N_Points_Out],
                        F64Array[N_Points_Out, L[4], N_VarsTV]]:
    len_x = len(x)
    stream = Sequential3_64(rtol, atol, len_x // 2)
    stream.open(x[0], y[0], 5)
    for index in range(1, len_x):
        stream.append(x[index], y[index])
    stream.close()
    return stream.x, stream.y # type: ignore[return-value]
# ======================================================================
batch_compressors_64 = (batch0_64,
                        batch1_64,
                        batch2_64,
                        batch3_64)
# ======================================================================
