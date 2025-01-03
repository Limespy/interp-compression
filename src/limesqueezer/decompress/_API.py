from typing import TYPE_CHECKING

import numpy as np

from .. import _lnumba as nb
from ..poly import makers
# ======================================================================
# Hinting types
if TYPE_CHECKING:
    from collections.abc import Callable
    from typing import TypeVar
    from .._lnumpy import F64Array


    N_Points = TypeVar('N_Points', int)
    N_Diffs = TypeVar('N_Diffs', int)
    N_Vars = TypeVar('N_Points', int)
    N_Coeffs = TypeVar('N_Coeffs', int)
else:
    Callable = object
    F64Array = object
    N_Points = N_Diffs = N_Vars = N_Coeffs = object
# ======================================================================
@nb.njit
def unpack(x_data: F64Array, y_data: F64Array, maker: ):
    maker = makers[len(y[0]) - 1]
# ======================================================================
@nb.jitclass({})
class Decompressed:
    def __init__(self, x: F64Array[int], y: F64Array[int, int, int]):
        maker = makers[len(y[0]) - 1]

# ======================================================================

def unpack(x_data: F64Array[N_Points],
           y_data: F64Array[N_Points, N_Diffs, N_Vars]):
    len_diffs = len(y_data[0])
    maker: Callable[[]] = makers[len_diffs]

    ppolys = []*len_diffs

    coeffs:
    for index, _Dx in enumerate(Dx, start = 1):
        maker()

    for _ in range(1, len_diffs):

    return [PPoly(coefficients, x)
