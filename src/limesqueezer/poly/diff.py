from typing import TYPE_CHECKING

from .. import _lnumba as nb
from .._lnumpy import f64
from .._lnumpy import up
# ======================================================================
if TYPE_CHECKING:
    from .._lnumpy import F64Array
    from .._typing import N_Vars
    from .._typing import N_Coeffs
else:
    F64Array = tuple
    N_Vars = int
# ======================================================================
@nb.njit
def to_out(coeffs: F64Array[int, N_Vars],
           out: F64Array[int, N_Vars]):
    """Differentiates p_n * x ** n + p_{n-1} * x**{n-1} + ...

    p_0
    """
    out[:] = coeffs[:-1] # Copies relevant part
    n_out = len(out)
    multiplier = f64(n_out)
    for index in range(n_out - up(1)):
        out[index] *= multiplier
        multiplier -= 1.
# ======================================================================
@nb.njit(nb.void(nb.f64[:,:], nb.size_t))
def in_place(coeffs: F64Array[int, N_Vars], n_out: int):
    """Differentiates p_n * x ** n + p_{n-1} * x**{n-1} + ... + p_0.

    Parameters
    ----------

    coeffs

    n_out
    """
    multiplier = f64(n_out)
    for index in range(n_out - 1):
        coeffs[index] *= multiplier
        multiplier -= 1.
# ======================================================================
