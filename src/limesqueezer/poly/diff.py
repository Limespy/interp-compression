from typing import TYPE_CHECKING

from .. import _lnumba as nb
from .._lnumpy import f64
from .._lnumpy import up
# ======================================================================
if TYPE_CHECKING:
    from .._lnumpy import F64Array
    from .._typing import N_Vars
else:
    F64Array = tuple
    N_Vars = int
# ======================================================================
@nb.njit
def to_out(coeffs: F64Array[int, N_Vars],
           out: F64Array[int, N_Vars],
           c_stop: int = 0):
    """Differentiates p_n * x ** n + p_{n-1} * x**{n-1} + ...

    p_0
    """
    out[:] = coeffs[:-1] # Copies relevant part
    n_out = len(out) - c_stop
    multiplier = f64(n_out)
    for index in range(n_out - up(1)):
        out[index] *= multiplier
        multiplier -= 1.
# ======================================================================
@nb.njit(nb.void(nb.f64[:,:], nb.size_t))
def in_place(coeffs: F64Array[int, N_Vars], n_out: int):
    """Differentiates p_n * x ** n + p_{n-1} * x**{n-1} + ...

    p_0
    """
    n_out -= up(1)
    multiplier = f64(n_out)
    for index in range(n_out):
        coeffs[index] *= multiplier
        multiplier -= 1.
# ======================================================================
@nb.njit(nb.void(nb.f64[:,:], nb.size_t))
def finish(coeffs: F64Array, n_out: int):
    """Differentiates p_n * x ** n + p_{n-1} * x**{n-1} + ...

    p_0
    """
    n_out -= up(1)
    multiplier = f64(n_out)
    for index in range(n_out):
        coeffs[index] *= multiplier
        multiplier -= 1.
