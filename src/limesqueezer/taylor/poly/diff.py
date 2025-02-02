from typing import TYPE_CHECKING

from .. import _lnumba as nb
from .._lnumpy import f64
from .._lnumpy import up
from .._types import Index
# ======================================================================
if TYPE_CHECKING:
    from .._lnumpy import F64Array
    from .._types import N_Vars
    from .._types import N_Coeffs
else:
    F64Array = tuple
    N_Vars = N_Coeffs = int
# ======================================================================
_0_Index = Index(0)
_1_Index = Index(1)
# ======================================================================
# @nb.njit
# def to_out(coeffs: F64Array[N_Coeffs, N_Vars],
#            out: F64Array[int, N_Vars]):
#     """Differentiates p_n * x ** n + p_{n-1} * x**{n-1} + ...

#     p_0
#     """
#     out[:] = coeffs[:-1] # Copies relevant part
#     n_out = Index(len(out))
#     multiplier = f64(n_out)
#     print(n_out - up(1))
#     for index in range(n_out - Index(1)):
#         out[index] *= multiplier
#         multiplier -= 1.
# ======================================================================
@nb.njit
def in_place_coeffs_vars(coeffs: F64Array[N_Coeffs, N_Vars], n_out: Index):
    """Differentiates p_n * x ** n + p_{n-1} * x**{n-1} + ... + p_0.

    Parameters
    ----------

    coeffs

    n_out
    """
    multiplier = f64(n_out)
    _1_f64 = f64(1.)
    for index in range(n_out - Index(1)):
        coeffs[index] *= multiplier
        multiplier -= _1_f64
# ======================================================================
@nb.njit
def in_place_vars_coeffs(coeffs: F64Array[N_Vars, N_Coeffs], n_out: Index):
    """Differentiates p_n * x ** n + p_{n-1} * x**{n-1} + ... + p_0.

    Parameters
    ----------

    coeffs :
        coefficient array

    n_out : int
        number of parameters in output, i.e. power for the first parameter
        in input
    """

    stop_coeffs = n_out - _1_Index
    stop_vars = Index(coeffs.shape[0])
    multiplier = f64(n_out)
    _1_f64 = f64(1.)
    i_coeff = _0_Index
    while i_coeff != stop_coeffs:
        i_var = _0_Index
        while i_var != stop_vars:
            coeffs[i_var, i_coeff] *= multiplier
            i_var += _1_Index
        multiplier -= _1_f64
        i_coeff += _1_Index
