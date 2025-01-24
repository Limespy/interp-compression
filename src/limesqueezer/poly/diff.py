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
    N_Vars = N_Coeffs = int
# ======================================================================
@nb.njit
def to_out(coeffs: F64Array[N_Coeffs, N_Vars],
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
@nb.njit
def in_place_coeffs_vars(coeffs: F64Array[N_Coeffs, N_Vars], n_out: int):
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
@nb.njit
def in_place_vars_coeffs(coeffs: F64Array[N_Vars, N_Coeffs], n_out: N_Coeffs):
    """Differentiates p_n * x ** n + p_{n-1} * x**{n-1} + ... + p_0.

    Parameters
    ----------

    coeffs :
        coefficient array

    n_out : int
        number of parameters in output, i.e. power for the first parameter
        in input
    """

    n_coeffs = n_out - 1
    n_vars = coeffs.shape[0]
    multiplier = f64(n_out)
    for i_coeff in range(n_coeffs):
        for i_var in range(n_vars):
            coeffs[i_var, i_coeff] *= multiplier
        multiplier -= 1.
