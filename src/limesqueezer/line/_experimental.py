import numpy as np

from .._lnumpy import F64Array
# ======================================================================
# BUILTIN COMPRESSION MODELS
# class FitSet(NamedTuple):
#      fit: tuple[FitFunction, FitFunction]
#      interpolate: tuple[Interpolator, Interpolator]
# ----------------------------------------------------------------------
def _fit_Poly10(x: F64Array,
                y: F64Array,
                x0: float,
                y0: F64Array,
                ) -> F64Array:
    """Takes block of data, previous fitting parameters and calculates next
    fitting parameters.

    Parameters
    ----------
    x : Float64Array
        x values of the points to be fitted
    y : Float64Array
        y values of the points to be fitted
    x0 : float
        Last compressed point x value
    y0 : Float64Array
        Last compressed point y value(s)

    Returns
    -------
    Float64Array
        Fitted y-values
    """

    Dx: F64Array = x - x0
    Dy: F64Array = y - y0
    return np.outer(Dx, Dx @ Dy / Dx.dot(Dx)) + y0
# ----------------------------------------------------------------------
def _interp_Poly10(x: F64Array,
                   x1: float, x2:float,
                   y1: F64Array, y2: F64Array
                     ) -> F64Array:
        """Interpolates between two consecutive points of compressed data."""
        return (y2 - y1) / (x2 - x1) * (x - x1) + y1
