from typing import NamedTuple
from typing import TYPE_CHECKING

import numpy as np

from ._aux import py_and_nb
# ======================================================================
# Hinting types
if TYPE_CHECKING:
    from collections.abc import Callable
    from ._aux import Float64Array
    from ._aux import MaybeArray

    FitFunction = Callable[[Float64Array, Float64Array, float, Float64Array], Float64Array]
    Interpolator = Callable[[MaybeArray, float, float, Float64Array, Float64Array], Float64Array]
else:
    Float64Array = MaybeArray = FitFunction = Interpolator = object
# ======================================================================
# BUILTIN COMPRESSION MODELS
class FitSet(NamedTuple):
     fit: tuple[FitFunction, FitFunction]
     interpolate: tuple[Interpolator, Interpolator]
# ----------------------------------------------------------------------
def _fit_Poly10(x: Float64Array,
                y: Float64Array,
                x0: float,
                y0: MaybeArray,
                ) -> Float64Array:
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

    Dx: Float64Array = x - x0
    Dy: Float64Array = y - y0
    return np.outer(Dx, Dx @ Dy / Dx.dot(Dx)) + y0
# ----------------------------------------------------------------------
def _interp_Poly10(x: MaybeArray,
                   x1: float, x2:float,
                   y1: Float64Array, y2: Float64Array
                     ) -> Float64Array:
        """Interpolates between two consecutive points of compressed data."""
        return (y2 - y1) / (x2 - x1) * (x - x1) + y1
# ----------------------------------------------------------------------
Poly10 = FitSet(py_and_nb(_fit_Poly10), py_and_nb(_interp_Poly10))
