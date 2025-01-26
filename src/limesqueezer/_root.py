from typing import overload

from numpy import reciprocal

from . import _lnumba as nb
from ._types import fIndex
from ._types import Index
# ======================================================================
f32 = nb.f32
f64 = nb.f64
_1_fIndex = fIndex(1.)
# ======================================================================
@overload
def linear(x0: f32, y0: f32, x1: f32, y1: f32) -> f32:
    ...
@overload
def linear(x0: f64, y0: f64, x1: f64, y1: f64) -> f64: # type: ignore[overload-cannot-match]
    ...
@nb.njit
def linear(x0, y0, x1, y1):
    """Calculates x such the f(x) = 0 from two points ((x0, y0), (x1, y1))
    using a line i.e. first degree polynomial P(x) = p1 * x + p0. Depending on
    how the endpoints are selected, is a secant method step or a regula falsi
    step.

    Parameters
    ----------
    x0 : _type_
        _description_
    y0 : _type_
        _description_
    x1 : _type_
        _description_
    y1 : _type_
        _description_

    Returns
    -------
    _type_
        _description_
    """
    return (x1 * y0 - x0 * y1) / (y0 - y1)
# ----------------------------------------------------------------------
@overload
def poly(x0: f32, y0: f32, x1: f32, y1: f32, n: f32) -> f32:
    ...
@overload
def poly(x0: f64, y0: f64, x1: f64, y1: f64, n: f64) -> f64: # type: ignore[overload-cannot-match]
    ...
@nb.njit
def poly(x0, y0, x1, y1, n):
    """Calculates x such the f(x) = 0 from two points ((x0, y0), (x1, y1))
    using a polynomial of form P(x) = p1 * x^n + p0.

    Parameters
    ----------
    x0 : _type_
        _description_
    y0 : _type_
        _description_
    x1 : _type_
        _description_
    y1 : _type_
        _description_

    Returns
    -------
    _type_
        _description_
    """
    # p1 * x**n + p0 = 0
    # x**n = -p0 / p1
    # x = (-p0 / p1)**1/n

    # p1 * x0**n + p0 = y0
    # p1 * x1**n + p0 = y1
    # -> p1 * (x1**n - x0**n) = y1 - y0
    # -> p1 = (y1 - y0) / (x1**n - x0**n)
    # -> - p0 / p1 = x0**n  - y0 / p1
    # = x0**n - y0 * (x1**n - x0**n) / (y1 - y0)
    # = (x0**n * y1 - x0**n * y0 - x1**n * y0 + x0**n * y0) / (y1 - y0)
    # = (x1**n * y0 - x0**n * y1) / (y0 - y1)
    return abs((x1 ** n * y0 - x0 ** n * y1) / (y0 - y1)
               )**reciprocal(n)
# ----------------------------------------------------------------------
@overload
def shift_rf(x_low: f32, x_high: f32, estimate: f32) -> f32:
    ...
@overload
def shift_rf(x_low: f64, x_high: f64, estimate: f64) -> f64: # type: ignore[overload-cannot-match]
    ...
@nb.njit
def shift_rf(x_low,  x_high, estimate):
    diff_to_low = estimate - x_low
    diff_to_high = x_high - estimate
    return (estimate + diff_to_low * 0.5
            if diff_to_low < diff_to_high else
            estimate - diff_to_high * 0.5)
# ----------------------------------------------------------------------
