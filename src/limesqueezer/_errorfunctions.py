# from typing import Generic
from typing import TYPE_CHECKING

import numpy as np

from . import _lnumba as nb
from ._aux import py_and_nb
from ._lnumpy import up
from .poly import diff
from .poly.interpolate import get_interpolators
from .poly.interpolate import interpolate
from .poly.interpolate import nbInterpolatorsList
# from typing import TypeVar
# ======================================================================
# Hinting types
if TYPE_CHECKING:

    from typing import Protocol
    from typing import TypeAlias

    from ._aux import F64Array
    from ._aux import TolerancesInternal

    from .poly import Interpolator
    from ._typing import TolType
    from ._typing import XType
    from ._typing import YType

    class ErrorExcessFunction(Protocol):
        def __call__(self,
                     y_sample: F64Array,
                     y_approx: F64Array,
                     tolerances: TolerancesInternal) -> float:
            ...
    class ErrorThresholdFunction(Protocol):
        def __call__(self,
                     y_sample: F64Array,
                     y_approx: F64Array,
                     tolerances: TolerancesInternal) -> bool:
            ...
    N_Coeffs: TypeAlias = int
    N_Diffs: TypeAlias = int
    N_Points: TypeAlias = int
    N_Vars: TypeAlias = int
else:
    F64Array = Interpolator = tuple
    TolerancesInternal = ErrorExcessFunction = tuple

    ErrorThresholdFunction = object
    TolType = XType = YType = object
    N_Coeffs = N_Diffs = N_Points = N_Vars = int
# ----------------------------------------------------------------------

# ======================================================================
class _ErrBase:
    def __init__(self,
                 rtol: F64Array[int, int],
                 atol: F64Array[int, int]) -> None:
        self.rtol = rtol
        self.atol = atol
        self.n_diffs = up(rtol.shape[0])
        self.n_var = up(rtol.shape[1])
# ======================================================================
# @nb.jitclass({'rtol': nb.ARO(2),
#               'atol': nb.ARO(2),
#               'n_diffs': nb.size_t,
#               'n_var': nb.size_t})
class MaxAbs_Sequential(_ErrBase):
    def call(self, x: F64Array[N_Points],
             y: F64Array[N_Points, N_Diffs, N_Vars],
             coeffs: F64Array[N_Coeffs, N_Vars],
             x0: float,
             start: int, stop: int, step: int) -> float:
        print('MaxAbs_Sequential.call')
        excess = -np.inf
        n = up(2) * self.n_diffs
        n_var = y.shape[2]
        for i_diff in range(self.n_diffs):
            print('\ti_diff', i_diff)
            n -= up(1)
            for i_sample in range(start, stop, step):
                approxes: F64Array[N_Vars] = interpolate(x[i_sample] - x0,
                                                         coeffs,
                                                         n)
                for i_var in range(n_var):
                    sample = y[i_sample, i_diff, i_var]

                    excess = max(excess,
                                 abs(sample - approxes[i_var])
                                 - abs(sample) * self.rtol[i_diff, i_var]
                                 - self.atol[i_diff, i_var])
                    if i_var == 0:
                        print('\tsample', sample)
                        print('\terr', abs(sample - approxes[i_var]))
                        print('\texcess', excess)
            diff.in_place(coeffs, n)
        return excess
    # ------------------------------------------------------------------
    def minimum(self, y0_min: F64Array[N_Diffs, N_Vars]):
        excess = -np.inf
        for y_diff, r_diff, a_diff in zip(y0_min, self.rtol, self.atol):
            for y, r, a in zip(y_diff, r_diff, a_diff):
                excess = max(excess, - abs(y) * r - a)
        return excess
# ======================================================================
@nb.jitclass({'rtol': nb.ARO(2),
              'atol': nb.ARO(2),
              'n_diffs': nb.size_t})
class MaxAbs_Sequential2(_ErrBase):
    def call(self, x: F64Array[N_Points],
             y: F64Array[N_Points, N_Diffs, N_Vars],
             coeffs: F64Array[N_Diffs, N_Coeffs, N_Vars],
             x0: float,
             start: int, stop: int, step: int) -> float:
        excess = -np.inf
        n_start = np.uintp(2) * self.n_diffs
        n = n_start
        n_var = y.shape[2]
        for i_diff in range(self.n_diffs):
            for i_sample in range(start, stop, step):
                approxes: F64Array[N_Vars] = interpolate(x[i_sample] - x0,
                                                         coeffs[i_diff],
                                                         n)
                for i_var in range(n_var):
                    s = y[i_sample, i_diff, i_var]
                    excess = max(excess,
                                 abs(s - approxes[i_var])
                                 - abs(s) * self.rtol[i_diff, i_var]
                                 - self.atol[i_diff, i_var])
            n -= np.uintp(1)
        return excess
    # ------------------------------------------------------------------
    def minimum(self, y0_min: F64Array[N_Diffs, N_Vars]):
        excess = -np.inf
        for y_diff, r_diff, a_diff in zip(y0_min, self.rtol, self.atol):
            for y, r, a in zip(y_diff, r_diff, a_diff):
                excess = max(excess, - abs(y) * r - a)
        return excess
# ======================================================================
@nb.jitclass({'rtol': nb.ARO(2),
              'atol': nb.ARO(2),
              'n_diffs': nb.size_t})
class MaxAbs_Sequential3(_ErrBase):
    def call(self, x: F64Array[N_Points],
             y: F64Array[N_Points, N_Diffs, N_Vars],
             coeffs: F64Array[N_Coeffs, N_Vars],
             x0: float,
             start: int, stop: int, step: int) -> float:
        excess = -np.inf
        n = np.uintp(2) * self.n_diffs
        n_var = y.shape[2]
        for i_diff in range(self.n_diffs):
            for i_sample in range(start, stop, step):
                approxes: F64Array[N_Vars] = interpolate(x[i_sample] - x0,
                                                         coeffs,
                                                         n)
                for i_var in range(n_var):
                    s = y[i_sample, i_diff, i_var]
                    excess = max(excess,
                                 abs(s - approxes[i_var])
                                 - abs(s) * self.rtol[i_diff, i_var]
                                 - self.atol[i_diff, i_var])
            n -= np.uintp(1)
            diff.in_place(coeffs, n)
        return excess
    # ------------------------------------------------------------------
    def minimum(self, y0_min: F64Array[N_Diffs, N_Vars]):
        excess = -np.inf
        for y_diff, r_diff, a_diff in zip(y0_min, self.rtol, self.atol):
            for y, r, a in zip(y_diff, r_diff, a_diff):
                excess = max(excess, - abs(y) * r - a)
        return excess
# ======================================================================
# ----------------------------------------------------------------------
def AbsEnd(y_sample: F64Array,
                y_approx: F64Array,
                tolerance_total: TolerancesInternal
                ) -> float:
    """Maximum of absolute errors relative to tolerance.

    Parameters
    ----------
    y_sample : F64Array
        Y values of points of data selected for error calculation
    y_approx : F64Array
        Y values from fitting the model into data
    tolerances : TolerancesInternal
        Tolerances for errors
        1) Relative error array
        2) Absolute error array
        3) Falloff array

    Returns
    -------
    float
        Error value. Should be <0 for fit to be acceptable
    """
    return np.max(np.abs(y_approx[-1] - y_sample[-1]) - tolerance_total[-1])
# ----------------------------------------------------------------------
def MaxAbs(y_sample: F64Array,
                y_approx: F64Array,
                tolerance_total: TolerancesInternal
                ) -> float:
    """Maximum of absolute errors relative to tolerance.

    Parameters
    ----------
    y_sample : F64Array
        Y values of points of data selected for error calculation
    y_approx : F64Array
        Y values from fitting the model into data
    tolerances : TolerancesInternal
        Tolerances for errors
        1) Relative error array
        2) Absolute error array
        3) Falloff array

    Returns
    -------
    float
        Error value. Should be <0 for fit to be acceptable
    """
    return np.amax(np.fabs(y_approx - y_sample) - tolerance_total)
# ----------------------------------------------------------------------
# def MaxAbs_sequential(y_sample: F64Array,
#                 y_approx: F64Array,
#                 tolerance_total: TolerancesInternal
#                 ) -> float:
#     """Maximum of absolute errors relative to tolerance.

#     Parameters
#     ----------
#     y_sample : F64Array
#         Y values of points of data selected for error calculation
#     y_approx : F64Array
#         Y values from fitting the model into data
#     tolerances : TolerancesInternal
#         Tolerances for errors
#         1) Relative error array
#         2) Absolute error array
#         3) Falloff array

#     Returns
#     -------
#     float
#         Error value. Should be <0 for fit to be acceptable
#     """
#     _max = -np.inf
#     for s, a, t in zip(y_sample, y_approx, tolerance_total):
#         _max = max(_max, abs(s - a) - t)
#     return _max
# ----------------------------------------------------------------------
def Threshold_MaxAbs_sequential(y_sample: F64Array,
                y_approx: F64Array,
                tolerance_total: TolerancesInternal
                ) -> float:
    """Maximum of absolute errors relative to tolerance.

    Parameters
    ----------
    y_sample : F64Array
        Y values of points of data selected for error calculation
    y_approx : F64Array
        Y values from fitting the model into data
    tolerances : TolerancesInternal
        Tolerances for errors
        1) Relative error array
        2) Absolute error array
        3) Falloff array

    Returns
    -------
    float
        Error value. Should be <0 for fit to be acceptable
    """
    n = 0
    _max = -np.inf
    for s, a, t in zip(y_sample, y_approx, tolerance_total):
        _max = max(_max, abs(s - a) - t)
        if _max > 0.:
            return (False, n, 0.)
        n += 1
    return (True, n, _max)
# ----------------------------------------------------------------------
def MaxMAbs(y_sample: F64Array,
                y_approx: F64Array,
                tolerance_total: TolerancesInternal
                ) -> float:
    """Maximum of mean excess errors relative to tolerance or maximum of the
    end values.

    Parameters
    ----------
    y_sample : F64Array
        Y values of points of data selected for error calculation
    y_approx : F64Array
        Y values from fitting the model into data
    tolerances : TolerancesInternal
        Tolerances for errors
        1) Relative error array
        2) Absolute error array
        3) Falloff array

    Returns
    -------
    float
        Error value. Should be <0 for fit to be acceptable
    """
    residuals = np.abs(y_approx - y_sample)
    return np.amax(np.mean(residuals - tolerance_total, 0))
# ----------------------------------------------------------------------
def MaxMAbs_AbsEnd(y_sample: F64Array,
                y_approx: F64Array,
                tolerance_total: TolerancesInternal
                ) -> float:
    """Maximum of absolute orrors relative to tolerance.

    Parameters
    ----------
    y_sample : F64Array
        Y values of points of data selected for error calculation
    y_approx : F64Array
        Y values from fitting the model into data
    tolerances : TolerancesInternal
        Tolerances for errors
        1) Relative error array
        2) Absolute error array
        3) Falloff array

    Returns
    -------
    float
        Error value. Should be <0 for fit to be acceptable
    """
    residuals = np.abs(y_approx - y_sample)
    excess = residuals - tolerance_total
    excess_end = np.amax(excess[-1:])
    excess_mean = np.amax(np.mean(excess, 0))
    return excess_end if excess_end > excess_mean else excess_mean
# ----------------------------------------------------------------------
def MaxMS(y_sample: F64Array,
                y_approx: F64Array,
                tolerance_total: TolerancesInternal
                ) -> float:
    '''Root mean square error.
    1. Calculate residuals squared
    2. Square root of mean along a column
    3. Find largest of those difference to tolerance

    Parameters
    ----------
    y_sample : F64Array
        Y values of points of data selected for error calculation
    y_approx : F64Array
        Y values from fitting the model into data
    tolerances : TolerancesInternal
        Tolerances for errors
        1) Relative tolerance array
        2) Absolute tolerance array
        3) Falloff array

    Returns
    -------
    float
        Error value. Should be <0 for fit to be acceptable
    '''
    residuals = y_approx - y_sample
    return np.amax(np.mean(residuals * residuals - tolerance_total, 0))
# ----------------------------------------------------------------------
def MaxMS_SEnd(y_sample: F64Array,
            y_approx: F64Array,
            tolerance_total: TolerancesInternal
            ) -> float:
    """Intended to clamp the end point within absolute value of tolerance for
    more stability. Returns bigger of:

    - root mean square error
    - end point maximum absolute error
    1. Calculate endpoint maximum absolute error
    2. Calculate residuals squared

    Parameters
    ----------
    y_sample : F64Array
        Y values of points of data selected for error calculation
    y_approx : F64Array
        Y values from fitting the model into data
    tolerances : TolerancesInternal
        Tolerances for errors
        1) Relative error array
        2) Absolute error array
        3) Falloff array

    Returns
    -------
    float
        Error value. Should be <0 for fit to be acceptable
    """

    residuals = y_approx - y_sample
    # Excess at the last point
    residuals *= residuals
    excess = residuals - tolerance_total
    excess_end = np.amax(excess[-1:])
    excess_mean = np.amax(np.mean(excess, 0))
    return excess_end if excess_end > excess_mean else excess_mean
# ======================================================================
def _maxsumabs(residuals: F64Array, tolerance: float | F64Array) -> float:
    return np.amax(np.sum(np.abs(residuals) - tolerance) / tolerance)
# ----------------------------------------------------------------------

errorfunctions: dict[str, tuple[ErrorExcessFunction, ErrorExcessFunction]] = {
    f.__name__: py_and_nb(f) for f in
    (AbsEnd, MaxAbs, MaxMAbs, MaxMAbs_AbsEnd, MaxMS, MaxMS_SEnd)
    }
