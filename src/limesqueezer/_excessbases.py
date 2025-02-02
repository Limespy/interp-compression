from warnings import warn

from ._lnumpy import F64Array
from ._lnumpy import inf64
from ._types import Excess
from ._types import Index
from ._types import XSingle
from .exceptions import NotImplementedWarning
# ======================================================================
class _Excessclass[*YSingleShape]:
    # ------------------------------------------------------------------
    def _calc_excess(self, x0: XSingle, start: Index, stop: Index, step: Index
                     ) -> Excess:
        raise NotImplementedError()
    # ------------------------------------------------------------------
    def _minimum(self, y0_min: F64Array[*YSingleShape]) -> Excess: # type: ignore[type-var]
        warn('base class method', NotImplementedWarning, stacklevel = 2)
        return -inf64
# ======================================================================
# # ----------------------------------------------------------------------
# def AbsEnd(y_sample: F64Array,
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
#     return np.max(np.abs(y_approx[-1] - y_sample[-1]) - tolerance_total[-1])
# # ----------------------------------------------------------------------
# def MaxAbs(y_sample: F64Array,
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
#     return np.amax(np.fabs(y_approx - y_sample) - tolerance_total)
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
# def Threshold_MaxAbs_sequential(y_sample: F64Array,
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
#     n = 0
#     _max = -np.inf
#     for s, a, t in zip(y_sample, y_approx, tolerance_total):
#         _max = max(_max, abs(s - a) - t)
#         if _max > 0.:
#             return (False, n, 0.)
#         n += 1
#     return (True, n, _max)
# # ----------------------------------------------------------------------
# def MaxMAbs(y_sample: F64Array,
#                 y_approx: F64Array,
#                 tolerance_total: TolerancesInternal
#                 ) -> float:
#     """Maximum of mean excess errors relative to tolerance or maximum of the
#     end values.

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
#     residuals = np.abs(y_approx - y_sample)
#     return np.amax(np.mean(residuals - tolerance_total, 0))
# # ----------------------------------------------------------------------
# def MaxMAbs_AbsEnd(y_sample: F64Array,
#                 y_approx: F64Array,
#                 tolerance_total: TolerancesInternal
#                 ) -> float:
#     """Maximum of absolute orrors relative to tolerance.

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
#     residuals = np.abs(y_approx - y_sample)
#     excess = residuals - tolerance_total
#     excess_end = np.amax(excess[-1:])
#     excess_mean = np.amax(np.mean(excess, 0))
#     return excess_end if excess_end > excess_mean else excess_mean
# # ----------------------------------------------------------------------
# def MaxMS(y_sample: F64Array,
#                 y_approx: F64Array,
#                 tolerance_total: TolerancesInternal
#                 ) -> float:
#     '''Root mean square error.
#     1. Calculate residuals squared
#     2. Square root of mean along a column
#     3. Find largest of those difference to tolerance

#     Parameters
#     ----------
#     y_sample : F64Array
#         Y values of points of data selected for error calculation
#     y_approx : F64Array
#         Y values from fitting the model into data
#     tolerances : TolerancesInternal
#         Tolerances for errors
#         1) Relative tolerance array
#         2) Absolute tolerance array
#         3) Falloff array

#     Returns
#     -------
#     float
#         Error value. Should be <0 for fit to be acceptable
#     '''
#     residuals = y_approx - y_sample
#     return np.amax(np.mean(residuals * residuals - tolerance_total, 0))
# # ----------------------------------------------------------------------
# def MaxMS_SEnd(y_sample: F64Array,
#             y_approx: F64Array,
#             tolerance_total: TolerancesInternal
#             ) -> float:
#     """Intended to clamp the end point within absolute value of tolerance for
#     more stability. Returns bigger of:

#     - root mean square error
#     - end point maximum absolute error
#     1. Calculate endpoint maximum absolute error
#     2. Calculate residuals squared

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

#     residuals = y_approx - y_sample
#     # Excess at the last point
#     residuals *= residuals
#     excess = residuals - tolerance_total
#     excess_end = np.amax(excess[-1:])
#     excess_mean = np.amax(np.mean(excess, 0))
#     return excess_end if excess_end > excess_mean else excess_mean
