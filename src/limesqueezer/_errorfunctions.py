from typing import override
from warnings import warn

import numpy as np

from ._lnumpy import F64Array
from ._lnumpy import inf64
from ._types import Coeffs
from ._types import Excess
from ._types import Index
from ._types import N_Diffs
from ._types import N_Vars
from ._types import TolsDiff
from ._types import TolsLine
from ._types import X
from ._types import XSingle
from ._types import YDiff
from ._types import YLine
from .exceptions import NotImplementedWarning
from .poly import diff
from .poly import interpolate
# ======================================================================
_0_Index = Index(0)
_1_Index = Index(1)
# ======================================================================
class _Errorclass[*YSingleShape]:
    # ------------------------------------------------------------------
    def _calc_excess(self, x0: XSingle, start: Index, stop: Index, step: Index
                     ) -> Excess:
        raise NotImplementedError()
    # ------------------------------------------------------------------
    def _minimum(self, y0_min: F64Array[*YSingleShape]) -> Excess: # type: ignore[type-var]
        warn('base class method', NotImplementedWarning, stacklevel = 2)
        return -inf64
# ======================================================================
class _MaxAbs_Taylor[N_DiffsTV: N_Diffs, N_VarsTV: N_Vars](_Errorclass):
    rtol: TolsDiff
    atol: TolsDiff
    _n_coeffs: Index
    _n_diffs: Index
    _n_vars: Index
    coeffs: Coeffs
    _x: X
    _y: YDiff
    # ------------------------------------------------------------------
    def __init__(self,
                 rtol: F64Array[N_DiffsTV, N_VarsTV],
                 atol: F64Array[N_DiffsTV, N_VarsTV]) -> None:
        self.__init(rtol, atol)
    # ------------------------------------------------------------------
    def __init(self,
               rtol: F64Array[N_DiffsTV, N_VarsTV],
               atol: F64Array[N_DiffsTV, N_VarsTV]) -> None:
        self.rtol = -rtol
        self.atol = -atol
        n_diffs, n_vars = rtol.shape
        self._n_diffs: Index = Index(n_diffs)
        self._n_vars: Index = Index(n_vars)
        self._n_coeffs: Index = Index(2 * n_diffs)
        self.coeffs = np.zeros((self._n_vars, self._n_coeffs))
    # ------------------------------------------------------------------
    def _calc_sample_excess(self, Dx: XSingle,
                            i_diff: Index,
                            n: Index,
                            i_sample: Index,
                            stop_var: Index,
                            excess: Excess) -> Excess:
        i_var: Index = _0_Index
        while i_var != stop_var:
            sample = self._y[i_sample, i_diff, i_var]
            approx = interpolate.single(Dx, self.coeffs[i_var], n)
            excess = max(excess,
                            abs(sample - approx)
                            + abs(sample) * self.rtol[i_diff, i_var]
                            + self.atol[i_diff, i_var])
            i_var += _1_Index
        return excess
    # ------------------------------------------------------------------
    def _calc_diff_excess(self,
                          x0: XSingle,
                          start_sample: Index,
                          stop_sample: Index,
                          step_sample: Index,
                          i_diff: Index,
                          n: Index,
                          excess: Excess) -> Excess:
        i_sample = start_sample
        stop_var = self._n_vars
        while i_sample < stop_sample:
            Dx: XSingle = self._x[i_sample] - x0
            excess = self._calc_sample_excess(
                Dx, i_diff, n, i_sample, stop_var, excess)
            i_sample += step_sample
        return excess
    # ------------------------------------------------------------------
    @override
    def _calc_excess(self, x0: XSingle, start: Index, stop: Index, step: Index
                     ) -> Excess:
        n: Index = self._n_coeffs - _1_Index
        excess = self._calc_diff_excess(x0, start, stop, step, _0_Index, n, -inf64)
        i_diff: Index = _1_Index
        stop_diff = self._n_diffs
        while i_diff != stop_diff:
            diff.in_place_vars_coeffs(self.coeffs, n)
            n -= _1_Index
            excess = self._calc_diff_excess(x0, start, stop, step, i_diff, n,
                                            excess)
            i_diff += _1_Index
        return excess
    # ------------------------------------------------------------------
    @override
    def _minimum(self, y0_min: F64Array[N_DiffsTV, N_VarsTV]) -> Excess:
        excess = -inf64
        stop_diff = self._n_diffs
        stop_var = self._n_vars

        i_diff: Index = _0_Index
        while i_diff != stop_diff:
            i_var: Index = _0_Index
            while i_var != stop_var:
                excess = max(excess,
                             (abs(y0_min[i_diff, i_var])
                              * self.rtol[i_diff, i_var])
                             + self.atol[i_diff, i_var])
                i_var += _1_Index
            i_diff += _1_Index
        return excess
# ======================================================================
class _MaxAbs_Line[N_VarsTV: N_Vars](_Errorclass):
    rtol: TolsLine
    atol: TolsLine
    _n_vars: Index
    coeffs: Coeffs
    _x: X
    _y: YLine
    # ------------------------------------------------------------------
    def __init__(self,
                 rtol: F64Array[N_VarsTV],
                 atol: F64Array[N_VarsTV]) -> None:
        self.__init(rtol, atol)
    # ------------------------------------------------------------------
    def __init(self,
               rtol: F64Array[N_VarsTV],
               atol: F64Array[N_VarsTV]) -> None:
        self.rtol = -rtol
        self.atol = -atol
        self._n_vars: Index = Index(rtol.shape[0])
        self.coeffs = np.zeros((self._n_vars, 2))
    # ------------------------------------------------------------------
    @override
    def _calc_excess(self, x0: XSingle, start: Index, stop: Index, step: Index
                     ) -> Excess:
        i_sample = start
        stop_var = self._n_vars
        excess = -inf64

        while i_sample < stop:
            Dx: XSingle = self._x[i_sample] - x0
            i_var = _0_Index
            while i_var < stop_var:
                sample = self._y[i_sample, i_var]
                p1, p0 = self.coeffs[i_var]
                approx = p1 * Dx + p0
                excess = max(excess,
                                abs(sample - approx)
                                + abs(sample) * self.rtol[i_var]
                                + self.atol[i_var])
                i_var += _1_Index
            i_sample += step
        return excess
    # ------------------------------------------------------------------
    @override
    def _minimum(self, y0_min: F64Array[N_VarsTV]) -> Excess:
        excess = (abs(y0_min[_0_Index]) * self.rtol[_0_Index]
                  + self.atol[_0_Index])

        i_var = _1_Index
        stop_var = self._n_vars
        while i_var != stop_var:
            excess = max(excess,
                         abs(y0_min[i_var]) * self.rtol[i_var]
                         + self.atol[i_var])
            i_var += _1_Index
        return excess
# class MaxAbs_Diff_Base_old(MaxAbs_Diff_Base):
#     # ------------------------------------------------------------------
#     def __init(self,
#                rtol: F64Array[N_Diffs, N_Vars],
#                atol: F64Array[N_Diffs, N_Vars]) -> None:
#         self.rtol = rtol
#         self.atol = atol
#         n_diffs, n_vars = rtol.shape
#         self._n_diffs: N_Diffs = up(n_diffs)
#         self._n_vars: N_Vars = up(n_vars)
#         self._n_coeffs: N_Coeffs = up(2 * n_diffs)
#         self.coeffs = np.zeros((self._n_coeffs, self._n_vars))
#     # ------------------------------------------------------------------
#     def _calc_excess(self, x0: XSingle, start: Index, stop: Index, step: Index
#                      ) -> Excess:
#         excess = -np.inf
#         n = self._n_coeffs
#         approxes = np.empty((self._n_vars, ), f64)
#         for i_diff in range(self._n_diffs):
#             n -= up(1)
#             for i_sample in range(start, stop, step):
#                 interpolate.group(self._x[i_sample] - x0, self.coeffs, n, approxes)
#                 for i_var in range(self._n_vars):
#                     sample = self._y[i_sample, i_diff, i_var]

#                     excess = max(excess,
#                                     abs(sample - approxes[i_var])
#                                     - abs(sample) * self.rtol[i_diff, i_var]
#                                     - self.atol[i_diff, i_var])
#             diff.in_place_coeffs_vars(self.coeffs, n)
#         return excess
# ======================================================================
# class _ErrBase:
#     def __init__(self,
#                  rtol: F64Array[int, int],
#                  atol: F64Array[int, int]) -> None:
#         self.rtol = rtol
#         self.atol = atol
#         self.n_diffs = up(rtol.shape[0])
#         self.n_var = up(rtol.shape[1])
# # ======================================================================
# @nb.jitclass({'rtol': nb.ARO(2),
#               'atol': nb.ARO(2),
#               'n_diffs': nb.size_t,
#               'n_var': nb.size_t})
# class MaxAbs_Diff(_ErrBase):
#     def call(self, x: F64Array[N_Points],
#              y: F64Array[N_Points, N_Diffs, N_Vars],
#              coeffs: F64Array[N_Coeffs, N_Vars],
#              x0: float,
#              start: int, stop: int, step: int) -> float:
#         # print('MaxAbs_Diff.call')
#         excess = -np.inf
#         n = up(2) * self.n_diffs
#         n_var = y.shape[2]
#         for i_diff in range(self.n_diffs):
#             # print('\ti_diff', i_diff)
#             n -= up(1)
#             for i_sample in range(start, stop, step):
#                 approxes: F64Array[N_Vars] = interpolate.group(x[i_sample] - x0,
#                                                          coeffs,
#                                                          n)
#                 for i_var in range(n_var):
#                     sample = y[i_sample, i_diff, i_var]

#                     excess = max(excess,
#                                  abs(sample - approxes[i_var])
#                                  - abs(sample) * self.rtol[i_diff, i_var]
#                                  - self.atol[i_diff, i_var])
#                     # if i_var == 0:
#                     #     print('\tsample', sample)
#                     #     print('\terr', abs(sample - approxes[i_var]))
#                     #     print('\texcess', excess)
#             diff.in_place(coeffs, n)
#         return excess
#     # ------------------------------------------------------------------
#     def minimum(self, y0_min: F64Array[N_Diffs, N_Vars]):
#         excess = -np.inf
#         for y_diff, r_diff, a_diff in zip(y0_min, self.rtol, self.atol):
#             for y, r, a in zip(y_diff, r_diff, a_diff):
#                 excess = max(excess, - abs(y) * r - a)
#         return excess
# # ======================================================================
# @nb.jitclass({'rtol': nb.ARO(2),
#               'atol': nb.ARO(2),
#               'n_diffs': nb.size_t})
# class MaxAbs_Diff2(_ErrBase):
#     def call(self, x: F64Array[N_Points],
#              y: F64Array[N_Points, N_Diffs, N_Vars],
#              coeffs: F64Array[N_Diffs, N_Coeffs, N_Vars],
#              x0: float,
#              start: int, stop: int, step: int) -> float:
#         """Already differentiated"""
#         excess = -np.inf
#         n_start = np.uintp(2) * self.n_diffs
#         n = n_start
#         n_var = y.shape[2]
#         for i_diff in range(self.n_diffs):
#             for i_sample in range(start, stop, step):
#                 approxes: F64Array[N_Vars] = interpolate(x[i_sample] - x0,
#                                                          coeffs[i_diff],
#                                                          n)
#                 for i_var in range(n_var):
#                     s = y[i_sample, i_diff, i_var]
#                     excess = max(excess,
#                                  abs(s - approxes[i_var])
#                                  - abs(s) * self.rtol[i_diff, i_var]
#                                  - self.atol[i_diff, i_var])
#             n -= np.uintp(1)
#         return excess
#     # ------------------------------------------------------------------
#     def minimum(self, y0_min: F64Array[N_Diffs, N_Vars]):
#         excess = -np.inf
#         for y_diff, r_diff, a_diff in zip(y0_min, self.rtol, self.atol):
#             for y, r, a in zip(y_diff, r_diff, a_diff):
#                 excess = max(excess, - abs(y) * r - a)
#         return excess
# # ======================================================================
# @nb.jitclass({'rtol': nb.ARO(2),
#               'atol': nb.ARO(2),
#               'n_diffs': nb.size_t})
# class MaxAbs_Diff3(_ErrBase):
#     def call(self, x: F64Array[N_Points],
#              y: F64Array[N_Points, N_Diffs, N_Vars],
#              coeffs: F64Array[N_Coeffs, N_Vars],
#              x0: float,
#              start: int, stop: int, step: int) -> float:
#         excess = -np.inf
#         n = np.uintp(2) * self.n_diffs
#         n_var = y.shape[2]
#         for i_diff in range(self.n_diffs):
#             for i_sample in range(start, stop, step):
#                 approxes: F64Array[N_Vars] = interpolate.group(x[i_sample] - x0,
#                                                          coeffs,
#                                                          n)
#                 for i_var in range(n_var):
#                     s = y[i_sample, i_diff, i_var]
#                     excess = max(excess,
#                                  abs(s - approxes[i_var])
#                                  - abs(s) * self.rtol[i_diff, i_var]
#                                  - self.atol[i_diff, i_var])
#             n -= np.uintp(1)
#             diff.in_place(coeffs, n)
#         return excess
#     # ------------------------------------------------------------------
#     def minimum(self, y0_min: F64Array[N_Diffs, N_Vars]):
#         excess = -np.inf
#         for y_diff, r_diff, a_diff in zip(y0_min, self.rtol, self.atol):
#             for y, r, a in zip(y_diff, r_diff, a_diff):
#                 excess = max(excess, - abs(y) * r - a)
#         return excess
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
# # ======================================================================
# def _maxsumabs(residuals: F64Array, tolerance: float | F64Array) -> float:
#     return np.amax(np.sum(np.abs(residuals) - tolerance) / tolerance)
# # ----------------------------------------------------------------------

# errorfunctions: dict[str, tuple[ErrorExcessFunction, ErrorExcessFunction]] = {
#     f.__name__: py_and_nb(f) for f in
#     (AbsEnd, MaxAbs, MaxMAbs, MaxMAbs_AbsEnd, MaxMS, MaxMS_SEnd)
#     }
