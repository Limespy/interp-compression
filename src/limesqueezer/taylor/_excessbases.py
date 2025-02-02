from typing import override

import numpy as np

from .._excessbases import _Excessclass
from .._lnumpy import F64Array
from .._lnumpy import inf64
from .._types import Coeffs
from .._types import Excess
from .._types import Index
from .._types import N_Diffs
from .._types import N_Vars
from .._types import TolsDiff
from .._types import X
from .._types import XSingle
from .._types import YDiff
from .poly import diff
from .poly import interpolate
# ======================================================================
_0_Index = Index(0)
_1_Index = Index(1)
# ======================================================================
class _MaxAbs[N_DiffsTV: N_Diffs, N_VarsTV: N_Vars](_Excessclass):
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
