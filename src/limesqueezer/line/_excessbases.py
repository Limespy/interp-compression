from typing import override

import numpy as np

from .._excessbases import _Excessclass
from .._lnumpy import F64Array
from .._lnumpy import inf64
from .._types import Coeffs
from .._types import Excess
from .._types import Index
from .._types import N_Vars
from .._types import TolsLine
from .._types import X
from .._types import XSingle
from .._types import YLine
# ======================================================================
_0_Index = Index(0)
_1_Index = Index(1)
# ======================================================================
class _MaxAbs[N_VarsTV: N_Vars](_Excessclass):
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
