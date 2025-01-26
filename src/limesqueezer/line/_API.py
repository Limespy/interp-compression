from .. import _lnumba as nb
from .. import _root
from .._base import _StreamBase
from .._errorfunctions import _MaxAbs_Line
from .._lnumpy import f64
from .._lnumpy import F64Array
from .._types import Excess
from .._types import fIndex
from .._types import Index
from .._types import N_Vars
from .._types import YLine
# ======================================================================
_0_Index = Index(0)
_1_Index = Index(1)
# ======================================================================
@nb.jitclass
@nb.clean
class Sequential_64(_StreamBase[N_Vars],
           _MaxAbs_Line[N_Vars]):
    _y: YLine
    # ------------------------------------------------------------------
    def __init__(self,
                 rtol: F64Array[N_Vars],
                 atol: F64Array[N_Vars],
                 preallocate: int = 322 # From Lucas sequence
                 ) -> None:

        self._StreamBase__init(rtol.shape, preallocate) # type: ignore[attr-defined]
        self._MaxAbs_Line__init(rtol, atol) # type: ignore[attr-defined]
    # ------------------------------------------------------------------
    def _estimate_secant(self,
                       index1: fIndex,
                       index2: fIndex,
                       excess1: Excess,
                       excess2: Excess) -> fIndex:
        return _root.linear(index1, excess1, index2, excess2)
    # ------------------------------------------------------------------
    def _estimate_base(self,
                       index1: fIndex,
                       index2: fIndex,
                       excess1: Excess,
                       excess2: Excess,
                       index_low: fIndex,
                       index_high: fIndex,
                       excess_low: Excess,
                       excess_high: Excess) -> fIndex:
        return _root.linear(index1, excess1, index2, excess2)
    # ------------------------------------------------------------------
    def _estimate_stable(self,
                        index1: fIndex,
                        index2: fIndex,
                        excess1: Excess,
                        excess2: Excess,
                        index_low: fIndex,
                        index_high: fIndex,
                        excess_low: Excess,
                        excess_high: Excess) -> fIndex:
        estimate = _root.linear(index_low, excess_low,
                               index_high, excess_high)
        return _root.shift_rf(index_low, index_high, estimate)
    # ------------------------------------------------------------------
    def _prepare_coeffs(self, ya):
        coeffs = self.coeffs
        for i in range(self._n_vars):
            coeffs[i, 1] = ya[i]
    # ------------------------------------------------------------------
    def _finish_coeffs(self, Dx, yb):
        coeffs = self.coeffs
        _Dx = f64(1.) / Dx
        for i in range(self._n_vars):
            coeffs[i, 0] = (yb[i] - coeffs[i, 1]) * _Dx
