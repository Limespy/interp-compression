from typing import Literal as L
from typing import TypeAlias

from .. import _lnumba as nb
from .. import _root
from .._base import _StreamBase
from .._errorfunctions import _MaxAbs_Taylor
from .._lnumpy import F64Array
from .._types import Excess
from .._types import fIndex
from .._types import Index
from .._types import N_Diffs
from .._types import N_Vars
from .._types import YDiff
from .._types import YDiff0
from .._types import YDiff1
from .._types import YDiff2
from .._types import YDiff3
from ..poly import make
# ======================================================================
_1_Index = Index(1)
# ======================================================================
class _Sequential[N_DiffsTV: N_Diffs, N_VarsTV: N_Vars
            ](_StreamBase[N_DiffsTV, N_VarsTV],
              _MaxAbs_Taylor[N_DiffsTV, N_VarsTV]):

    # Value constraints
    #
    # N_Points = N_Compressed + N_Uncompressed
    # BufferSize >= N_Points
    # N_Coeffs = 2 * N_Diffs
    _y: YDiff
    error_order: fIndex
    # ------------------------------------------------------------------
    def __init__(self,
                 rtol: F64Array[N_DiffsTV, N_VarsTV],
                 atol: F64Array[N_DiffsTV, N_VarsTV],
                 preallocate: int = 322 # From Lucas sequence
                 ) -> None:
        self._StreamBase__init(rtol.shape, preallocate) # type: ignore[attr-defined]
        self._MaxAbs_Taylor__init(rtol, atol) # type: ignore[attr-defined]
        self.error_order: fIndex = fIndex(self._n_coeffs - _1_Index)
    # ------------------------------------------------------------------
    def _estimate_secant(self,
                       index1: fIndex,
                       index2: fIndex,
                       excess1: Excess,
                       excess2: Excess) -> fIndex:
        return _root.poly(index1, excess1, index2, excess2, self.error_order)
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
        return _root.poly(index1, excess1, index2, excess2, self.error_order)
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
        estimate = _root.poly(index_low, excess_low,
                               index_high, excess_high,
                               self.error_order)
        return _root.shift_rf(index_low, index_high, estimate)
# ======================================================================
@nb.jitclass
@nb.clean
class Sequential0_64(_Sequential[L[1], N_Vars]):
    _y: YDiff0
    # ------------------------------------------------------------------
    def _prepare_coeffs(self, ya):
        make.prepare1_64(ya, self.coeffs)
    # ------------------------------------------------------------------
    def _finish_coeffs(self, Dx, yb):
        make.finish1_64(Dx, yb, self.coeffs)
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
# ======================================================================
@nb.jitclass
@nb.clean
class Sequential1_64(_Sequential[L[2], N_Vars]):
    """_summary_

    Parameters
    ----------
    rtol : F64Array[]
        relative tolerances
    """
    _y: YDiff1
    # ------------------------------------------------------------------
    def _prepare_coeffs(self, ya):
        make.prepare3_64(ya, self.coeffs)
    # ------------------------------------------------------------------
    def _finish_coeffs(self, Dx, yb):
        make.finish3_64(Dx, yb, self.coeffs)
# ======================================================================
@nb.jitclass
@nb.clean
class Sequential2_64(_Sequential[L[3], N_Vars]):
    _y: YDiff2
    # ------------------------------------------------------------------
    def _prepare_coeffs(self, ya):
        make.prepare5_64(ya, self.coeffs)
    # ------------------------------------------------------------------
    def _finish_coeffs(self, Dx, yb):
        make.finish5_64(Dx, yb, self.coeffs)
# ======================================================================
@nb.jitclass
@nb.clean
class Sequential3_64(_Sequential[L[4], N_Vars]):
    _y: YDiff3
    # ------------------------------------------------------------------
    def _prepare_coeffs(self, ya):
        make.prepare7_64(ya, self.coeffs)
    # ------------------------------------------------------------------
    def _finish_coeffs(self, Dx, yb):
        make.finish7_64(Dx, yb, self.coeffs)
# ======================================================================
Sequential: TypeAlias = (Sequential0_64 |
                         Sequential1_64 |
                         Sequential2_64 |
                         Sequential3_64)

sequential_compressors_64 = (Sequential0_64,
                             Sequential1_64,
                             Sequential2_64,
                             Sequential3_64)
