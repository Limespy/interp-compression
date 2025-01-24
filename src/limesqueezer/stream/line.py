from .. import _lnumba as nb
from .._errorfunctions import _MaxAbs_Line_Base
from .._lnumpy import f64
from .._lnumpy import F64Array
from .._typing import Excess
from .._typing import fIndex
from .._typing import Index
from .._typing import N_Vars
from .._typing import YLine
from ._base import _StreamBase
# ======================================================================
_0_Index = Index(0)
_1_Index = Index(1)
# ======================================================================
@nb.jitclass
@nb.clean
class Line_64(_StreamBase[N_Vars],
           _MaxAbs_Line_Base[N_Vars]):
    _y: YLine
    # ------------------------------------------------------------------
    def __init__(self,
                 rtol: F64Array[N_Vars],
                 atol: F64Array[N_Vars],
                 preallocate: int = 322 # From Lucas sequence
                 ) -> None:

        self._StreamBase__init(rtol.shape, preallocate) # type: ignore[attr-defined]
        self._MaxAbs_Line_Base__init(rtol, atol) # type: ignore[attr-defined]
    # ------------------------------------------------------------------
    def _prepare_coeffs(self, ya):
        coeffs = self.coeffs
        for i in range(self._n_vars):
            coeffs[i, 1] = ya[i]
    # ------------------------------------------------------------------
    def _finish_coeffs(self, Dx, yb):
        coeffs = self.coeffs
        _Dx = 1. / Dx
        for i in range(self._n_vars):
            coeffs[i, 0] = (yb[i] - coeffs[i, 1]) * _Dx
