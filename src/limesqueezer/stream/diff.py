from typing import Literal as L

from .. import _lnumba as nb
from .._errorfunctions import _MaxAbs_Diff_Base
from .._lnumpy import f64
from .._lnumpy import F64Array
from .._typing import Excess
from .._typing import fIndex
from .._typing import Index
from .._typing import N_Diffs
from .._typing import N_Points
from .._typing import N_Vars
from .._typing import XSingle
from .._typing import YDiff
from .._typing import YDiff0
from .._typing import YDiff1
from .._typing import YDiff2
from .._typing import YDiff3
from .._typing import YDiffSingle
from ..poly import make
from ._base import _StreamBase
# ======================================================================
_0_Index = Index(0)
_1_Index = Index(1)
# ======================================================================
class _Diff[N_DiffsTV: N_Diffs, N_VarsTV: N_Vars
            ](_StreamBase[N_DiffsTV, N_VarsTV],
              _MaxAbs_Diff_Base[N_DiffsTV, N_VarsTV]):

    # Value constraints
    #
    # N_Points = N_Compressed + N_Uncompressed
    # BufferSize >= N_Points
    # N_Coeffs = 2 * N_Diffs
    _y: YDiff
    # ------------------------------------------------------------------
    def __init__(self,
                 rtol: F64Array[N_DiffsTV, N_VarsTV],
                 atol: F64Array[N_DiffsTV, N_VarsTV],
                 preallocate: int = 322 # From Lucas sequence
                 ) -> None:

        self._StreamBase__init(rtol.shape, preallocate) # type: ignore[attr-defined]
        self._MaxAbs_Diff_Base__init(rtol, atol) # type: ignore[attr-defined]
# ======================================================================
@nb.jitclass
@nb.clean
class Diff0_64(_Diff[L[1], N_Vars]):
    _y: YDiff0
    # ------------------------------------------------------------------
    def _prepare_coeffs(self, ya):
        make.prepare1(ya, self.coeffs)
    # ------------------------------------------------------------------
    def _finish_coeffs(self, Dx, yb):
        make.finish1(Dx, yb, self.coeffs)
# ======================================================================
@nb.jitclass
@nb.clean
class Diff1_64(_Diff[L[2], N_Vars]):
    _y: YDiff1
    # ------------------------------------------------------------------
    def _prepare_coeffs(self, ya):
        make.prepare3(ya, self.coeffs)
    # ------------------------------------------------------------------
    def _finish_coeffs(self, Dx, yb):
        make.finish3(Dx, yb, self.coeffs)
# ======================================================================
@nb.jitclass
@nb.clean
class Diff2_64(_Diff[L[3], N_Vars]):
    _y: YDiff2
    # ------------------------------------------------------------------
    def _prepare_coeffs(self, ya):
        make.prepare5(ya, self.coeffs)
    # ------------------------------------------------------------------
    def _finish_coeffs(self, Dx, yb):
        make.finish5(Dx, yb, self.coeffs)
# ======================================================================
@nb.jitclass
@nb.clean
class Diff3_64(_Diff[L[4], N_Vars]):
    _y: YDiff3
    # ------------------------------------------------------------------
    def _prepare_coeffs(self, ya):
        make.prepare7(ya, self.coeffs)
    # ------------------------------------------------------------------
    def _finish_coeffs(self, Dx, yb):
        make.finish7(Dx, yb, self.coeffs)
# ======================================================================
@nb.njitC
def compress_diff3_64[N_VarsTV: N_Vars,
                      N_Points_In: N_Points,
                      N_Points_Out: N_Points
                      ](x: F64Array[N_Points_In],
             y: F64Array[N_Points_In, L[4], N_VarsTV],
             rtol: F64Array[L[4], N_VarsTV],
             atol: F64Array[L[4], N_VarsTV]
             ) -> tuple[F64Array[N_Points_Out],
                        F64Array[N_Points_Out, L[4], N_VarsTV]]:
    len_x = len(x)
    stream = Diff3_64(rtol, atol, len_x // 2)
    stream.open(x[0], y[0], 5)
    for index in range(1, len_x):
        stream.append(x[index], y[index])
    stream.close()
    return stream.x, stream.y # type: ignore[return-value]

# ======================================================================
diffstreams_64 = (Diff0_64, Diff1_64, Diff2_64, Diff3_64)
# ======================================================================
@nb.jitclass
class Diff3Direct(_Diff[L[4], N_Vars]):
    # ------------------------------------------------------------------
    def _prepare_coeffs(self, ya: F64Array[N_Diffs, N_Vars]) -> None:
        coeffs = self.coeffs
        for i in range(ya.shape[1]):
            coeffs[i, 4] = ya[3, i]
            coeffs[i, 5] = ya[2, i]
            coeffs[i, 6] = ya[1, i]
            coeffs[i, 7] = ya[0, i]
    # ------------------------------------------------------------------
    def _finish_coeffs(self, Dx: XSingle, yb: F64Array[N_Diffs, N_Vars]
                       ) -> None:
        coeffs = self.coeffs
        _1_6 = 0.16666666666666666666666666666666666666666666667
        Dx2 = Dx * Dx
        Dx3 = Dx2 * Dx

        _Dx1 = 1./Dx
        _Dx2 = _Dx1 * _Dx1

        _Dx4 = _Dx2 * _Dx2
        _Dx5 = _Dx4 * _Dx1
        _Dx6 = _Dx5 * _Dx1
        _Dx7 = _Dx6 * _Dx1

        a00 = a01 = a02 = a03 = _Dx7
        a00 *= -20.
        a01 *= 10.
        a02 *= -2.
        a03 *= _1_6

        a10 = a11 = a12 = a13 = _Dx6
        a10 *= 70.
        a11 *= -34.
        a12 *= 6.5
        a13 *= -0.5

        a20 = a21 = a22 = a23 = _Dx5
        a20 *= -84.
        a21 *= 39.
        a22 *= -7.
        a23 *= 0.5

        a30 = a31 = a32 = a33 = _Dx4
        a30 *= 35.
        a31 *= -15.
        a32 *= 2.5
        a33 *= -_1_6

        for i in range(yb.shape[1]):
            ya0 = coeffs[i, 7]
            ya1Dx = coeffs[i, 6] * Dx
            ya2Dx2 = coeffs[i, 5] * Dx2
            ya3Dx3 = coeffs[i, 4] * Dx3

            Dy0 = yb[0, i] - ya0 - ya1Dx - 0.5 * ya2Dx2 - _1_6 * ya3Dx3
            Dy1 = yb[1, i] * Dx - ya1Dx - ya2Dx2 - 0.5 * ya3Dx3
            Dy2 = yb[2, i] * Dx2 - ya2Dx2 - ya3Dx3
            Dy3 = yb[3, i] * Dx3 - ya3Dx3

            coeffs[i, 0] = a00 * Dy0 + a01 * Dy1 + a02 * Dy2 + a03 * Dy3
            coeffs[i, 1] = a10 * Dy0 + a11 * Dy1 + a12 * Dy2 + a13 * Dy3
            coeffs[i, 2] = a20 * Dy0 + a21 * Dy1 + a22 * Dy2 + a23 * Dy3
            coeffs[i, 3] = a30 * Dy0 + a31 * Dy1 + a32 * Dy2 + a33 * Dy3
            coeffs[i, 4] *= _1_6
            coeffs[i, 5] *= 0.5
