from collections.abc import Callable
from typing import Any
from typing import Literal as L
from typing import TypeAlias
from typing import TypeVar

from .._lnumpy import f64
from .._lnumpy import F64Array
from .._lnumpy import u64
from .._lnumpy import up
# ======================================================================
T = TypeVar('T')

N_CoeffsTV = TypeVar('N_CoeffsTV', bound = int)
N_DiffsTV = TypeVar('N_DiffsTV', bound = int)
N_PointsTV = TypeVar('N_PointsTV', bound = int)
N_SamplesTV = TypeVar('N_SamplesTV', bound = int)
N_VarsTV = TypeVar('N_VarsTV', bound = int)
# ----------------------------------------------------------------------
Length: TypeAlias = int
N_Compressed: TypeAlias = Length
N_Uncompressed: TypeAlias = Length
N_Points: TypeAlias = Length
N_Diffs: TypeAlias = L[1, 2, 3, 4]
N_Coeffs: TypeAlias = L[2, 4, 6, 8]
N_Vars: TypeAlias = Length
BufferSize: TypeAlias = Length
# ----------------------------------------------------------------------
X: TypeAlias = F64Array[BufferSize]
XSingle: TypeAlias = f64
# ----------------------------------------------------------------------
YLine: TypeAlias = F64Array[BufferSize, N_Vars]

YDiff: TypeAlias = F64Array[BufferSize, N_Diffs, N_Vars]
YDiff0: TypeAlias = F64Array[BufferSize, L[1], N_Vars]
YDiff1: TypeAlias = F64Array[BufferSize, L[2], N_Vars]
YDiff2: TypeAlias = F64Array[BufferSize, L[3], N_Vars]
YDiff3: TypeAlias = F64Array[BufferSize, L[4], N_Vars]

Y: TypeAlias = YDiff | YLine

YLineSingle: TypeAlias = F64Array[N_Vars]
YDiffSingle: TypeAlias = F64Array[N_Diffs, N_Vars]

YSingle: TypeAlias = YLineSingle | YDiffSingle
# ----------------------------------------------------------------------
Excess: TypeAlias = f64
Index: TypeAlias = up
fIndex: TypeAlias = f64

Coeffs: TypeAlias = F64Array[N_Coeffs, N_Vars]
TolsDiff: TypeAlias = F64Array[N_Diffs, N_Vars]
TolsLine: TypeAlias = F64Array[N_Vars]
