from collections.abc import Callable
from collections.abc import Iterable
from functools import partial
from typing import Any
from typing import cast
from typing import Generic
from typing import Protocol
from typing import Self
from typing import TypeAlias
from typing import TypeVar

from ._lnumpy import F64Array
# ======================================================================
PASS_THROUGH = lambda _:_
# ======================================================================
# Typevar
T = TypeVar('T')
# ======================================================================
def copy_signature(source: T) -> Callable[..., T]:
    return PASS_THROUGH
# ======================================================================
def copy_type(source: T, target: Any) -> T:
    return target
# ======================================================================
def Cast(Type):
    return partial(cast, Type)
# ======================================================================
# Types

N_Coeffs = TypeVar('N_Coeffs', bound = int)
N_Diffs = TypeVar('N_Diffs', bound = int)
N_Points = TypeVar('N_Points', bound = int)
N_Samples = TypeVar('N_Samples', bound = int)
N_Vars = TypeVar('N_Vars', bound = int)

TolType = F64Array[N_Diffs, N_Vars]
PolyCoeffType = F64Array[N_Coeffs, N_Vars]
XSamplesType: TypeAlias = F64Array[N_Samples]
YSampleType: TypeAlias = F64Array[N_Samples, N_Diffs, N_Vars]
