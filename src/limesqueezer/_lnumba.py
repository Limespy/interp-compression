import warnings
from typing import TYPE_CHECKING

from numba import *
from numba import core # type: ignore[attr-defined]
from numba import errors
from numba import types # type: ignore[attr-defined]
from numba.core.types.containers import ListType
from numba.typed import List as _List
# ======================================================================
jitclass = experimental.jitclass # type: ignore[attr-defined]
njit = njit # type: ignore[attr-defined]
void = types.void
warnings.filterwarnings('ignore',
                        '.*unsafe cast from uint64 to int64. Precision may be lost.',
                        errors.NumbaTypeSafetyWarning)
# ======================================================================
if TYPE_CHECKING:
    from typing import TypeAlias
    import numpy as np

    f32: TypeAlias = np.float32
    f64: TypeAlias = np.float64

    i16: TypeAlias = np.int16
    i32: TypeAlias = np.int32
    i64: TypeAlias = np.int64

    u16: TypeAlias = np.uint16
    u32: TypeAlias = np.uint32
    u64: TypeAlias = np.uint64

    Type: TypeAlias = core.types.Type
    Signature: TypeAlias = core.typing.templates.Signature

    class _ListTypeMeta(_List.__class__):
        def __getitem__(self, _type):
            return ListType(_type)

    class List(_List, metaclass = _ListTypeMeta):
        ...

else:
    f32 = f4
    f64 = f8

    i16 = i2
    i32 = i4
    i64 = i8

    u16 = u2
    u32 = u4
    u64 = u8

    Type = Signature = object

    class _ListTypeImpl:
        def __call__(self, *args, **kwargs):
            return _List(*args, **kwargs)
        # --------------------------------------------------------------
        def __getitem__(self, _type):
            return ListType(_type)

    List = _ListTypeImpl()
# ======================================================================
IS_CACHE = False
IS_FASTMATH = False
IS_PARALLEL = False

njitC = njit(cache = IS_CACHE)
njitF = njit(fastmath = IS_FASTMATH)
njitP = njit(parallel = IS_PARALLEL)
njitCF = njit(cache = IS_CACHE, fastmath = IS_FASTMATH)
# ======================================================================
def A(dim: int = 1, dtype = f64) -> Type:
    return types.Array(dtype, dim, 'C')
# ----------------------------------------------------------------------
def ARO(dim: int = 1, dtype = f64) -> Type:
    return types.Array(dtype, dim, 'C', readonly = True)
