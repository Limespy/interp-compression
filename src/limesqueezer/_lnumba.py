import warnings
from typing import TYPE_CHECKING

import numba as nb
from numba import core # type: ignore[attr-defined]
from numba import errors # type: ignore[attr-defined]
from numba.core.types.containers import ListType
from numba.experimental.jitclass.base import JitClassType
from numba.typed import List
# ======================================================================
types = nb.types
PASS_THROUGH = lambda x: x
# ======================================================================
if TYPE_CHECKING:
    from collections.abc import Sequence
    from collections.abc import Callable
    from typing import overload
    from typing import TypeAlias

    from numba.types import Type
    import numpy as np

    from ._lnumpy import f32
    from ._lnumpy import f64
    from ._lnumpy import i16
    from ._lnumpy import i32
    from ._lnumpy import i64
    from ._lnumpy import u16
    from ._lnumpy import u32
    from ._lnumpy import u64
    from ._lnumpy import up

    Signature: TypeAlias = core.typing.templates.Signature

    SpecType: TypeAlias = dict[str, Type] | Sequence[tuple[str, Type]]

else:
    Callable = tuple
    overload = PASS_THROUGH

    f32 = nb.f4
    f64 = nb.f8

    i16 = nb.i2
    i32 = nb.i4
    i64 = nb.i8

    u16 = nb.u2
    u32 = nb.u4
    u64 = nb.u8
    up = nb.uintp

    Type = Signature = SpecType = object
# ======================================================================
void = types.void
none = types.none

_jitclass = nb.experimental.jitclass # type: ignore[attr-defined]

size_t = nb.size_t # type: ignore[attr-defined]
# ======================================================================
warnings.filterwarnings('ignore',
                        '.*unsafe cast from uint64 to int64. Precision may be lost.',
                        errors.NumbaTypeSafetyWarning)
warnings.filterwarnings('ignore',
                        '.*irst-class function type feature is experimental',
                        errors.NumbaExperimentalFeatureWarning)
# ======================================================================
IS_NUMBA = True
IS_CACHE = True
IS_FASTMATH = True
IS_PARALLEL = False
# ======================================================================
if TYPE_CHECKING or IS_NUMBA:
    njit = nb.njit # type: ignore[attr-defined]
else:
    def njit(signature_or_function = None,
             locals = None,
             cache = False,
             pipeline_class = None,
             boundscheck = None,
             **_):
        return (signature_or_function
                if callable(signature_or_function)
                else PASS_THROUGH)
# ======================================================================
njitC = njit(cache = IS_CACHE)
njitF = njit(fastmath = IS_FASTMATH)
njitP = njit(parallel = IS_PARALLEL)
njitCF = njit(cache = IS_CACHE, fastmath = IS_FASTMATH)
# ======================================================================
def A(dim: int = 1, dtype = f64, ro = False) -> Type:
    return types.Array(dtype, dim, 'C', ro)
# ----------------------------------------------------------------------
def ARO(dim: int = 1, dtype = f64) -> Type:
    return types.Array(dtype, dim, 'C', readonly = True)
# ----------------------------------------------------------------------
_JITCLASS_UNSUPPORTED = {'__init_subclass__',
                         '__class_getitem__',
                         '__type_params__',
                         '__orig_bases__',
                         '__parameters__'}
def clean[T](cls: type[T]) -> type[T]:
    _dict: dict[str, object] = {}
    for base in reversed(cls.__mro__[:-1]):
        for k, v in base.__dict__.items():
            if k not in _JITCLASS_UNSUPPORTED:
                if k == '__annotations__':
                    if (a := _dict.get(k)) is None:
                        _dict[k] = v
                    else:
                        a |= v
                else:
                    _dict[k] = v
    return type(cls.__name__, (object, ), _dict)

@overload
def jitclass[T](cls_or_spec: type[T] = ..., spec: SpecType | None = ..., /
                     ) -> type[T]:
    ...
@overload
def jitclass[T](cls_or_spec: SpecType | None = ..., spec: None = ..., /
                     ) -> Callable[[type[T]], type[T]]:
    ...
if TYPE_CHECKING or IS_NUMBA:
    def jitclass(cls_or_spec = None, spec = None, /):
        return _jitclass(cls_or_spec, spec)
else:
    def jitclass(cls_or_spec = None, spec = None, /):
        return cls_or_spec if isinstance(cls_or_spec, type) else PASS_THROUGH
