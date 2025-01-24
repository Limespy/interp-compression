from typing import TYPE_CHECKING
# ======================================================================
# Hinting types
if TYPE_CHECKING:
    from ._typing_type_checking import *
else:
    from ._typing_runtime import *
# ======================================================================
PASS_THROUGH = lambda _:_
# ======================================================================
def copy_signature(source: T) -> Callable[..., T]:
    return PASS_THROUGH
# ======================================================================
def copy_type(source: T, target: Any) -> T:
    return target
# ======================================================================
# def _cast(Type):
#     return partial(cast, Type)
# ======================================================================
# class CastDec(Generic[T]):
#     def __init__(cls, _type: type[T]):
#         ...
#     def __call__(self, function: Any) -> T:
#         return function
