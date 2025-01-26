from typing import TYPE_CHECKING
# ======================================================================
# Hinting types
if TYPE_CHECKING:
    from ._types_checking import *
else:
    from ._types_runtime import *
# ======================================================================
PASS_THROUGH = lambda _:_
# ======================================================================
def copy_signature(source: T) -> Callable[..., T]:
    return PASS_THROUGH
# ======================================================================
def copy_type(source: T, target: Any) -> T:
    return target
