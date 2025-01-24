from importlib import import_module
from sys import modules as _modules
from typing import TYPE_CHECKING

from ._API import *
# ======================================================================
# Hinting types
if TYPE_CHECKING:
    from types import ModuleType

    from . import diff
    from . import interpolate
    from . import make

else:
    ModuleType = object
# ======================================================================
def __getattr__(name: str) -> ModuleType:
    if name not in ('diff', 'interpolate', 'make'):
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module = import_module(f'.{name}', __package__)
    setattr(_modules[__package__], name, module)
    return module
