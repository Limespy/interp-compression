from importlib import import_module
from sys import modules as _modules
from typing import TYPE_CHECKING

from ._API import *
# ======================================================================
# Hinting types
if TYPE_CHECKING:
    from types import ModuleType

    from . import decompress
    from . import poly
else:
    ModuleType = object
# ======================================================================
__version__ = '2.0.0'
# ======================================================================
def __getattr__(name: str) -> ModuleType:
    if name not in {'decompress', 'poly'}:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module = import_module(f'.{name}', __package__)
    setattr(_modules[__package__], name, module)
    return module
