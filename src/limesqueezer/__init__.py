from importlib import import_module
from sys import modules as _modules
from typing import TYPE_CHECKING

from ._API import *
# ======================================================================
# Hinting types
if TYPE_CHECKING:
    from types import ModuleType

    from . import line
    from . import taylor
else:
    ModuleType = object
# ======================================================================
__version__ = '2.0.0'
# ======================================================================
def __getattr__(name: str) -> ModuleType | str:
    if name in {'line', 'taylor'}:
        module = import_module(f'.{name}', __package__)
        setattr(_modules[__package__], name, module)
        return module
    elif name == '__version__':
        from importlib import metadata
        return metadata.version(__package__)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
