import sys
from typing import TYPE_CHECKING
from typing import TypeAlias

import matplotlib.pyplot as plt
import numpy as np

from . import _lnumba as nb
# ======================================================================
# Hinting types
if TYPE_CHECKING:
    from collections.abc import Callable
    from collections.abc import Sequence
    from typing import Any
    from typing import TypeVar

    from ._typing import F64Array
    from ._typing import I64Array

    RTol: TypeAlias = F64Array
    ATol: TypeAlias = F64Array

    MaybeArray: TypeAlias = float | F64Array
    TolInput: TypeAlias = float | Sequence[float]
    TolInternal: TypeAlias = F64Array
    SqrtRange = Callable[[int], I64Array]
    Function = TypeVar('Function', Callable)
else:
    Any = F64Array = I64Array = RTol = ATol = Function = object
# ======================================================================
# Global parameters
G: dict[str, Any] = {'timed': False,
                     'debug': False,
                     'profiling': False,
                     'runtime': 0}
PASS_THROUGH = lambda x:x
# ======================================================================

# AUXILIARIES
def maybejit(use_numba, function: Function, *args, **kwargs
             ) -> Function:
    return nb.njit(function, *args, **kwargs) if use_numba else function
# ======================================================================
def py_and_nb(function: Function, **kwargs) -> tuple[Function, Function]:
    return (function, nb.njit(function, **kwargs))
# ======================================================================
def to_ndarray(item: Any, shape: tuple[int, ...] = ()
               ) -> F64Array:
    if not hasattr(item, '__iter__'): # Not some iterable
        if -1 in shape: # Array of shape length of dimensions with one item
            return np.array(item, ndmin = len(shape))
        else:
            return np.full(shape, item) # Array of copies in the shape
    elif not isinstance(item, np.ndarray): # Iterable into array
        item = np.array(item)
    return item if shape == () else item.reshape(shape)
# ======================================================================

def _sqrtrange(n: int) -> I64Array:
    """~ sqrt(n + 2) equally spaced integers including the n."""
    inds = np.arange(0, n + 1, round(np.sqrt(n + 1)), np.int64)
    inds[-1] = n
    return inds
#───────────────────────────────────────────────────────────────────────
sqrtranges = py_and_nb(_sqrtrange)
# ======================================================================
def wait(text: str = '') -> None:
    if input(text) in ('e', 'q', 'exit', 'quit'): sys.exit()
# ======================================================================
def stats(x_data, xc):
    # What if the data was compressed by sampling at the minimum interval of the compressed
    datarange = x_data[-1] - x_data[0]
    minslice = np.min(np.diff(xc))
    maxslice = np.max(np.diff(xc))
    # ameanslice = (minslice + maxslice) / 2
    # gmeanslice = (minslice * maxslice)**0.5
    hmeanslice = 2 / (1 / minslice + 1 / maxslice)

    return f'''{len(x_data) / len(xc):.0f} compression ratio
{datarange / hmeanslice / len(xc):.1f} x better than mean slices
{datarange / minslice / len(xc):.1f} x better than minimum slices'''
# ======================================================================
# DEBUG
def debugsetup(x: F64Array, y: F64Array, tol: float, fitset, start
               ) -> dict[str, Any]:
    _G = {'x': x,
          'y': y,
          'tol': tol,
          'interp': fitset._interpolate,
          'start': start}

    _G['fig'], axs = plt.subplots(3,1)
    for ax in axs:
        ax.grid()
    _G['ax_data'], _G['ax_res'], _G['ax_root']  = axs

    _G['ax_data'].fill_between(x, (y - tol).flatten(), (y + tol).flatten(),
                               alpha=.3, color = 'blue')

    _G['line_fit'], = _G['ax_data'].plot(0, 0, '.', color = 'orange',
                                                label = 'fit')
    _G['ax_res'].axhline(color = 'red', linestyle = '--')
    _G['ax_root'].set_ylabel('Tolerance left')
    _G['ax_root'].axhline(color = 'red', linestyle = '--')

    plt.ion()
    plt.show()
    wait('Initialised')
    return _G

def _reset_ax(key: str, ylabel: str) -> None:
    G[key].clear()
    G[key].grid()
    G[key].axhline(color = 'red', linestyle = '--')
    G[key].set_ylabel(ylabel)

def _set_xy(key, x, y):
    G[key].set_xdata(x)
    G[key].set_ydata(y)
