"""Command line interface for processing command line input."""
from limedev.CLI import get_main

from . import _API as ls
from . import reference as ref
from .auxiliaries import Float64Array
from .auxiliaries import G
# ======================================================================
X_DATA, Y_DATA = ref.raw_sine_x2_normal(1e4, std=0.00001)
# ======================================================================
def block(use_numba: bool = False,
          is_timed: bool = False,
          is_debug: bool = False) -> int:
    """Demonstrates block compression."""
    G['timed'] = is_timed
    G['debug'] = is_debug

    xc, yc = ls.compress(X_DATA, Y_DATA,
                         tolerances = (1e-2, 1e-3, 1),
                         use_numba = use_numba,
                         errorfunction = 'MaxAbs')
    print(ls.stats(X_DATA, xc))

    if is_timed:
        print(f'runtime {G["runtime"]*1e3:.1f} ms')

    return 0
# ======================================================================
def _stream(X_DATA: Float64Array,
            Y_DATA: Float64Array,
            tol: tuple[float, float, float],
            use_numba: int):

    with ls.Stream(X_DATA[0], Y_DATA[0],
                   tolerances = tol,
                   use_numba = use_numba) as record:
        for x, y in zip(X_DATA[1:], Y_DATA[1:]):
            record(x, y)
    return record.x, record.y
# ----------------------------------------------------------------------
def stream(use_numba: bool = False,
           is_timed: bool = False,
           is_debug: bool = False) -> int:
    """Demonstrates stream compression."""
    G['timed'] = is_timed
    G['debug'] = is_debug

    xc, yc = _stream(X_DATA, Y_DATA, (1e-2, 1e-3, 1), use_numba)

    if is_timed:
        print(f'runtime {G["runtime"]*1e3:.1f} ms')

    return 0
# ======================================================================
def both(use_numba: bool = False,
         is_timed: bool = False,
         is_debug: bool = False) -> int:
    """Demonstrates both block and stream compression."""
    G['timed'] = is_timed
    G['debug'] = is_debug

    xcb, ycb = ls.compress(X_DATA, Y_DATA,
                           tolerances = (1e-2, 1e-3, 1),
                           use_numba = use_numba, initial_step = 100, errorfunction = 'MaxAbs')
    xcs, ycs = _stream(X_DATA, Y_DATA, (1e-2, 1e-3, 1.), use_numba)
    for i, (xb, xs) in enumerate(zip(xcb,xcs)):
        if xb != xs:
            print(f'Deviation at {i=}, {xb=}, {xs=}')
            break
    for i, (xb, xs) in enumerate(zip(reversed(xcb),reversed(xcs))):
        if xb != xs:
            print(f'Deviation at {i=}, {xb=}, {xs=}')
            break
    print(xcb)
    print(xcs)

    if is_timed:
        print(f'runtime {G["runtime"]*1e3:.1f} ms')
    return 0
# ======================================================================
def run(args: list[str], use_numba: int, is_plot: bool, is_timed: bool):

    if is_timed:
        print(f'runtime {G["runtime"]*1e3:.1f} ms')
# ======================================================================
main = get_main(__name__)
