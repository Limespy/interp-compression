import sys
from time import perf_counter

import numba as nb
import numpy as np
from limedev.test import BenchmarkResultsType
from limedev.test import eng_round
from limedev.test import run_timed
# ======================================================================
# Setting process to realtime
if sys.platform == 'win32':
    # Based on:
    #   "Recipe 496767: Set Process Priority In Windows" on ActiveState
    #   http://code.activestate.com/recipes/496767/
    import win32api, win32process, win32con

    pid = win32api.GetCurrentProcessId()
    handle = win32api.OpenProcess(win32con.PROCESS_ALL_ACCESS, True, pid)
    win32process.SetPriorityClass(handle,
                                    win32process.REALTIME_PRIORITY_CLASS)
else:
    import os

    os.nice(1)
# ======================================================================
N_VARS = 100
N_POINTS = 1000
INTERVAL = 0.2
X_RAW = np.arange(0., N_POINTS * INTERVAL, INTERVAL, np.float64)
_x = X_RAW.reshape(-1, 1) + np.linspace(0., 1., N_VARS)
_sin = np.sin(_x)
_cos = np.cos(_x)

Y_RAW = np.stack((_sin, _cos, -_sin, -_cos), axis = 1)

RTOL = np.full(Y_RAW.shape[1:], 0.01, dtype = np.float64)
ATOL = np.full(Y_RAW.shape[1:], 0.01, dtype = np.float64)
# ======================================================================
def _nb_compare(function):
    return ((function, 'no'), (nb.njit(function), 'with'))
# ======================================================================
# def error_excess() -> dict[str, dict[str, float]]:
#     results = {}
#     for function in (_errorfunctions.MaxAbs, _errorfunctions.MaxAbs_sequential):
#         comparison = {}
#         for naming, function in _nb_compare(function):
#             result, prefix = eng_round(run_timed(function)())
#             comparison[f'{naming} numba [{prefix}s]'] = result
#         results[function.__name__] = comparison
#     return results
# ======================================================================
def _import():
    results: dict[str, float] = {}

    # First
    t0 = perf_counter()
    from limesqueezer.stream.diff import Diff3
    result, prefix = eng_round(perf_counter() - t0)
    results[f'First [{prefix}s]'] = result

    # Min
    def do_import():
        from limesqueezer.stream.diff import Diff3

    result, prefix = eng_round(run_timed(do_import,
                                         n_samples = 250, t_min_s = 0.002
                                         )())
    results[f'Min [{prefix}s]'] = result
    return results
# ======================================================================
def _diff():
    from limesqueezer.poly import diff
    results = {}
    coeffs = np.full((Y_RAW.shape[1], 2 * Y_RAW.shape[0]), 1e-50, np.float64)
    result, prefix = eng_round(run_timed(diff.in_place_vars_coeffs,
                                         n_samples = 250, t_min_s = 0.002
                                         )(coeffs.copy(), 7))
    results[f'vars, coeffs [{prefix}s]'] = result

    coeffs = np.full((2 * Y_RAW.shape[0], Y_RAW.shape[1]), 1e-50, np.float64)
    result, prefix = eng_round(run_timed(diff.in_place_coeffs_vars,
                                         n_samples = 250, t_min_s = 0.002
                                         )(coeffs.copy(), 7))
    results[f'coeffs, vars [{prefix}s]'] = result
    return results
# ======================================================================
def _make():
    from limesqueezer.poly import make
    results = {}
    ia = 1
    ib = 10
    xa = X_RAW[ia]
    xb = X_RAW[ib]
    ya = Y_RAW[ia]
    yb = Y_RAW[ib]
    Dx = xa - xb

    coeffs = np.zeros((2 * yb.shape[0], yb.shape[1]))
    result, prefix = eng_round(run_timed(make.omake7, n_samples = 250, t_min_s = 0.002
                                         )(Dx, ya, yb, coeffs))
    results[f'old [{prefix}s]'] = result
    coeffs = np.zeros((yb.shape[1], 2 * yb.shape[0]))
    result, prefix = eng_round(run_timed(make.make7, n_samples = 250, t_min_s = 0.002
                                         )(Dx, ya, yb, coeffs))
    results[f'new [{prefix}s]'] = result
    return results
# ======================================================================
def _diffstream():
    from limesqueezer.stream.diff import Diff3
    from limesqueezer.stream.diff import Diff3Direct
    results = {}

    # ------------------------------------------------------------------
    def compress(stream: Diff3):
        stream.open(X_RAW[0], Y_RAW[0])

        for index in range(1, len(X_RAW)):
            stream.append(X_RAW[index], Y_RAW[index])

        stream.close()


    for Stream in (Diff3, Diff3Direct):
        subresults: dict[str, dict[str, float]] = {'Initialisation': {},
                      'Factory': {},
                      'Compression': {}}

        @nb.njit
        def factory(rtol,
                    atol,
                    preallocate: int = 322 # From Lucas sequence
                            ) -> Diff3:
            return Stream(rtol, atol, preallocate)

        # Initialisation
        ## First
        t0 = perf_counter()
        Stream(RTOL, ATOL)
        result, prefix = eng_round(perf_counter() - t0)
        subresults['Initialisation'][f'First [{prefix}s]'] = result

        ## Min
        result, prefix = eng_round(run_timed(Stream,
                                            t_min_s = 0.002, n_samples = 250
                                            )(RTOL, ATOL))
        subresults['Initialisation'][f'Min [{prefix}s]'] = result

        # Factory
        ## First
        t0 = perf_counter()
        factory(RTOL, ATOL)
        result, prefix = eng_round(perf_counter() - t0)
        subresults['Factory'][f'First [{prefix}s]'] = result

        ## Min
        result, prefix = eng_round(run_timed(factory,
                                            t_min_s = 0.002, n_samples = 250
                                            )(RTOL, ATOL))
        subresults['Factory'][f'Min [{prefix}s]'] = result

        # Comppression
        compress(Stream(RTOL, ATOL))
        result, prefix = eng_round(run_timed(compress,
                                            t_min_s = 0.01, n_samples = 50)(Stream(RTOL, ATOL)))
        subresults['Compression'][f'Min [{prefix}s]'] = result

        results[Stream.__name__.strip('_')] = subresults
    return results
# ======================================================================
def _compress():
    import time
    from limesqueezer.stream.diff import compress_diff3
    results: dict[str, dict[str, float]] = {'diff': {}}



    # First
    t0 = time.perf_counter()
    xc, yc = compress_diff3(X_RAW, Y_RAW, RTOL, ATOL)
    result, prefix = eng_round(time.perf_counter() - t0)
    results['diff'][f'First [{prefix}s]'] =  result

    # Min
    result, prefix = eng_round(run_timed(compress_diff3,
                                         t_min_s = 0.02, n_samples = 25
                                         )(X_RAW, Y_RAW, RTOL, ATOL))

    results['diff'][f'Min [{prefix}s]'] = result
    results['diff'][f'Ratio'] = N_POINTS / len(xc)
    return results
# ======================================================================
def main() -> BenchmarkResultsType:

    results = {}
    for subbenchmark in (_import,
                         _diffstream,
                         _compress,
                         _make,
                         _diff):
        print(subbenchmark.__name__)
        results[subbenchmark.__name__.strip('_')] = subbenchmark()
    from limesqueezer import __version__
    return __version__, results
