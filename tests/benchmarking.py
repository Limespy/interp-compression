import limesqueezer as ls
import numba as nb
import numpy as np
from limedev.test import BenchmarkResultsType
from limedev.test import eng_round
from limedev.test import run_timed
from limesqueezer import _errorfunctions
from limesqueezer import reference
from limesqueezer._aux import py_and_nb
# ======================================================================
X_DATA, Y_DATA1 = reference.raw_sine_x2(1e4)
Y_DATA2 = np.array((Y_DATA1, Y_DATA1[::-1])).T

ls.G['timed'] = False
# ======================================================================
def _run(use_numba: bool) -> None:
    ls.compress(X_DATA, Y_DATA2,
                tolerances = (1e-3, 1e-4, 1),
                use_numba = use_numba,
                errorfunction = 'MaxAbs')
# ======================================================================
def _nb_compare(function):
    return ((function, 'no'), (nb.njit(function), 'with'))
# ======================================================================
def error_excess() -> dict[str, dict[str, float]]:
    results = {}
    for function in (_errorfunctions.MaxAbs, _errorfunctions.MaxAbs_sequential):
        comparison = {}
        for naming, function in _nb_compare(function):
            result, prefix = eng_round(run_timed(function))
            comparison[f'{naming} numba [{prefix}s]'] = result
        results[function.__name__] = comparison
    return results
# ======================================================================
def main() -> BenchmarkResultsType:
    results = {}
    # for name, function, use_numba in (('no_numba', _run, False),
    #                                   ('with_numba', _run, True)):
    #     result, prefix = eng_round(run_timed(function, use_numba))
    #     results[f'{name} [{prefix}s]'] = result
    results['Error excess'] = error_excess()
    return ls.__version__, results
