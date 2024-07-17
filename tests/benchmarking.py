import limesqueezer as ls
import numpy as np
from limedev.test import BenchmarkResultsType
from limedev.test import eng_round
from limedev.test import run_timed
from limesqueezer import reference
# ======================================================================
X_DATA, Y_DATA1 = reference.raw_sine_x2(1e4)
Y_DATA2 = np.array((Y_DATA1, Y_DATA1[::-1])).T

ls.G['timed'] = True
# ======================================================================
def _run(use_numba: bool) -> float:
    ls.compress(X_DATA, Y_DATA2,
                tolerances = (1e-3, 1e-4, 1),
                use_numba = use_numba,
                errorfunction = 'MaxAbs')
    return ls.G['runtime']
# ======================================================================
def main() -> BenchmarkResultsType:
    results = {}
    for name, function, use_numba in (('no_numba', _run, False),
                                      ('with_numba', _run, True)):
        result, prefix = eng_round(run_timed(function, use_numba))
        results[f'{name} [{prefix}s]'] = result
    return ls.__version__, results
