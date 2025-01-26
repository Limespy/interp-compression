from multiprocessing import Process
from os import getenv

import numpy as np
import pytest
from limesqueezer.taylor import sequential
# ======================================================================
# Auxiliary
# pytestmark = pytest.mark.filterwarnings('ignore::numba.core.errors.NumbaExperimentalFeatureWarning')

parametrize = pytest.mark.parametrize
# ======================================================================
N_VARS = 2
N_POINTS = 100
INTERVAL = 0.2
X_RAW = np.arange(0., N_POINTS * INTERVAL, INTERVAL, np.float64)
_x = X_RAW.reshape(-1, 1) + np.linspace(0., 2. * np.pi, N_VARS)
_sin = np.sin(_x)
_cos = np.cos(_x)

Y_RAW = np.stack((_sin, _cos, -_sin, -_cos), axis = 1)

RTOL = np.full(Y_RAW.shape[1:], 0.01, dtype = np.float64)
ATOL = np.full(Y_RAW.shape[1:], 0.01, dtype = np.float64)
# ======================================================================
@pytest.mark.filterwarnings(
    *(('ignore:overflow encountered in scalar add:RuntimeWarning',
       'ignore:overflow encountered in scalar subtract:RuntimeWarning')
       if getenv('NUMBA_DISABLE_JIT', 0) == 1 else ('',)))
@parametrize(('n_diffs', 'DiffStream'),
             (tuple(enumerate(sequential.sequential_compressors_64,
                              start = 1))))
@pytest.mark.SLOW
@pytest.mark.SLOW1
def test_diffstream(n_diffs, DiffStream):
    diffstream = DiffStream(RTOL[:n_diffs], ATOL[:n_diffs])
    diffstream.open(X_RAW[0], Y_RAW[0, :n_diffs])
    try:
        for index in range(1, len(X_RAW)):
            diffstream.append(X_RAW[index], Y_RAW[index,:n_diffs])
        diffstream.close()
    except:
        print(index)
        raise
    print(diffstream._index_b0, diffstream._x[diffstream._index_b0])
    assert diffstream.xc[0] == X_RAW[0]
    assert diffstream.xc[-1] == X_RAW[-1]
    assert np.all(diffstream.yc[0] == Y_RAW[0, :n_diffs])
    assert np.all(diffstream.yc[-1] == Y_RAW[-1, :n_diffs])
    assert diffstream.yc.shape[1:] == (n_diffs, N_VARS)
    assert 2 <= len(diffstream) == len(diffstream.xc) == len(diffstream.yc)< N_POINTS/2
    assert len(diffstream.xb) == len(diffstream.yb) == 0
# ======================================================================
# class Test_batch:
#     batch_compressors = ()

#     @parametrize
#     def test_equal_to_individual(self, batch, )
# ======================================================================
def compress(x, y):
    diffstream = sequential.Sequential3_64(RTOL, ATOL)
    diffstream.open(x[0], y[0])
    for index in range(1, len(x)):
        diffstream.append(x[index], y[index])
    diffstream.close()
# ----------------------------------------------------------------------
def test_in_multiprocessing():
    process = Process(target = compress,
                      args = (X_RAW, Y_RAW))
    process.start()
    process.join()
