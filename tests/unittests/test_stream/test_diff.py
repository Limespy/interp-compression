import numpy as np
import pytest
from limesqueezer.stream import diff
# ======================================================================
# Auxiliary
# pytestmark = pytest.mark.filterwarnings('ignore::numba.core.errors.NumbaExperimentalFeatureWarning')

parametrize = pytest.mark.parametrize
# ======================================================================
N_VARS = 2
N_POINTS = 1000
INTERVAL = 0.2
X_RAW = np.arange(0., N_POINTS * INTERVAL, INTERVAL, np.float64)
_x = X_RAW.reshape(-1, 1) + np.linspace(0., 2. * np.pi, N_VARS)
_sin = np.sin(_x)
_cos = np.cos(_x)

Y_RAW = np.stack((_sin, _cos, -_sin, -_cos), axis = 1)

RTOL = np.full(Y_RAW.shape[1:], 0.01, dtype = np.float64)
ATOL = np.full(Y_RAW.shape[1:], 0.01, dtype = np.float64)
# ======================================================================
@parametrize(('n_diffs', 'DiffStream'),
             (tuple(enumerate(diff.diffstreams, start = 1))))
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
    assert diffstream.yc.shape[1:] == (n_diffs, N_VARS)
    assert 2 <= len(diffstream) == len(diffstream.xc) == len(diffstream.yc)< N_POINTS/2
    assert len(diffstream.xb) == len(diffstream.yb) == 0
