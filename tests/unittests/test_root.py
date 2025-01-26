import numba as nb
import numpy as np
import pytest
from limesqueezer import _root
# ======================================================================
parametrize = pytest.mark.parametrize
# ======================================================================
class Interp:
    x0: float = 0.
    y0: float = -1.
    x1: float = 1.
    y1: float = 1.
# ======================================================================
class Test_linear(Interp):
    # ------------------------------------------------------------------
    @staticmethod
    @nb.njit
    def f(array, x0, y0, x1, y1):
        x = _root.linear(x0, y0, x1, y1)
        print(type(x))
        array[0] = x
    # ------------------------------------------------------------------
    @parametrize(('dtype', ), ((np.float32,), (np.float64,)))
    def test_dtype(self, dtype: type[np.float32] | type[np.float64]):
        array = np.zeros((1,), dtype)
        self.f(array, dtype(self.x0), dtype(self.y0),
                      dtype(self.x1), dtype(self.y1))
        # assert False
# ======================================================================
class Test_poly(Interp):
    # ------------------------------------------------------------------
    @staticmethod
    @nb.njit
    def f(array, x0, y0, x1, y1, n):
        x = _root.poly(x0, y0, x1, y1, n)
        print(type(x))
        array[0] = x
    # ------------------------------------------------------------------
    @parametrize(('dtype', ), ((np.float32,), (np.float64,)))
    def test_dtype(self, dtype: type[np.float32] | type[np.float64]):
        array = np.zeros((1,), dtype)
        self.f(array, dtype(self.x0), dtype(self.y0),
                       dtype(self.x1), dtype(self.y1),
                       dtype(3.))
        # assert False
