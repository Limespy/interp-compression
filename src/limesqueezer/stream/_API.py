from typing import Generic
from typing import TYPE_CHECKING
from typing import TypeVar

import numpy as np
from matplotlib import pyplot as plt

from .. import _lnumba as nb
from .._errorfunctions import MaxAbs_Sequential
from .._lnumpy import f64
from .._lnumpy import up
from ..poly import diff
from ..poly import make
from ..poly.interpolate import interpolate
# ======================================================================
# Hinting types
if TYPE_CHECKING:
    from collections.abc import Callable

    from .._typing import F64Array
    # from .._typing import N_Coeffs
    # from .._typing import N_Diffs
    # from .._typing import N_Points
    # from .._typing import N_Samples
    # from .._typing import N_Vars
    from .._typing import UPArray
else:
    Callable = tuple
    F64Array = UPArray = tuple
    All_TypeVars = object
# ======================================================================
N_Compressed = TypeVar('N_Compressed', bound = int)
N_Uncompressed = TypeVar('N_Uncompressed', bound = int)
N_Points = TypeVar('N_Points', bound = int)
N_Diffs = TypeVar('N_Diffs', bound = int)
N_Coeffs = TypeVar('N_Coeffs', bound = int)
N_Samples = TypeVar('N_Samples', bound = int)
N_Vars = TypeVar('N_Vars', bound = int)
BufferSize = TypeVar('BufferSize', bound = int)
# ======================================================================
LUCAS = (76, 123, 199, 322, 521, 843, 1364, 2207, 3571, 5778, 9349)
_MIN_CONVERGENCE = 0.1
# ======================================================================
@nb.njit
def _linear(x0: float, y0: float, x1: float, y1: float) -> float:
    return (x1 * y0 - x0 * y1) / (y0 - y1)
# ----------------------------------------------------------------------
@nb.njit
def _shifted_rf(x_rf: float, x_low: float, x_high: float) -> float:
    diff_to_low = x_rf - x_low
    diff_to_high = x_high - x_rf
    return (x_rf + diff_to_low * 0.5
            if diff_to_low < diff_to_high else
            x_rf - diff_to_high * 0.5)
# ----------------------------------------------------------------------
@nb.njit
def _estimate_index(index1: float,
                    index2: float,
                    excess1: float,
                    excess2: float,
                    index_low: float,
                    index_high: float,
                    excess_low: float,
                    excess_high: float):
    """Heuristics for selecting the next index to try.

    Constraining index: index_low < index_try < index_high
    """
    index_secant = _linear(index1, excess1, index2, excess2)
    index_try = index_secant

    if not index_low < index_try < index_high:
        # Secant not converging
        index_rf = _linear(index_low, excess_low, index_high, excess_high)
        index_try = _shifted_rf(index_rf, index_low, index_high)
        # index_bin = (index_high + index_low) / 2

    if abs(index_try - index2) < _MIN_CONVERGENCE * (index_high - index_low):
        # Slow convergence
        index_try = (index_high + index_low) / 2
    return index_try
# ======================================================================
# @nb.jitclass
class Stream(Generic[N_Compressed,
                     N_Uncompressed,
                     N_Points,
                     N_Samples,
                     N_Diffs,
                     N_Coeffs,
                     N_Vars,
                     BufferSize]):

    # Value constraints
    #
    # N_Points = N_Compressed + N_Uncompressed
    # BufferSize >= N_Points
    # N_Coeffs = 2 * N_Diffs

    def __init__(self,
                 rtol: F64Array[N_Diffs, N_Vars],
                 atol: F64Array[N_Diffs, N_Vars],
                 preallocate: int = 322 # From Lucas sequence
                 ) -> None:
        n_diffs, n_vars = rtol.shape
        self._n_diffs: N_Diffs = n_diffs
        self._n_vars: N_Vars = n_vars
        self._n_coeffs: N_Coeffs = up(2 * n_diffs)
        self._size: BufferSize = up(preallocate)
        self._size_previous = up(self._size * 0.618033988749)
        self.index0: up = up(0) # Index of x0 and y0
        self.index1: float = 1.
        self.index_buffer_start: up = up(1)
        self.index_buffer_end: up = up(1)

        self._until_next: up = up(5) # TODO some heuristic for this

        self.excess_obj: MaxAbs_Sequential = MaxAbs_Sequential(rtol, atol)

        self.prepare_poly: make.Prepare[N_Diffs, N_Diffs, N_Vars]
        self.finish_poly: make.Finish[N_Diffs, N_Diffs, N_Vars]
        self.prepare_poly, self.finish_poly = make.get_split_maker(n_diffs)
        self.coeffs: F64Array[N_Diffs, N_Coeffs, N_Vars] = (
            np.zeros((self._n_coeffs, self._n_vars)))

        self._x: F64Array[BufferSize] = np.zeros((self._size, ), f64)
        self._y: F64Array[BufferSize, N_Diffs, N_Vars] = (
            np.zeros((self._size, self._n_diffs, self._n_vars), f64))
    # ------------------------------------------------------------------
    @property
    def xc(self) -> F64Array[N_Compressed]:
        return self._x[:self.index0 + up(1)]
    # ------------------------------------------------------------------
    @property
    def yc(self) -> F64Array[N_Compressed, N_Diffs, N_Vars]:
        return self._y[:self.index0 + up(1)]
    # ------------------------------------------------------------------
    @property
    def xb(self) -> F64Array[N_Uncompressed]:
        return self._x[self.index_buffer_start:self.index_buffer_end]
    # ------------------------------------------------------------------
    @property
    def yb(self) -> F64Array[N_Uncompressed, N_Diffs, N_Vars]:
        return self._y[self.index_buffer_start:self.index_buffer_end]
    # ------------------------------------------------------------------
    @property
    def x(self) -> F64Array[N_Points]:
        return np.concatenate((self.xc, self.xb))
    # ------------------------------------------------------------------
    @property
    def y(self) -> F64Array[N_Points, N_Diffs, N_Vars]:
        return np.concatenate((self.yc, self.yb))
    # ------------------------------------------------------------------
    def __len__(self) -> int:
        return int(self.index0
                   + self.index_buffer_end
                   - self.index_buffer_start) + 1
    # ------------------------------------------------------------------
    def _unify(self, index_new_buffer_start: int) -> None:
        """Makes the x and y arrays contiguous.

        Moves the uncomporessed buffer section back next to the end of
        of the compressed section
        """
        print('_unify')

        shift = self.index_buffer_start - index_new_buffer_start

        index_new_buffer_end: up = self.index_buffer_end - shift

        self._x[index_new_buffer_start:index_new_buffer_end] = self.xb
        self._y[index_new_buffer_start:index_new_buffer_end] = self.yb

        self.index_buffer_end = index_new_buffer_end
        self.index_buffer_start = index_new_buffer_start
    # ------------------------------------------------------------------
    def _allocate(self):
        """Increases the size of x and y arrays.

        Allocation size based on Lucas numbers
        """
        print('_allocate')

        self._x = np.concatenate((self._x,
                                 np.zeros((self._size_previous, ), f64)))
        self._y = np.concatenate((self._y,
                                 np.zeros((self._size_previous,
                                           *self._y.shape[1:]), f64)))
        _allocated_current: up = self._size
        self._size += self._size_previous
        self._size_previous = _allocated_current
    # ------------------------------------------------------------------
    def _get_space(self):
        print('_get_space')
        index_new_buffer_start: up = self.index0 + up(1)

        if self.index_buffer_start == index_new_buffer_start:
            self._allocate()
        else:
            self._unify(index_new_buffer_start)
   # ------------------------------------------------------------------
    def _prepare(self):
        print('_prepare')

        # Setting up 0-point
        y0 = self._y[self.index0]

        # Setting up polynomials
        self.prepare_poly(y0, self.coeffs)

        # Checking y-values for the minimum value
        self.excess1: f64 = self.excess_obj.minimum(y0)
        self.index1: f64 = f64(self.index_buffer_start)
    # ------------------------------------------------------------------
    def prime(self, x: float, y: F64Array[N_Diffs, N_Vars]):
        """Sets first value and makes the compressor ready to receive
        values."""
        print('prime')
        self._x[0] = x
        self._y[0] = y
        self._prepare()
    # ------------------------------------------------------------------
    def _check(self, index: up) -> float:
        """Calculates excess, index is index of last item contained."""
        print('_check')
        print('\tindex_buffer_start', self.index_buffer_start)
        print('\tindex_buffer_end', self.index_buffer_end)
        print('\tindex', index)

        x0 = self._x[self.index0]

        # c0 = self.coeffs[self._n_diffs:].copy()

        # Finish the coefficient
        self.finish_poly(self._x[index] - x0,
                         self._y[index],
                         self.coeffs)
        sample_step = up((index - self.index_buffer_start)**0.5)

        # _x = self._x[self.index_buffer_start:index:sample_step]
        # _y = self._y[self.index_buffer_start:index:sample_step, :, 0]

        # print(self.coeffs[:,0])
        # coeffs = self.coeffs.copy()
        # plt.clf()
        # for i_diff, n in enumerate(range(self._n_coeffs - up(1),
        #                                  self._n_diffs, - 1)):
        #     print(n)
        #     print(coeffs[:, 0])
        #     plt.plot(_x, _y[:, i_diff], '.', label = 'data')
        #     plt.plot(_x, interpolate(_x - x0,
        #                             coeffs[:,0],
        #                             n),
        #             label = 'poly')

        #     plt.legend()
        #     # raise Exception
        #     plt.show()

        #     diff.in_place(coeffs, n)
        # assert np.allclose(c0, self.coeffs[self._n_diffs:], 1e-14, 0.)

        return self.excess_obj.call(self._x,
                                  self._y,
                                  self.coeffs,
                                  x0,
                                  self.index_buffer_start + sample_step - up(1),
                                  index,
                                  sample_step)
    # ------------------------------------------------------------------
    def _update_first_stage(self, index2: float, excess2: float):
        print('_update_first_stage')
        # Secant method
        estimate = (excess2 * (self.index1 - index2) / (excess2 - self.excess1))
        if estimate < 0.: # Fall back
            estimate = index2 - self.index1
        self._until_next = up(1.5 * estimate)
        self.excess1 = excess2
        self.index1 = index2
    # ------------------------------------------------------------------
    def _second_stage(self,
                   index1: float,
                   index2: float,
                   excess1: float,
                   excess2: float) -> up:
        print('_second_stage')
        """Binomial search and square root sample validation."""
        index_low: f64 = index1
        index_high: f64 = index2
        excess_low: f64 = excess1
        excess_high: f64 = excess2
        print('\tindexes', index1, index2)
        print('\texcess', excess1, excess2)

        while index_high - index_low > 1.:

            index_try = _estimate_index(index1, index2,
                                        excess1, excess2,
                                        index_low, index_high,
                                        excess_low, excess_high)
            # Checking validity
            excess = self._check(up(index_try))
            print('\texcess', excess)
            # Update the low and high
            if excess < 0.:
                index_low = round(index_try, 0)
                excess_low = excess
            else:
                index_high = round(index_try, 0)
                excess_high = excess

            index1 = index2
            index2 = f64(index_try)
            excess1 = excess2
            excess2 = excess
        return up(index_low)
    # ------------------------------------------------------------------
    def _update(self, index_compress: up):
        print('_update')

        self.index0 += up(1)
        self._x[self.index0] = self._x[index_compress]
        self._y[self.index0] = self._y[index_compress]
        index_buffer_start_next = index_compress + up(1)
        self._until_next = index_buffer_start_next - self.index_buffer_start
        self.index_buffer_start = index_buffer_start_next

        self._prepare()
    # ------------------------------------------------------------------
    def _compress(self) -> bool:
        print('_compress')
        # First stage
        index2_uintp: up = self.index_buffer_end - up(1)
        excess2 = self._check(index2_uintp)
        print('\texcess', excess2)
        index2 = f64(index2_uintp)
        if excess2 < 0.:
            self._update_first_stage(index2, excess2)
            return False
        # Second stage
        index_compress: up = self._second_stage(self.index1, index2,
                                                self.excess1, excess2)
        self._update(index_compress)
        return True
    # ------------------------------------------------------------------
    def append(self, x: float, y: F64Array[N_Diffs, N_Vars]) -> None:
        print('append')
        # Making sure the buffer has space
        if self._size == self.index_buffer_end:
            self._get_space()

        # Adding to buffer
        self._x[self.index_buffer_end] = x
        self._y[self.index_buffer_end] = y

        self.index_buffer_end += up(1)

        # Maybe compressing
        self._until_next -= up(1)
        print('\t_until_next', self._until_next)
        if self._until_next == up(0):
            self._compress()
    # ------------------------------------------------------------------
    def open(self):
        ...
    # ------------------------------------------------------------------
    def close(self):
        print('close')
        while self._compress():
            pass

        self._update(self.index_buffer_end - up(1))
