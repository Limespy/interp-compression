from typing import Generic
from typing import TYPE_CHECKING
from typing import TypeVar

import numpy as np

from .. import _lnumba as nb
from .._errorfunctions import MaxAbs_Sequential
from .._lnumpy import f64
from .._lnumpy import up
from ..poly import diff
from ..poly import make
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
BufferSize = TypeVar('BufferSize', bound = np.uintp)
# ======================================================================
# @nb.njit
def _linear(x0: float, y0: float, x1: float, y1: float) -> float:
    return (x1 * y0 - x0 * y1) / (y0 - y1)
# ----------------------------------------------------------------------
# @nb.njit
def _shifted_rf(x_rf: float, x_low: float, x_high: float) -> float:
    diff_to_low = x_rf - x_low
    diff_to_high = x_high - x_rf
    return (x_rf + diff_to_low * 0.5 if diff_to_low < diff_to_high
            else x_rf - diff_to_high * 0.5)
# ----------------------------------------------------------------------
# @nb.njit
def _estimate_index(index1: float,
                    index2: float,
                    excess1: float,
                    excess2: float,
                    index_low: float,
                    index_high: float,
                    excess_low: float,
                    excess_high: float):
    index_secant = _linear(index1, excess1, index2, excess2)
    index_try = index_secant

    if index_try < index_low or index_try > index_high:
        # Secant not converging
        index_rf = _linear(index_low, excess_low, index_high, excess_high)
        index_try = _shifted_rf(index_rf, index_low, index_high)
        # index_bin = (index_high + index_low) / 2
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

        self.excess_obj = MaxAbs_Sequential(rtol, atol)
        self.prepare_poly: make.Prepare[N_Diffs, N_Diffs, N_Vars]
        self.finish_poly: make.Finish[N_Diffs, N_Diffs, N_Vars]
        self.prepare_poly, self.finish_poly = make.get_split_maker(n_diffs)

        self.coeffs: list[F64Array[int, N_Vars]] = [
            np.zeros((n_coeffs, n_vars))
            for n_coeffs in range(n_diffs * 2, n_diffs, -1)]

        self.index0: np.uintp = 0 # Index of x0 and y0
        self.index1: float = 1.
        self.index_buffer_start: np.uintp = 1
        self.index_buffer_end: np.uintp = 1
        self._size: BufferSize = preallocate
        self._allocated_previous = np.uintp(self._size * 0.618033988749)
        self._until_next: np.uintp = 5 # TODO some heuristic for this


        self._x: F64Array[BufferSize] = np.zeros((self._size, ), f64)
        self._y: F64Array[BufferSize, N_Diffs, N_Vars] = np.zeros((self._size, n_diffs, n_vars), f64)

        self.x0: float = np.inf
    # ------------------------------------------------------------------
    @property
    def xc(self) -> F64Array[N_Compressed]:
        return self._x[:self.index0 + 1]
    # ------------------------------------------------------------------
    @property
    def yc(self) -> F64Array[N_Compressed, N_Diffs, N_Vars]:
        return self._y[:self.index0 + 1]
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
        return np.concatenate(self.xc, self.yb)
    # ------------------------------------------------------------------
    @property
    def y(self) -> F64Array[N_Points, N_Diffs, N_Vars]:
        return np.concatenate(self.yc, self.yb)
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

        self._x = np.concatenate(self._x,
                                 np.zeros((self._allocated_previous, ), f64))
        self._y = np.concatenate(self._y,
                                 np.zeros((self._allocated_previous,
                                           *self._y.shape[1:]), f64))
        _allocated_current: np.uintp = self._size
        self._size += self._allocated_previous
        self._allocated_previous = _allocated_current
    # ------------------------------------------------------------------
    def _get_space(self):
        print('_get_space')
        index_new_buffer_start: np.uintp = self.index0 + up(1)

        if self.index_buffer_start == index_new_buffer_start:
            self._allocate()
        else:
            self._unify(index_new_buffer_start)
    # ------------------------------------------------------------------
    def _check(self, index: np.uintp) -> float:
        """Calculates excess, index is index of last item contained."""
        print('_check')
        print('index_buffer_start', self.index_buffer_start)
        print('index', index)
        print('step')
        sample_step = up((self.index_buffer_end - self.index_buffer_start)**0.5)
        sampling: UPArray[N_Samples] = np.arange(
            self.index_buffer_start, index, sample_step, up)

        x0 = self._x[self.index0]
        self.finish_poly(self._x[index] - x0, self._y[index], self.coeffs)
        return self.excess_obj.call(self._x[sampling] - x0,
                                  self._y[sampling],
                                  self.coeffs)
    # ------------------------------------------------------------------
    def _update_first_stage(self, index2: float, excess2: float):
        print('_update_first_stage')
        # Secant method
        estimate = (excess2 * (self.index1 - index2) / (excess2 - self.excess1))
        if estimate < 0.: # Fall back
            estimate = index2 - self.index1
        self._until_next_try = np.uintp(1.5 * estimate)
        self.excess1 = excess2
        self.index1 = index2
    # ------------------------------------------------------------------
    def _second_stage(self,
                   index1: float,
                   index2: float,
                   excess1: float,
                   excess2: float) -> np.uintp:
        print('_second_stage')
        """Binomial search and square root sample validation."""
        index_low: f64 = index1
        index_high: f64 = index2
        excess_low: f64 = excess1
        excess_high: f64 = excess2

        while index_low - index_low < 1:

            index_try = _estimate_index(index1, index2,
                                        excess1, excess2,
                                        index_low, index_high,
                                        excess_low, excess_high)

            # Checking validity
            excess = self._check(np.uintp(index_try))
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
        return np.uintp(index_low)
    # ------------------------------------------------------------------
    def _update(self, index_compress: np.uintp):
        print('_update')

        self.index0 += np.uintp(1)
        self._x[self.index0] = self._x[index_compress]
        self._y[self.index0] = self._y[index_compress]
        self._until_next = (index_compress
                            + np.uintp(1)
                            - self.index_buffer_start)
        self.index_buffer_start = index_compress + np.uintp(1)

        self._prepare()
    # ------------------------------------------------------------------
    def _compress(self) -> bool:
        print('_compress')
        # First stage
        index2_uintp: np.uintp = self.index_buffer_end - 1
        excess2 = self._check(index2_uintp)
        print('excess', excess2)
        index2 = f64(index2_uintp)
        if excess2 < 0.:
            self._update_first_stage(index2, excess2)
            return
        else: # Second stage
            index_compress: np.uintp = self._second_stage(self.index1, index2,
                                                self.excess1, excess2)
            self._update(index_compress)
            return True
    # ------------------------------------------------------------------
    def _prepare(self):
        print('_prepare')

        # Setting up 0-point
        y0 = self._y[self.index0]

        # Setting up polynomials
        self.prepare_poly(y0, self.coeffs)
        # Checking y-values for the minimum value
        self.excess1 = self.excess_obj.minimum(y0)
        self.index1 = f64(self.index_buffer_start)
    # ------------------------------------------------------------------
    def prime(self, x: float, y: F64Array[N_Diffs, N_Vars]):
        """Sets first value and makes the compressor ready to receive
        values."""
        print('prime')
        self._x[0] = x
        self._y[0] = y
        self._prepare()
    # ------------------------------------------------------------------
    def append(self, x: float, y: F64Array[N_Diffs, N_Vars]) -> None:
        print('append')
        # Making sure the buffer has space
        if self._size == self.index_buffer_end:
            self._unify()
            if self._size == self.index_buffer_end:
                self._allocate()

        # Adding to buffer
        self._x[self.index_buffer_end] = x
        self._y[self.index_buffer_end] = y

        self.index_buffer_end += 1

        # Maybe compressing
        self._until_next -= 1
        if self._until_next == 0:
            self._compress()
    # ------------------------------------------------------------------
    def open(self):
        ...
    # ------------------------------------------------------------------
    def close(self):
        ...
