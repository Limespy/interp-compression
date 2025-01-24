import numpy as np

from .._lnumpy import f64
from .._lnumpy import F64Array
from .._lnumpy import inf64
from .._lnumpy import up
from .._root import _linear
from .._root import _shifted_rf
from .._typing import BufferSize
from .._typing import Excess
from .._typing import fIndex
from .._typing import Index
from .._typing import N_Compressed
from .._typing import N_Diffs
from .._typing import N_Points
from .._typing import N_Uncompressed
from .._typing import N_Vars
from .._typing import X
from .._typing import XSingle
from .._typing import Y
from .._typing import YSingle
from .._typing import Length
from .._errorfunctions import _Errorclass
# ======================================================================
_MIN_CONVERGENCE = 0.1
_0_Index = Index(0)
_1_Index = Index(1)
# ======================================================================
class _XYDynArray[*YSingleShape,
                  N_CompressedTV: N_Compressed,
                  N_UncompressedTV: N_Uncompressed,
                  N_PointsTV: N_Points]:
    _size: Index
    _size_previous: Index
    _x: X
    _y: Y
    index0: Index
    index_b0: Index
    index_b1: Index
    # ------------------------------------------------------------------
    def __init__(self,
                 y_single_shape: tuple[*YSingleShape],
                 preallocate: Length = 322, # From Lucas sequence
                 ) -> None:
        self.__init(y_single_shape, preallocate)
    # ------------------------------------------------------------------
    def __init(self, y_single_shape: tuple[*YSingleShape], preallocate: int) -> None:
        self.index0 = _0_Index - _1_Index# Index of x0 and y0
        self.index_b0 = _1_Index
        self.index_b1 = _1_Index

        self._size: BufferSize = Index(preallocate)
        self._size_previous = Index(self._size * 0.618033988749) # golden ratio
        self._x = np.zeros((self._size, ), f64)
        self._y = np.zeros((self._size, *y_single_shape), f64) # type: ignore[arg-type]
    # ------------------------------------------------------------------
    @property
    def xc(self) -> F64Array[N_CompressedTV]:
        return self._x[:self.index0 + _1_Index]
    # ------------------------------------------------------------------
    @property
    def yc(self) -> F64Array[N_CompressedTV, *YSingleShape]: # type: ignore[type-var]
        return self._y[:self.index0 + _1_Index]
    # ------------------------------------------------------------------
    @property
    def xb(self) -> F64Array[N_UncompressedTV]:
        return self._x[self.index_b0:self.index_b1]
    # ------------------------------------------------------------------
    @property
    def yb(self) -> F64Array[N_UncompressedTV, *YSingleShape]: # type: ignore[type-var]
        return self._y[self.index_b0:self.index_b1]
    # ------------------------------------------------------------------
    @property
    def x(self) -> F64Array[N_PointsTV]:
        return np.concatenate((self.xc, self.xb))
    # ------------------------------------------------------------------
    @property
    def y(self) -> F64Array[N_PointsTV, *YSingleShape]: # type: ignore[type-var]
        return np.concatenate((self.yc, self.yb))
    # ------------------------------------------------------------------
    @property
    def lenc(self) -> Length:
        return Length(self.index0 + _1_Index)
    # ------------------------------------------------------------------
    @property
    def lenb(self) -> Length:
        return Length(self.index_b1 - self.index_b0)
    # ------------------------------------------------------------------
    def __len__(self) -> Length:
        return Length(self.index0 + _1_Index + self.index_b1 - self.index_b0)
    # ------------------------------------------------------------------
    def _unify(self, index_new_buffer_start: Index) -> None:
        """Makes the x and y arrays contiguous.

        Moves the uncomporessed buffer section back next to the end of
        of the compressed section
        """
        shift = self.index_b0 - index_new_buffer_start

        index_new_buffer_end: Index = self.index_b1 - shift

        self._x[index_new_buffer_start:index_new_buffer_end] = self.xb
        self._y[index_new_buffer_start:index_new_buffer_end] = self.yb

        self.index_b0 = index_new_buffer_start
        self.index_b1 = index_new_buffer_end
    # ------------------------------------------------------------------
    def _allocate(self) -> None:
        """Increases the size of x and y arrays.

        Allocation size based on Lucas numbers
        """

        self._x = np.concatenate((self._x,
                                 np.zeros((self._size_previous, ), f64)))
        self._y = np.concatenate((self._y,
                                 np.zeros((self._size_previous,
                                           *self._y.shape[1:]), f64)))
        _allocated_current: up = self._size
        self._size += self._size_previous
        self._size_previous = _allocated_current
    # ------------------------------------------------------------------
    def _get_space(self) -> None:
        index_new_buffer_start: Index = self.index0 + _1_Index

        if self.index_b0 == index_new_buffer_start:
            self._allocate()
        else:
            self._unify(index_new_buffer_start)
# ======================================================================
class _StreamBase[*YSingleShape](
    _XYDynArray[*YSingleShape, N_Compressed, N_Uncompressed, N_Points],
    _Errorclass[*YSingleShape]):
    excess1: Excess
    index1: fIndex
    until_next: Index
    # ------------------------------------------------------------------
    def __init__(self, y_single_shape: tuple[*YSingleShape], preallocate: int
                 ) -> None:
        self.__init(y_single_shape, preallocate)
    # ------------------------------------------------------------------
    def __init(self, y_single_shape: tuple[*YSingleShape], preallocate: int
               ) -> None:
        self._XYDynArray__init(y_single_shape, preallocate) # type: ignore[attr-defined]
        self.excess1: Excess = inf64
    # ------------------------------------------------------------------
    def open(self, x: XSingle,
             y: F64Array[*YSingleShape], # type: ignore[type-var]
             initial: int = 5
             ) -> None:
        """Sets inidices to beginning, inputs the first value, and makes the
        compressor ready to receive values."""
        self.index0 = _0_Index  # Index of x0 and y0
        self.index_b0 = _1_Index
        self.index_b1 = _1_Index
        self.until_next = Index(initial) # TODO some heuristic for this
        self._x[self.index0] = x
        self._y[self.index0] = y
        self._prepare(y)
    # ------------------------------------------------------------------
    def _prepare(self, y0: F64Array[*YSingleShape]) -> None:
        # Setting up polynomials
        self._prepare_coeffs(y0)

        # Checking y-values for the minimum value
        self.excess1: Excess = self._minimum(y0)
        self.index1: fIndex = f64(0.)
    # ------------------------------------------------------------------
    def close(self) -> None:
        """Compresses the rest of uncompressed data"""
        while self._compress():
            pass
        self._update(self.index_b1 - self.index_b0 - _1_Index)
    # ------------------------------------------------------------------
    def append(self, x: XSingle, y: F64Array[*YSingleShape]) -> None:
        """Inserts the (x, y) datum and maybe tries to compress

        Parameters
        ----------
        x : XSingle
            x-value of the point
        y : F64Array[
            y-values of the point
        """
        # Making sure the buffer has space
        if self._size == self.index_b1:
            self._get_space()

        # Adding to buffer
        self._x[self.index_b1] = x
        self._y[self.index_b1] = y

        if self.until_next == _0_Index:
            self._compress()
        else:
            self.index_b1 += _1_Index
            self.until_next -= _1_Index
    # ------------------------------------------------------------------
    def _compress(self) -> bool:
        """Trying compression. First with stage 1 inside the function and if 
        optimal point exceeded, then calling stage 2 and updating the buffer.

        Returns
        -------
        bool
            Whether optimal point was found or not
        """
        # Stage 1
        index_next = self.index_b1 - self.index_b0
        excess2 = self._check(index_next)
        index2 = fIndex(index_next)
        if excess2 < 0.:
            self._update_stage_1(index2, excess2)
            return False
        # Stage 2
        self._update(self._stage_2(self.index1, index2,
                                   self.excess1, excess2))
        return True
    # ------------------------------------------------------------------
    def _update_stage_1(self, index2: fIndex, excess2: Excess) -> None:
        """Next step is either

        - geometric mean of secant method estimate and double the previous step
        - or double the previous step
        
        depending on whether secant method estimate would be backwards

        Parameters
        ----------
        index2 : fIndex
            Index that was tried
        excess2 : Excess
            Excess at that index
        """

        # Secant method
        excess_diff = self.excess1 - excess2
        # secant = (e2 / (e1 - e2) * (i2 - i1))
        # double previous = 2 * (i2 - i1)
        scaler = (2. * excess2 / excess_diff)**0.5 if excess_diff < 0. else 2.

        # If step is less than the amount currently in the buffer,
        # until_next is 0, if larger, the difference
        leftover = self.index_b1 - self.index_b0

        self.until_next = (max(Index(scaler * (index2 - self.index1)), # type: ignore[call-overload]
                               leftover)
                           - leftover)

        self.excess1 = excess2
        self.index1 = index2
    # ------------------------------------------------------------------
    def _update(self, index: Index) -> None:
        """

        Parameters
        ----------
        index : Index
            index relative to the uncomporessed buffer
        """

        self.index0 += _1_Index
        index_compress = index + self.index_b0
        y0 = self._y[index_compress]

        self._x[self.index0] = self._x[index_compress]
        self._y[self.index0] = y0

        index_b0_next = index_compress + _1_Index

        step = index_b0_next - self.index_b0
        leftover = self.index_b1 - index_b0_next
        self.until_next = (max(step, leftover) # type: ignore[call-overload]
                           - leftover)

        self.index_b0 = index_b0_next

        self._prepare(y0)
    # ------------------------------------------------------------------
    @staticmethod
    def _estimate_index(index1: fIndex,
                    index2: fIndex,
                    excess1: Excess,
                    excess2: Excess,
                    index_low: fIndex,
                    index_high: fIndex,
                    excess_low: Excess,
                    excess_high: Excess) -> fIndex:
        """Heuristics for selecting the next index to try.

        Constraining index: index_low < index_try < index_high
        """
        index_try = round(_linear(index1, excess1, index2, excess2), 0)
        if not (index_low < index_try < index_high):
            # Secant not converging, using regula falsi
            index_try = round(_shifted_rf(index_low, excess_low,
                                    index_high, excess_high), 0)
        if (abs(index_try - index2) <
            (_MIN_CONVERGENCE * (index_high - index_low))):
            # Slow convergence, using binomial
            index_try = round((index_high + index_low) * 0.5)
        return round(index_try,  0)
    # ------------------------------------------------------------------
    def _step_stage_2(self,
                      index1: fIndex,
                      index2: fIndex,
                        excess1: Excess,
                        excess2: Excess,
                        index_low: fIndex,
                        index_high: fIndex,
                        excess_low: Excess,
                        excess_high: Excess
                        ) -> tuple[fIndex, fIndex, Excess, Excess,
                                   fIndex, fIndex, Excess, Excess]:
        index_try = self._estimate_index(index1, index2,
                                         excess1, excess2,
                                         index_low, index_high,
                                         excess_low, excess_high)
        # Checking validity
        excess = self._check(Index(index_try))

        return ((index2, index_try, excess2, excess,
                index_try, index_high, excess, excess_high)
                if excess < 0. else
                (index2, index_try, excess2, excess,
                index_low, index_try, excess_low, excess))
    # ------------------------------------------------------------------
    def _stage_2(self,
                   index1: fIndex,
                   index2: fIndex,
                   excess1: fIndex,
                   excess2: fIndex) -> Index:
        """Binomial search and square root sample validation."""
        index_low: f64 = index1
        index_high: f64 = index2
        excess_low: f64 = excess1
        excess_high: f64 = excess2

        while index_high - index_low > 1.:
            (index1,
             index2,
             excess1,
             excess2,
             index_low,
             index_high,
             excess_low,
             excess_high) = self._step_stage_2(index1, index2,
                                               excess1, excess2,
                                               index_low, index_high,
                                               excess_low, excess_high)
        return Index(index_low)
    # ------------------------------------------------------------------
    def _check(self, index: Index) -> Excess:
        """Calculates excess, index is index of last item contained.

        Parameters
        ----------
        index : Index
            Index relative to uncompressed buffer start
        """
        if index == _0_Index:
            return self.excess1

        x0 = self._x[self.index0]
        index_global = index + self.index_b0

        # Finish the coefficient
        self._finish_coeffs(self._x[index_global] - x0, self._y[index_global])

        sample_step = Index(index**0.5)
        start = self.index_b0 + sample_step - _1_Index # Starting one step away
        stop = index_global - sample_step//2 # Stopping approx one step away

        return self._calc_excess(x0, start, stop, sample_step)
    # ------------------------------------------------------------------
    def _prepare_coeffs(self, ya: F64Array[*YSingleShape]) -> None:
        ...
    # ------------------------------------------------------------------
    def _finish_coeffs(self, Dx: XSingle, yb: F64Array[*YSingleShape]
                       ) -> None:
        ...
