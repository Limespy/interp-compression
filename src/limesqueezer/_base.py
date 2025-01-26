from warnings import filterwarnings

import numpy as np

from ._errorfunctions import _Errorclass
from ._lnumpy import f64
from ._lnumpy import F64Array
from ._lnumpy import inf64
from ._lnumpy import up
from ._types import BufferSize
from ._types import Excess
from ._types import fIndex
from ._types import Index
from ._types import Length
from ._types import N_Compressed
from ._types import N_Diffs
from ._types import N_Points
from ._types import N_Uncompressed
from ._types import N_Vars
from ._types import X
from ._types import XSingle
from ._types import Y
from ._types import YSingle
# ======================================================================
_MIN_CONVERGENCE = 0.1 # TODO some better heuristic for this

_0_Index = Index(0)
_1_Index = Index(1)
_0_fIndex = fIndex(0.)
_1_fIndex = fIndex(1.)
_2_fIndex = fIndex(2.)

filterwarnings('error',
               'invalid value encountered in scalar subtract',
               RuntimeWarning)
# ======================================================================
class _XYDynArray[*YSingleShape,
                  N_CompressedTV: N_Compressed,
                  N_UncompressedTV: N_Uncompressed,
                  N_PointsTV: N_Points]:
    _size: Index
    _size_previous: Index
    _x: X
    _y: Y
    _index0: Index
    _index_b0: Index
    _index_b1: Index
    # ------------------------------------------------------------------
    def __init__(self,
                 y_single_shape: tuple[*YSingleShape],
                 preallocate: Length = 322, # From Lucas sequence
                 ) -> None:
        self.__init(y_single_shape, preallocate)
    # ------------------------------------------------------------------
    def __init(self, y_single_shape: tuple[*YSingleShape], preallocate: int) -> None:
        self._index0 = _0_Index - _1_Index# Index of x0 and y0
        self._index_b0 = _1_Index
        self._index_b1 = _1_Index

        self._size: BufferSize = Index(preallocate)
        self._size_previous = Index(self._size * 0.618033988749) # golden ratio
        self._x = np.zeros((self._size, ), f64)
        self._y = np.zeros((self._size, *y_single_shape), f64) # type: ignore[arg-type]
    # ------------------------------------------------------------------
    @property
    def xc(self) -> F64Array[N_CompressedTV]:
        return self._x[:self._index0 + _1_Index]
    # ------------------------------------------------------------------
    @property
    def yc(self) -> F64Array[N_CompressedTV, *YSingleShape]: # type: ignore[type-var]
        return self._y[:self._index0 + _1_Index]
    # ------------------------------------------------------------------
    @property
    def xb(self) -> F64Array[N_UncompressedTV]:
        return self._x[self._index_b0:self._index_b1]
    # ------------------------------------------------------------------
    @property
    def yb(self) -> F64Array[N_UncompressedTV, *YSingleShape]: # type: ignore[type-var]
        return self._y[self._index_b0:self._index_b1]
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
        return Length(self._index0 + _1_Index)
    # ------------------------------------------------------------------
    @property
    def lenb(self) -> Length:
        return Length(self._index_b1 - self._index_b0)
    # ------------------------------------------------------------------
    def __len__(self) -> Length:
        return Length(self._index0 + _1_Index + self._index_b1 - self._index_b0)
    # ------------------------------------------------------------------
    def _unify(self, index_new_buffer_start: Index) -> None:
        """Makes the x and y arrays contiguous.

        Moves the uncomporessed buffer section back next to the end of
        of the compressed section
        """
        shift = self._index_b0 - index_new_buffer_start

        index_new_buffer_end: Index = self._index_b1 - shift

        self._x[index_new_buffer_start:index_new_buffer_end] = self.xb
        self._y[index_new_buffer_start:index_new_buffer_end] = self.yb

        self._index_b0 = index_new_buffer_start
        self._index_b1 = index_new_buffer_end
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
        index_new_buffer_start: Index = self._index0 + _1_Index

        if self._index_b0 == index_new_buffer_start:
            self._allocate()
        else:
            self._unify(index_new_buffer_start)
# ======================================================================
class _StreamBase[*YSingleShape](
    _XYDynArray[*YSingleShape, N_Compressed, N_Uncompressed, N_Points],
    _Errorclass[*YSingleShape]):
    excess1: Excess
    index1: fIndex
    _until_next: Index
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
        self._index0 = _0_Index  # Index of x0 and y0
        self._index_b0 = _1_Index
        self._index_b1 = _1_Index
        self._until_next = Index(initial) # TODO some heuristic for this
        self._x[self._index0] = x
        self._y[self._index0] = y
        self._prepare(y)
    # ------------------------------------------------------------------
    def append(self, x: XSingle, y: F64Array[*YSingleShape]) -> None:
        """Inserts the (x, y) datum and maybe tries to compress.

        Parameters
        ----------
        x : XSingle
            x-value of the point
        y : F64Array[
            y-values of the point
        """
        # Making sure the buffer has space
        if self._size == self._index_b1:
            self._get_space()

        # Adding to buffer
        self._x[self._index_b1] = x
        self._y[self._index_b1] = y

        if self._until_next == _0_Index:
            self._compress()
        else:
            self._until_next -= _1_Index
        self._index_b1 += _1_Index
    # ------------------------------------------------------------------
    def close(self) -> None:
        """Compresses the rest of uncompressed data."""
        while self._compress():
            pass
        left = self._index_b1 - self._index_b0
        if left:
            self._update(left - _1_Index)
    # ------------------------------------------------------------------
    def _prepare(self, y0: F64Array[*YSingleShape]) -> None:
        # Setting up polynomials
        self._prepare_coeffs(y0)

        # Checking y-values for the minimum value
        self.excess1: Excess = self._minimum(y0)
        self.index1: fIndex = _0_fIndex
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
        index_next = self._index_b1 - self._index_b0
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
    def _check(self, index: Index) -> Excess:
        """Calculates excess, index is index of last item contained.

        Parameters
        ----------
        index : Index
            Index relative to uncompressed buffer start
        """
        if index == _0_Index:
            return self.excess1

        x0 = self._x[self._index0]
        index_global = index + self._index_b0

        # Finish the coefficient
        self._finish_coeffs(self._x[index_global] - x0, self._y[index_global])

        step = Index(index**0.5)
        start = self._index_b0 + step - _1_Index # Starting one step away
        stop = index_global - step//2 # Stopping approx one step away
        return self._calc_excess(x0, start, stop, step)
    # ------------------------------------------------------------------
    def _update_stage_1(self, index2: fIndex, excess2: Excess) -> None:
        """Next step is either.

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

        self._until_next = self._estimate_next_stage_1_step(index2, excess2)
        self.excess1 = excess2
        self.index1 = index2
    # ------------------------------------------------------------------
    def _estimate_next_stage_1_step(self, index2: fIndex, excess2: Excess
                                    ) -> Index:

        step = index2 - self.index1
        if (self.excess1 - excess2) < 0.:
            index_secant = self._estimate_secant(self.index1, index2,
                                                 self.excess1, excess2)

            step_secant = index_secant - self.index1
            # step = step_secant
            step = (step * step_secant)**0.5
            # step = (step + step_secant) * 0.5

        return Index(round(step))
    # ------------------------------------------------------------------
    def _estimate_secant(self,
                       index1: fIndex,
                       index2: fIndex,
                       excess1: Excess,
                       excess2: Excess) -> fIndex:
        return 2 * index2 - index1
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

        while index_high - index_low > _2_fIndex:
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

        if index_high - index_low == _2_fIndex:
            index_try = Index((index_high + index_low) * 0.5)
            excess = self._check(index_try)
            if excess < _0_fIndex:
                return index_try

        return Index(index_low)
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
    def _estimate_index(self,
                        index1: fIndex,
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
        index_try = round(self._estimate_base(index1, index2,
                                        excess1, excess2,
                                        index_low, index_high,
                                        excess_low, excess_high), 0)
        if not (index_low < index_try < index_high):
            index_try = round(self._estimate_stable(index1, index2,
                                              excess1, excess2,
                                              index_low, index_high,
                                              excess_low, excess_high), 0)
        if (abs(index_try - index2) <
            (_MIN_CONVERGENCE * (index_high - index_low))):
            # Slow convergence, using binomial
            # TODO some better heuristic for detecting and dealing with this
            index_try = round((index_high + index_low) * 0.5, 0)
        return index_try
    # ------------------------------------------------------------------
    def _estimate_base(self,
                       index1: fIndex,
                       index2: fIndex,
                       excess1: Excess,
                       excess2: Excess,
                       index_low: fIndex,
                       index_high: fIndex,
                       excess_low: Excess,
                       excess_high: Excess) -> fIndex:
        return (index_low + index_high) * 0.5
    # ------------------------------------------------------------------
    def _estimate_stable(self,
                        index1: fIndex,
                        index2: fIndex,
                        excess1: Excess,
                        excess2: Excess,
                        index_low: fIndex,
                        index_high: fIndex,
                        excess_low: Excess,
                        excess_high: Excess) -> fIndex:
        return (index_low + index_high) * 0.5
    # ------------------------------------------------------------------
    def _update(self, index: Index) -> None:
        """

        Parameters
        ----------
        index : Index
            index relative to the uncomporessed buffer
        """

        self._index0 += _1_Index
        index_compress = index + self._index_b0
        y0 = self._y[index_compress]

        self._x[self._index0] = self._x[index_compress]
        self._y[self._index0] = y0

        index_b0_next = index_compress + _1_Index

        step = index_b0_next - self._index_b0
        leftover = self._index_b1 - index_b0_next
        self._until_next = (max(step, leftover) # type: ignore[call-overload]
                            - leftover)

        self._index_b0 = index_b0_next

        self._prepare(y0)
    # ------------------------------------------------------------------
    def _prepare_coeffs(self, ya: F64Array[*YSingleShape]) -> None:
        ...
    # ------------------------------------------------------------------
    def _finish_coeffs(self, Dx: XSingle, yb: F64Array[*YSingleShape]
                       ) -> None:
        ...
