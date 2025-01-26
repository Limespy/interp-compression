import warnings
from typing import Any
from typing import cast
from typing import NamedTuple

import numpy as np
from limesqueezer.poly import interpolate
from matplotlib import pyplot as plt

from . import sequential
from .._lnumba import IS_NUMBA
from .._lnumpy import F64Array
from .._lnumpy import inf64
from .._types import Excess
from .._types import fIndex
from .._types import Index
from .._types import XSingle
# ======================================================================
warnings.filterwarnings('ignore',
                        '.*overflow encountered in scalar add',
                        RuntimeWarning)
warnings.filterwarnings('ignore',
                        '.*overflow encountered in scalar subtract',
                        RuntimeWarning)
warnings.filterwarnings('error',
                        'invalid value encountered in scalar power',
                        RuntimeWarning)
# ----------------------------------------------------------------------
def _pause_input():
    if input() == 'q':
        raise SystemExit(2)
# ----------------------------------------------------------------------
def _pause_plt():
    plt.pause(0.01)
# ======================================================================
class PlotInfo(NamedTuple):
    data: list[Any]
    step: Any
    index: Any
# ======================================================================
class Debug_64(sequential._Sequential):
    axs: PlotInfo
    polys: PlotInfo
    compressed: list
    n_checks: int
    n_samples: int
# ======================================================================
def make_debug(n_diffs: int, *,
               is_pause: bool = True,
               is_animation: bool = True,
               is_verbose: bool = True) -> type[Debug_64]:

    # Processing options
    cls = sequential.sequential_compressors_64[n_diffs - 1]

    if (base := cls.__base__) not in (object, None):
        cls = cast(type[sequential.Sequential], base)

    pause = (_pause_input if is_pause
             else (_pause_plt if is_animation else (lambda : None)))

    _print = print if is_verbose else lambda *_ : None

    class Inner(cls): # type: ignore[valid-type,misc]
        axs: PlotInfo
        polys: PlotInfo
        compressed: list
        n_checks: int
        n_samples: int
        # --------------------------------------------------------------
        def __init__(self, rtol, atol, preallocate = 322):
            super().__init__(rtol, atol, preallocate)
            if is_animation:
                self.fig = plt.figure()
                axs = self.fig.subplots(self._n_diffs + 2, 1)
                self.axs = PlotInfo(axs[:self._n_diffs],
                                    axs[self._n_diffs],
                                    axs[self._n_diffs+1])
                self.axs.step.axhline()
                self.axs.step.grid()

                self.polys = PlotInfo(
                    [ax.plot(np.array((), np.float64),
                             np.array((), np.float64))[0]
                     for ax in self.axs.data],
                    self.axs.step.plot(np.array((), np.float64),
                                       np.array((), np.float64))[0],
                    self.axs.index.plot(np.array((), np.float64),
                                        np.array((), np.float64))[0])
                self.compressed = [ax.plot(np.array((), np.float64),
                                            np.array((), np.float64),
                                            'o',
                                            color = 'blue')[0]
                                for ax in self.axs.data]
                plt.ion()
                plt.show()
            self.n_checks = 0
            self.n_samples = 0
        # --------------------------------------------------------------
        def _plot_xy(self, x, y) -> None:
            if is_animation:
                for ax, y_diff in zip(self.axs.data, y):
                    for _y in y_diff:
                        ax.plot(x, _y, '.', color = 'black')
                plt.pause(0.001)
                pause()
        # --------------------------------------------------------------
        def _unify(self, index_new_buffer_start):
            _print(f'Unifying to index {index_new_buffer_start}')
            _print(' before', self.index_b0, self.index_b1)
            out = super()._unify(index_new_buffer_start)
            _print(' after', self.index_b0, self.index_b1)
            return out
        # --------------------------------------------------------------
        def _allocate(self):
            _print('Allocating bigger arrays')
            _print(' before', len(self._x), len(self._y))
            out = super()._allocate()
            _print(' after', len(self._x), len(self._y))
            return out
        # --------------------------------------------------------------
        def _get_space(self):
            _print('Getting more space')
            return super()._get_space()
        # --------------------------------------------------------------
        def _prepare(self, y0):
            yc = self.yc
            if is_animation:
                for i_diff, line in enumerate(self.compressed):
                    line.set_data(self.xc, yc[:,i_diff, 0])
                plt.pause(0.001)
            out = super()._prepare(y0)
            if is_animation:
                self.axs.step.plot(self.index1, self.excess1, '.', color = 'black')
            return out
        # --------------------------------------------------------------
        def _check(self, index: Index) -> Excess:
            _print(f'Check in\n', index)
            self.n_checks += 1
            excess = super()._check(index)
            if is_animation:
                self.axs.step.plot(index, excess, '.', color = 'black')
            _print('Check out\n', excess)
            plt.pause(0.001)
            pause()
            return excess
        # --------------------------------------------------------------
        def _calc_excess(self, x0, start, stop, step) -> Excess:
            _print('Calc excess in\n', x0, start, stop, step)

            index = stop + step//2
            _print(' index b', index, 'xb', self._x[index])

            _print(' x range', self._x[self.index0], self._x[index])

            _print(' ya', self._y[self.index0, :, 0], 'yb', self._y[index, :, 0])

            excess = super()._calc_excess(x0, start, stop, step)
            _print('Calc exess out\n', excess)
            return excess
        # --------------------------------------------------------------
        def _calc_diff_excess(self,
                            x0: XSingle,
                            start: Index,
                            stop: Index,
                            step: Index,
                            i_diff: Index,
                            n: Index,
                            excess: Excess) -> Excess:
            _print(f'Calc diff excess in \n', x0, start, stop, i_diff, n, excess)
            index = stop + step//2
            if is_animation:
                x_plot = np.linspace(self._x[self.index0], self._x[index])
                # _print(' x range', x_plot[0], x_plot[-1])

                y_plot = interpolate.single(x_plot - x0, self.coeffs[0], n)
                # _print(' ya', y_plot[0], 'yb', y_plot[-1])

                self.polys.data[i_diff].set_data(x_plot, y_plot)

                plt.pause(0.001)
            excess = super()._calc_diff_excess(
                x0, start, stop, step, i_diff, n, excess)
            if excess == -inf64:
                raise ValueError
            _print(f'Calc diff excess out\n', excess)
            pause()
            return excess
        # --------------------------------------------------------------
        def _calc_sample_excess(self, Dx: XSingle,
                                i_diff: Index,
                                n: Index,
                                i_sample: Index,
                                stop_var: Index,
                                excess: Excess) -> Excess:
            _print('Calc sample excess in\n',
                   Dx, i_diff, n, i_sample, stop_var, excess)
            self.n_samples += 1
            excess = super()._calc_sample_excess(
                Dx, i_diff, n, i_sample, stop_var, excess)
            _print('Calc sample excess out\n', excess)
            return excess
        # --------------------------------------------------------------
        def _update_stage_1(self, index2: fIndex, excess2: Excess) -> None:
            _print(f'Updating the first stage with index {index2} being {excess2}')
            out = super()._update_stage_1(index2, excess2)
            if is_animation:
                self.axs.step.axvline(index2 + fIndex(self.until_next),
                                    color = 'orange')
            return out
        # --------------------------------------------------------------
        def _estimate_secant(self,
                       index1: fIndex,
                       index2: fIndex,
                       excess1: Excess,
                       excess2: Excess) -> fIndex:
            _print('Estimating with secant in\n',
                  index1, index2, excess1, excess2)
            index = super()._estimate_secant(index1, index2, excess1, excess2)
            _print('Estimating with secant out\n', index)
            return index
        # --------------------------------------------------------------
        def _step_stage_2(self, index1, index2, excess1, excess2,
                        index_low, index_high, excess_low, excess_high):
            _print('Stage 2 step int\n',
                index1, index2, excess1, excess2, '\n',
                index_low, index_high, excess_low, excess_high)
            (index1,
            index2,
            excess1,
            excess2,
            index_low,
            index_high,
            excess_low,
            excess_high) = super()._step_stage_2(index1, index2,
                                                    excess1, excess2,
                                                    index_low, index_high,
                                                    excess_low, excess_high)
            _print('Stage 2 step out\n',
                index1, index2, excess1, excess2, '\n',
                index_low, index_high, excess_low, excess_high)
            return super()._step_stage_2(index1, index2, excess1, excess2,
                                        index_low, index_high, excess_low, excess_high)
        # --------------------------------------------------------------
        def _stage_2(self, index1, index2, excess1, excess2):
            _print('Stage 2 in\n',
                index1, index2, excess1, excess2)
            index = super()._stage_2(index1, index2, excess1, excess2)
            _print('Stage 2 out\n', index)
            return index
        # --------------------------------------------------------------
        def _update(self, index: Index) -> None:
            _print(f'Compressing to index {index}')
            if is_animation:
                self.axs.step.clear()
                self.axs.step.axhline()
            out = super()._update(index)
            return out
        # --------------------------------------------------------------
        def _estimate_index(self,
                            index1: fIndex,
                            index2: fIndex,
                            excess1: Excess,
                            excess2: Excess,
                            index_low: fIndex,
                            index_high: fIndex,
                            excess_low: Excess,
                            excess_high: Excess) -> fIndex:
            _print('Estimating index\n',
                index1, index2, excess1, excess2, '\n',
                index_low, index_high, excess_low, excess_high)
            estimate = super()._estimate_index(index1, index2,
                                        excess1, excess2,
                                        index_low, index_high,
                                        excess_low, excess_high)
            _print(f'Estimate {estimate}')
            return estimate
        # --------------------------------------------------------------
        def append(self, x: float, y):
            _print('Append in\n', x, y)
            _print(' until next', self.until_next)
            if is_animation:
                self._plot_xy(x, y)

            return super().append(x, y)
    return Inner
