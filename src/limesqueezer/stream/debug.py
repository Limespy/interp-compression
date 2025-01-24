import warnings
from typing import Any
from typing import NamedTuple

import numpy as np
from limesqueezer.poly import interpolate
from matplotlib import pyplot as plt

from .._typing import Excess
from .._typing import fIndex
from .._typing import Index
from .._typing import XSingle
from .diff import Diff3_64
# ======================================================================

warnings.filterwarnings('ignore',
                        '.*overflow encountered in scalar add',
                        RuntimeWarning)
warnings.filterwarnings('ignore',
                        '.*overflow encountered in scalar subtract',
                        RuntimeWarning)
# ======================================================================
class PlotInfo(NamedTuple):
    data: Any
    step: Any
    index: Any
# ======================================================================
class DiffStreamDebug(Diff3_64.__base__): # type: ignore[name-defined]
    def __init__(self, rtol, atol, preallocate = 322):
        super().__init__(rtol, atol, preallocate)
        self.fig = plt.figure()
        axs = self.fig.subplots(self._n_diffs + 2, 1)
        self.axs = PlotInfo(axs[:self._n_diffs],
                            axs[self._n_diffs],
                            axs[self._n_diffs+1])
        self.polys = PlotInfo([ax.plot(np.array((), np.float64),
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
    # ------------------------------------------------------------------
    def _plot_xy(self, x, y) -> None:
        for ax, y_diff in zip(self.axs.data, y):
            for _y in y_diff:
                ax.plot(x, _y, '.', color = 'black')
        plt.pause(0.001)
        input()
    # ------------------------------------------------------------------
    def _unify(self, index_new_buffer_start):
        print(f'Unifying to index {index_new_buffer_start}')
        print(' before', self.index_b0, self.index_b1)
        out = super()._unify(index_new_buffer_start)
        print(' after', self.index_b0, self.index_b1)
        return out
    # ------------------------------------------------------------------
    def _allocate(self):
        print('Allocating bigger arrays')
        print(' before', len(self._x), len(self._y))
        out = super()._allocate()
        print(' after', len(self._x), len(self._y))
        return out
    # ------------------------------------------------------------------
    def _get_space(self):
        print('Getting more space')
        return super()._get_space()
    # ------------------------------------------------------------------
    def _prepare(self, y0):
        yc = self.yc
        for i_diff, line in enumerate(self.compressed):
            line.set_data(self.xc, yc[:,i_diff, 0])
        plt.pause(0.001)
        return super()._prepare(y0)
    # ------------------------------------------------------------------
    def _check(self, index):
        print(f'Check in\n', index)
        excess = super()._check(index)
        self.axs.step.plot(index, excess, '.', color = 'black')
        print('Check out\n', excess)
        plt.pause(0.001)
        input()
        return excess
    # ------------------------------------------------------------------
    def _calc_excess(self, x0, start, stop, step):
        print('Calc excess in\n', x0, start, stop, step)

        index = stop + step//2
        print(' index b', index, 'xb', self._x[index])

        print(' x range', self._x[self.index0], self._x[index])

        print(' ya', self._y[self.index0, 0], 'yb', self._y[index, 0])

        excess = super()._calc_excess(x0, start, stop, step)
        print('Calc exess out\n', excess)
        return excess
    # ------------------------------------------------------------------
    def _calc_diff_excess(self,
                          x0: XSingle,
                          start: Index,
                          stop: Index,
                          step: Index,
                          i_diff: Index,
                          n: Index,
                          excess: Excess):
        print(f'Calc diff excess in \n', x0, start, stop, i_diff, n, excess)
        index = stop + step//2
        x_plot = np.linspace(self._x[self.index0], self._x[index])
        # print(' x range', x_plot[0], x_plot[-1])

        y_plot = interpolate.single(x_plot - x0, self.coeffs[0], n)
        # print(' ya', y_plot[0], 'yb', y_plot[-1])

        self.polys.data[i_diff].set_data(x_plot, y_plot)

        plt.pause(0.001)
        excess = super()._calc_diff_excess(
            x0, start, stop, step, i_diff, n, excess)
        print(f'Calc diff excess out\n', excess)
        input()
        return excess
    # ------------------------------------------------------------------
    def _update_stage_1(self, index2: float, excess2: float):
        print(f'Updating the first stage with index {index2} being {excess2}')
        return super()._update_stage_1(index2, excess2)
    # ------------------------------------------------------------------
    def _step_stage_2(self, index1, index2, excess1, excess2,
                      index_low, index_high, excess_low, excess_high):
        print('Stage 2 step int\n',
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
        print('Stage 2 step out\n',
              index1, index2, excess1, excess2, '\n',
              index_low, index_high, excess_low, excess_high)
        return super()._step_stage_2(index1, index2, excess1, excess2,
                                     index_low, index_high, excess_low, excess_high)
    # ------------------------------------------------------------------
    def _stage_2(self, index1, index2, excess1, excess2):
        print('Stage 2 in\n',
              index1, index2, excess1, excess2)
        index = super()._stage_2(index1, index2, excess1, excess2)
        print('Stage 2 out\n', index)
        return index
    # ------------------------------------------------------------------
    def _update(self, index_compress):
        print(f'Compressing to index {index_compress}')
        return super()._update(index_compress)
    # ------------------------------------------------------------------
    # @staticmethod
    def _estimate_index(self,
                        index1: fIndex,
                        index2: fIndex,
                        excess1: Excess,
                        excess2: Excess,
                        index_low: fIndex,
                        index_high: fIndex,
                        excess_low: Excess,
                        excess_high: Excess) -> fIndex:
        print('Estimating index\n',
              index1, index2, excess1, excess2, '\n',
              index_low, index_high, excess_low, excess_high)
        estimate = super()._estimate_index(index1, index2,
                                       excess1, excess2,
                                       index_low, index_high,
                                       excess_low, excess_high)
        print(f'Estimate {estimate}')
        return estimate
    # ------------------------------------------------------------------
    def append(self, x: float, y):
        print('Append in\n', x, y)
        print(' until next', self.until_next)
        self._plot_xy(x, y)

        return super().append(x, y)
