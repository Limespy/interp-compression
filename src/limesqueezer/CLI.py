"""Command line interface for processing command line input."""
# ======================================================================
# IMPORT
import argparse
import pathlib
import sys

import numpy as np
from limedev.CLI import get_main

from . import _API as ls
from . import reference as ref
from .auxiliaries import Float64Array
from .auxiliaries import G
# ======================================================================
def run(args: list[str], use_numba: int, is_plot: bool, is_timed: bool):
    x_data, y_data = ref.raw_sine_x2_normal(1e4, std=0.00001)
    # y_data[1000] += 1
    if args[0] == 'block':
        xc, yc = ls.compress(x_data, y_data, tolerances = (1e-2, 1e-3, 1),
                    use_numba = use_numba, errorfunction = 'MaxAbs')
        print(ls.stats(x_data, xc))
    elif args[0] == 'stream':
        xc, yc = _stream(x_data, y_data, (1e-2, 1e-3, 1), use_numba)
    elif args[0] == 'both':
        xcb, ycb = ls.compress(x_data, y_data, tolerances = (1e-2, 1e-3, 1), use_numba = use_numba, initial_step = 100, errorfunction = 'MaxAbs')
        xcs, ycs = _stream(x_data, y_data, (1e-2, 1e-3, 1.), use_numba)
        for i, (xb, xs) in enumerate(zip(xcb,xcs)):
            if xb != xs:
                print(f'Deviation at {i=}, {xb=}, {xs=}')
                break
        for i, (xb, xs) in enumerate(zip(reversed(xcb),reversed(xcs))):
            if xb != xs:
                print(f'Deviation at {i=}, {xb=}, {xs=}')
                break
        print(xcb)
        print(xcs)

        xc, yc = xcb, ycb
    if is_timed: print(f'runtime {G["runtime"]*1e3:.1f} ms')
# ======================================================================
def _stream(x_data: Float64Array,
            y_data: Float64Array,
            tol: tuple[float, float, float],
            use_numba: int):

    with ls.Stream(x_data[0], y_data[0], tolerances = tol, use_numba = use_numba) as record:
        for x, y in zip(x_data[1:], y_data[1:]):
            record(x, y)
    return record.x, record.y
# ======================================================================
# def main(args: list[str] = sys.argv[1:]):
#     """The main command line app."""
#     parser = argparse.ArgumentParser(description = '')
#     is_verbose, args = get_kwarg('--verbose', args)
#     is_plot, args = get_kwarg('--plot', args)
#     is_save, args = get_kwarg('--save', args)
#     is_show, args = get_kwarg('--show', args)
#     G['timed'], args = get_kwarg('--timed', args)
#     G['debug'], args = get_kwarg('--debug', args)
#     use_numba, args = get_kwarg('--numba', args)
#     use_numba = int(use_numba)
#     if len(args) == 0:
#         print(helpstring)
#         return
#     path_cwd = pathlib.Path.cwd()

#     if is_verbose: print('Selected path is:\n\t%s' % path_cwd)
#     run(args, use_numba, is_plot, G['timed'])

main = get_main(__name__)
