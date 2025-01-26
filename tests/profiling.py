import sys

import limesqueezer as ls
import numpy as np
# ======================================================================
# Setting process to realtime
if sys.platform == 'win32':
    # Based on:
    #   "Recipe 496767: Set Process Priority In Windows" on ActiveState
    #   http://code.activestate.com/recipes/496767/
    import win32api, win32process, win32con

    pid = win32api.GetCurrentProcessId()
    handle = win32api.OpenProcess(win32con.PROCESS_ALL_ACCESS, True, pid)
    win32process.SetPriorityClass(handle,
                                    win32process.REALTIME_PRIORITY_CLASS)
else:
    import os

    os.nice(-20 - os.nice())
# ======================================================================
N_VARS = 100
N_POINTS = 1000
INTERVAL = 0.2
X_RAW = np.arange(0., N_POINTS * INTERVAL, INTERVAL, np.float64)
_x = X_RAW.reshape(-1, 1) + np.linspace(0., 1., N_VARS)
_sin = np.sin(_x)
_cos = np.cos(_x)

Y_RAW = np.stack((_sin, _cos, -_sin, -_cos), axis = 1)

RTOL = np.full(Y_RAW.shape[1:], 0.01, dtype = np.float64)
ATOL = np.full(Y_RAW.shape[1:], 0.01, dtype = np.float64)
# ======================================================================
def appending():
    stream = ls.taylor.Sequential3_64(RTOL, ATOL)
    stream.open(X_RAW[0], Y_RAW[0])

    for index in range(1, len(X_RAW)):
        stream.append(X_RAW[index], Y_RAW[index])

    stream.close()
