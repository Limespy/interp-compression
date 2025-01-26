from limedev.CLI import get_main
# ======================================================================
def debug(*, interval: float = 0.2,
          n_points: int = 1000,
          n_diffs: int =  4,
          n_vars: int = 1,
          no_animation: bool = False,
          no_pause: bool = False,
          no_print: bool = False) -> int:

    import numpy as np
    from .debug import make_debug

    # setup
    x_raw = np.arange(0., n_points * interval, interval, np.float64)
    _x = x_raw.reshape(-1, 1) + np.linspace(0., 1., n_vars)
    _sin = np.sin(_x)
    _cos = np.cos(_x)
    diffs = (_sin, _cos, -_sin, -_cos)
    y_raw = np.stack(diffs[:n_diffs], axis = 1)

    rtol = np.full(y_raw.shape[1:], 0.01, dtype = np.float64)
    atol = np.full(y_raw.shape[1:], 0.01, dtype = np.float64)

    stream = make_debug(n_diffs,
                        is_animation = not no_animation,
                        is_pause = not no_pause,
                        is_verbose = not no_print)(rtol, atol)

    stream.open(x_raw[0], y_raw[0])

    for x, y in zip(x_raw[1:], y_raw[1:]):
        stream.append(x, y)
    stream.close()
    print('Compression ratio', len(x_raw) / len(stream))
    print('Compressed', len(stream))
    print('Check evaluations', stream.n_checks)
    print('Sample evaluations', stream.n_samples)
    return 0
# ======================================================================
main = get_main(__name__)
