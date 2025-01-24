from . import _lnumba as nb
# ======================================================================

# ======================================================================
@nb.njit
def _linear(x0: float, y0: float, x1: float, y1: float) -> float:
    return (x1 * y0 - x0 * y1) / (y0 - y1)
# ----------------------------------------------------------------------
@nb.njit
def _shifted_rf(x_low: float, y_low: float, x_high: float, y_high: float
                ) -> float:
    estimate = _linear(x_low, y_low, x_high, y_high)
    diff_to_low = estimate - x_low
    diff_to_high = x_high - estimate
    return (estimate + diff_to_low * 0.5
            if diff_to_low < diff_to_high else
            estimate - diff_to_high * 0.5)
