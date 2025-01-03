# ======================================================================
def diff(p: Float64Array):
    out = np.arange(1., len(p), dtype = np.float64)
    out *= p[1:]
    return out
# ======================================================================
def _make_poly(n: int):

    _1_f = rfrange(n)

    NM = np.ones((n, 2 * n), dtype = np.float64)
    M = np.ones((n, n), dtype = np.float64)

    for i in range(1, n):
        NM[i] = NM[i-1]
        NM[i][i-1] = 0.
        NM[i][i+1:] *= np.arange(2, 2* n - i + 1, dtype = np.float64)

    N = NM[:,:n]
    M = NM[:,n:]

    M_1 = linalg.inv(M)
    print('M_1', M_1)
    return _1_f, M_1, N
# ----------------------------------------------------------------------
def make_poly(x0: float, y0: Y, x1: float, y1: Y) -> PolyCoeff:
    print(y0, y1)

    n = len(y0)
    y_alpha = np.array(y0)
    y_beta = np.array(y1)
    Dx = x1 - x0
    Dxn = Dx

    for i in range(1, n):
        y_alpha[i] *= Dxn
        y_beta[i] *= Dxn
        Dxn *= Dx

    _1_f, M_1, N = _make_poly(n)
    p_alpha = _1_f *  y_alpha
    p_raw = np.concatenate((p_alpha, M_1 @ (y_beta - N @ p_alpha)))
    print('p_raw', p_raw)

    _1_Dx = 1./(x1 - x0)
    _1_Dxn = _1_Dx

    for i in range(1, len(p_raw)):
        p_raw[i] *= _1_Dxn
        _1_Dxn *= _1_Dx

    print('p', p_raw)

    return p_raw
