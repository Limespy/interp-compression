import numpy as np
import pytest
from limesqueezer import _base
from limesqueezer import _lnumba as nb
# ======================================================================
parametrize = pytest.mark.parametrize

# ======================================================================
class Test_XYDynArray:
    # ------------------------------------------------------------------
    @pytest.mark.filterwarnings(
        'ignore:overflow encountered in scalar add:RuntimeWarning',
        'ignore:overflow encountered in scalar subtract:RuntimeWarning')
    @parametrize(('n_dim',), ((1,),(2,)))
    def test_init(self, n_dim: int):
        cls = _base._XYDynArray
        shape = (1,)* n_dim
        preallocate = 100
        a = cls(shape, preallocate = preallocate)

        assert a._x.shape == (preallocate,)
        assert a._y.shape == (preallocate, *shape)

        assert a.lenc == 0
        assert a.xc.shape == (0,)
        assert a.yc.shape == (0, *shape)

        assert a.lenb == 0
        assert a.xb.shape == (0,)
        assert a.yb.shape == (0, *shape)

        assert len(a) == 0
        assert type(a.lenc) == type(a.lenb) == type(len(a)) == int
    # ------------------------------------------------------------------
    @pytest.mark.filterwarnings(
        'ignore:overflow encountered in scalar subtract:RuntimeWarning')
    def test_default_preallocation(self):
        cls = _base._XYDynArray
        shape = (1, 1)

        a = cls(shape) # default preallocation
        assert a._x.dtype == a._y.dtype == np.float64
        assert a._x.shape == (322,)
        assert a._y.shape == (322, *shape)
    # ------------------------------------------------------------------
class Test_StreamBase:
    # ------------------------------------------------------------------
    @pytest.mark.filterwarnings(
            'ignore:base class method:limesqueezer.exceptions.NotImplementedWarning',
            'ignore:overflow encountered in scalar add:RuntimeWarning',
            'ignore:overflow encountered in scalar subtract:RuntimeWarning')
    def test_init(self):
        shape = (1, 1)
        preallocate = 100
        n_points = 5

        a = _base._StreamBase(shape, preallocate = preallocate)

        assert a._x.shape == (preallocate,)
        assert a._y.shape == (preallocate, *shape)

        assert a.lenc == 0
        assert a.xc.shape == (0,)
        assert a.yc.shape == (0, *shape)

        assert a.lenb == 0
        assert a.xb.shape == (0,)
        assert a.yb.shape == (0, *shape)

        assert len(a) == 0


        x = np.linspace(0., 1., n_points, dtype = np.float64)
        y = np.zeros((n_points, *shape), dtype = np.float64)
        a.open(x[0], y[0])

        assert a.lenc == 1
        assert a.xc.shape == (1,)
        assert a.yc.shape == (1, *shape)

        assert a.lenb == 0
        assert a.xb.shape == (0,)
        assert a.yb.shape == (0, *shape)

        assert len(a) == 1

        a.append(x[1], y[1])

        assert a.lenc == 1
        assert a.xc.shape == (1,)
        assert a.yc.shape == (1, *shape)

        assert a.lenb == 1
        assert a.xb.shape == (1,)
        assert a.yb.shape == (1, *shape)

        assert len(a) == 2

        for n_buff, (_x, _y) in enumerate(zip(x[2:], y[2:]), start = 2):
            a.append(_x, _y)
            assert a.lenc == 1
            assert a.xc.shape == (1,)
            assert a.yc.shape == (1, *shape)

            assert a.lenb == n_buff
            assert a.xb.shape == (n_buff,)
            assert a.yb.shape == (n_buff, *shape)

            assert len(a) == n_buff + 1

        assert np.all(a._x[:n_points] == x)
        assert np.all(a._y[:n_points] == y)

    # ------------------------------------------------------------------
