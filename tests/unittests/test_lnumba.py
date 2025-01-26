from os import getenv

import pytest
from limesqueezer import _lnumba as nb
# ======================================================================
parametrize = pytest.mark.parametrize
# ======================================================================
class Test_clean:
    def test_remove_Generic(self):

        class C[T]:
            a: T
            b: int
            def __init__(self, a: T) -> None:
                self.__init(a)
                self.b = 1

            def __init(self, a: T) -> None:
                self.a = a

            def f(self) -> T:
                return self.a

            def g(self) -> str:
                return 'C'

        class D[T](C[T]):
            def __init__(self, a: T) -> None:
                self.__init(a)

            def __init(self, a: T) -> None:
                self._C__init(a) # type: ignore[attr-defined]

            def g(self) -> str:
                return 'D'
        class E(D[float]):
            a: float
            # def g(self) -> str:
            #     return 'E'

        assert E(1.).g() == 'D'

        if nb.IS_NUMBA and (getenv('NUMBA_DISABLE_JIT', 0) == 0):
            with pytest.raises(TypeError,
                            match = 'class members are not yet supported'):
                nb.jitclass(E)

        cE = nb.clean(E)

        assert cE(1.).g() == 'D'

        nb.jitclass(cE)(1.).f()
