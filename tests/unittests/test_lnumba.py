import pytest
from limesqueezer import _lnumba as nb
# ======================================================================
parametrize = pytest.mark.parametrize
# ======================================================================
class Test_jitclass:
    def test_remove_Generic(self):

        class C[T]:
            a: T
            b: int
            def __init__(self, a: T) -> None:
                self.__init(a)
                self.b = 1.

            def __init(self, a: T) -> None:
                self.a = a

            def f(self) -> T:
                return self.a

        class D[T](C[T]):
            def __init__(self, a: T) -> None:
                self.__init(a)

            def __init(self, a: T) -> None:
                self._C__init(a)
        @nb.jitclass
        class E(D[float]):
            a: float

        E(1.).f()
