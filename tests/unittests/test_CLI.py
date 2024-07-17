import pytest
from limesqueezer.cli_ import main

parametrize = pytest.mark.parametrize
@parametrize(('args',), (((),),
                         (('block',),),
                         (('stream',),),
                         (('both',),)))
def test_main(args: tuple[str, ...]):
    main(*args)
