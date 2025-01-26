from importlib import metadata

import limesqueezer as ls
# ======================================================================
def test_version():
    assert ls.__version__ == metadata.version(ls.__name__)
