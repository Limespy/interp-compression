"""Entrypoint module, in case you use `python -m limesqueezer`"""
from .cli import main
raise SystemExit(main())
