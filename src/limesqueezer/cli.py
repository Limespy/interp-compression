def main(args: list[str] | None = None):
    if args is None:
        from sys import argv
        args = argv[1:]
    if args and args[0] == '--version':
        from importlib import metadata
        print(__package__, metadata.version(__package__))
