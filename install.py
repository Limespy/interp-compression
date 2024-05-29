import os
import pathlib
import sys

optional = []
flags = set()
for arg in sys.argv[1:]:
    if arg.startswith('--'):
        flags.add(arg[2:])
    else:
        optional.append(arg)

os.system('git config pull.rebase true')
os.system(f'pip install -e .[{','.join(optional)}]')

if 'dev' in optional and not ('--minimal' in flags):
    os.system('pre-commit install')
    os.system(f'pre-commit run --files  {pathlib.Path(__file__)}')
