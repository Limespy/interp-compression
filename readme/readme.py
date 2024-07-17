import limesqueezer
from limedev import readme
#=======================================================================
def main(pyproject: readme.PyprojectType):
    """This gets called by the limedev."""
    name = pyproject['tool']['limedev']['full_name']

    semi_description = f'''
        {name} is a toolkit where NumPy array are lossily compressed using
        spline fitting.'''
    return readme.make(limesqueezer, semi_description,
                       name = name,
                       abbreviation = 'ls')
#=======================================================================
