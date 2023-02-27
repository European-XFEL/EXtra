
"""Import proxy machinery.

This allows a module to proxy any import symbols from another module
not limited to its __all__ symbols by generating a __getattr__ handler.
"""

from importlib import import_module


def generate_getattr(qual_name, proxied_package):
    """Generate proxying getattr handler.

    Args:
        qual_name (str): Qualfied name of the module using this
            function, i.e. its __name__ symbol.
        proxied_package: Package to import the module from instead.

    Returns:
        (Callable) Function to use as module-level __gettattr__ handler.
    """

    # Take the last part of the qualified name as module to import from
    # the proxied package.
    module_name = qual_name[qual_name.rfind('.')+1:]

    def getattr_(name):
        if not name.startswith('__'):
            mod = import_module(f'{proxied_package}.{module_name}')
            return mod.__dict__[name]

    return getattr_
