from .dsl_parser import DSLParser
from .package import Package
from .magic import update, delete

def load_ipython_extension(ipython):
    from .magic import load_ipython_extension as magic_load
    magic_load(ipython)

# Auto-register the IPython magic when imported inside a Jupyter/IPython session.
# This means %%pivotal works immediately without requiring %load_ext pivotal.
try:
    ip = get_ipython()  # type: ignore[name-defined]
    if ip is not None:
        from .magic import PivotalMagics, _active_magics as _am
        import pivotal.magic as _magic_mod
        if _am is None:
            m = PivotalMagics(ip)
            _magic_mod._active_magics = m
            ip.register_magics(m)
except NameError:
    pass  # Not running inside IPython

__all__ = ['DSLParser', 'Package', 'load_ipython_extension', 'update', 'delete']
