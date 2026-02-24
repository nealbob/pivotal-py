from .dsl_parser import DSLParser

def load_ipython_extension(ipython):
    from .magic import load_ipython_extension as magic_load
    magic_load(ipython)

# Auto-register the IPython magic when imported inside a Jupyter/IPython session.
# This means %%pivotal works immediately without requiring %load_ext pivotal.
try:
    ip = get_ipython()  # type: ignore[name-defined]
    if ip is not None:
        from .magic import PivotalMagics
        ip.register_magics(PivotalMagics)
except NameError:
    pass  # Not running inside IPython

__all__ = ['DSLParser', 'load_ipython_extension']
