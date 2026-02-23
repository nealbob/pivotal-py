from .dsl_parser import DSLParser

def load_ipython_extension(ipython):
    from .magic import load_ipython_extension as magic_load
    magic_load(ipython)

__all__ = ['DSLParser', 'load_ipython_extension']
