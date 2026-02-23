from pathlib import Path

__version__ = "0.1.0"


def _jupyter_labextension_paths():
    """Tell JupyterLab where the prebuilt extension bundle lives."""
    return [
        {
            "src": str(Path(__file__).parent / "labextension"),
            "dest": "@pivotal/jupyterlab",
        }
    ]
