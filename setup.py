"""Build the Cython scoring extension. Gracefully no-ops if Cython/numpy aren't installed."""
from pathlib import Path

from setuptools import setup

ext_modules = []
pyx_path = Path("src/splade_easy/_scoring.pyx")
if pyx_path.exists():
    try:
        import numpy as np
        from Cython.Build import cythonize

        ext_modules = cythonize(
            [str(pyx_path)],
            language_level=3,
            compiler_directives={
                "boundscheck": False,
                "wraparound": False,
                "cdivision": True,
                "initializedcheck": False,
            },
        )
        for ext in ext_modules:
            ext.include_dirs = [np.get_include()]
    except ImportError:
        # Cython/numpy not available at build time — fall through with no extensions
        pass

setup(ext_modules=ext_modules)
