from setuptools import setup
from Cython.Build import cythonize
# import numpy

setup(
    ext_modules=cythonize("reinforcement_learning/wrapper_helper.pyx"),
    # include_dirs=[numpy.get_include()],
)
