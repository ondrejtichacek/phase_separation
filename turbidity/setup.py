from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize
import numpy

print(numpy.get_include())

ext_modules = [
    Extension(
        "bp",
        ["bp.pyx"],
        include_dirs = [numpy.get_include()],
        extra_compile_args=['-O3', '-march=native', '-ffast-math', '-fopenmp', '-lgomp'], #, '-mavx512f', '-mavx512cd'],
        # extra_link_args=['-fopenmp', '-lgomp'],
    ),
    # Extension(
    #     "example",
    #     ["example.pyx"],
    #     extra_compile_args=['-O3', '-march=native', '-ffast-math', '-fopenmp', '-lgomp'], #, '-mavx512f', '-mavx512cd'],
    #     extra_link_args=['-fopenmp', '-lgomp'],
    # ),
]

setup(
    name='bp',
    version='1.0',
    ext_modules=cythonize(ext_modules, build_dir="temp"),
    include_dirs = [numpy.get_include()], #Include directory not hard-wired    
)

# from setuptools import setup, find_packages

# setup(name='ondra', version='1.0', packages=find_packages())

# pip install -e .