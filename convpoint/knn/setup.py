from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
import numpy
import sys

ext_modules = [Extension(
       "nearest_neighbors",
       sources=["knn.pyx", "knn_.cxx",],  # source file(s)
       include_dirs=["./", numpy.get_include()],
       language="c++",
       extra_compile_args = [ "-std=c++11"]+ (["/openmp"] if sys.platform.startswith("win") else ["-fopenmp"]),
       extra_link_args= [ "-std=c++11"]+ (["/openmp"] if sys.platform.startswith("win") else ["-fopenmp"]),
  )]

setup(
    name = "KNN NanoFLANN",
    ext_modules = ext_modules,
    cmdclass = {'build_ext': build_ext},
)
