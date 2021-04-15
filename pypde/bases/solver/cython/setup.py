from distutils.core import Extension, setup
from Cython.Build import cythonize

# define an extension that will be cythonized and compiled
ext = Extension(name="tdma_c", sources=["tdma_c.pyx"])
setup(ext_modules=cythonize(ext))

ext = Extension(name="utda_c", sources=["utda_c.pyx"])
setup(ext_modules=cythonize(ext))