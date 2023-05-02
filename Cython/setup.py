from distutils.core import setup, Extension
from Cython.Distutils import build_ext
import numpy as np
import os

os.environ["CC"] = 'mpiicc'
os.environ["LINKCC"] = 'mpiicc'
os.environ["LD"] = 'mpiicc'

mpi_compile_args = ['-O3']
mpi_link_args = ['-O3']

ext_modules = [
    Extension(
        "alltoall",sources=["alltoall.pyx"],language='c',
        include_dirs=[np.get_include()],
        libraries=["mpi"],
        library_dirs=[],
        extra_compile_args = mpi_compile_args,
        extra_link_args = mpi_link_args,
    )
]

setup(
    name = "alltoall",
    cmdclass = {"build_ext":build_ext},
    ext_modules = ext_modules
)
