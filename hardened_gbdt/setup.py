from distutils.core import Extension, setup

import numpy
from Cython import Compiler
from Cython.Build import cythonize

Compiler.Options.fast_fail = True

extensions = [
    Extension(
        "pyestimator",
        sources=[
            "py_estimator.pyx",
            "./src/constant_time.cpp",
            "./src/data.cpp",
            "./src/estimator.cpp",
            "./src/parameters.cpp",
            "./src/gbdt/custom_cauchy.cpp",
            "./src/gbdt/dp_ensemble.cpp",
            "./src/gbdt/dp_tree.cpp",
            "./src/gbdt/loss.cpp",
            "./src/gbdt/tree_rejection.cpp",
            "./src/gbdt/utils.cpp",
        ],
        include_dirs=[
            "./include/",
            numpy.get_include(),
        ],
        define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")],
    )
]

setup(
    ext_modules=cythonize(
        extensions,
        language_level=3,
    ),
)

# to build the extension, run `python setup.py build_ext --inplace`.
