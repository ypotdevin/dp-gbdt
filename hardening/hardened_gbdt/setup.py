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
            "./src/estimator.cpp",
            "./src/parameters.cpp",
            "./src/constant_time.cpp",
            "./src/cli_parser.cpp",
            "./src/gbdt/tree_rejection.cpp",
            "./src/gbdt/loss.cpp",
            "./src/gbdt/utils.cpp",
            "./src/gbdt/custom_cauchy.cpp",
            "./src/gbdt/dp_ensemble.cpp",
            "./src/gbdt/dp_tree.cpp",
            "./src/gbdt/data.cpp",
        ],
        include_dirs=[
            "./include/",
            "./include/gbdt/",
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
