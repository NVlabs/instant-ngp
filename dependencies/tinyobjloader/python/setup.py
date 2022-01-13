import setuptools
import platform

from distutils.command.build_ext import build_ext

with open("README.md", "r") as fh:
    long_description = fh.read()

# Adapted from https://github.com/pybind/python_example/blob/master/setup.py
class get_pybind_include(object):
    """Helper class to determine the pybind11 include path
    The purpose of this class is to postpone importing pybind11
    until it is actually installed, so that the ``get_include()``
    method can be invoked. """

    def __init__(self, user=False, pep517=False):
        self.user = user
        self.pep517 = pep517

    def __str__(self):
        import os
        import pybind11

        interpreter_include_path = pybind11.get_include(self.user)

        if self.pep517:
            # When pybind11 is installed permanently in site packages, the headers
            # will be in the interpreter include path above. PEP 517 provides an
            # experimental feature for build system dependencies. When installing
            # a package from a source distribvution, first its build dependencies
            # are installed in a temporary location. pybind11 does not return the
            # correct path for this condition, so we glom together a second path,
            # and ultimately specify them _both_ in the include search path.
            # https://github.com/pybind/pybind11/issues/1067
            return os.path.abspath(
                os.path.join(
                    os.path.dirname(pybind11.__file__),
                    "..",
                    "..",
                    "..",
                    "..",
                    "include",
                    os.path.basename(interpreter_include_path),
                )
            )
        else:
            return interpreter_include_path


# unix = default compiler name?
copt = {"unix": ["-std=c++11"], "gcc": ["-std=c++11"], "clang": ["std=c++11"]}
# TODO: set C++ version for msvc? {'msvc': ["/std:c++14"] }

# ext_compile_args = ["-std=c++11"]
# ext_link_args = []

# https://stackoverflow.com/questions/724664/python-distutils-how-to-get-a-compiler-that-is-going-to-be-used
class build_ext_subclass(build_ext):
    def build_extensions(self):
        c = self.compiler.compiler_type
        if c in copt:
            for e in self.extensions:
                e.extra_compile_args = copt[c]

        # if lopt.has_key(c):
        #    for e in self.extensions:
        #        e.extra_link_args = lopt[ c ]
        build_ext.build_extensions(self)


# Developer option
#
# if platform.system() == "Darwin":
#    # XCode10 or later does not support libstdc++, so we need to use libc++.
#    # macosx-version 10.6 does not support libc++, so we require min macosx version 10.9.
#    ext_compile_args.append("-stdlib=libc++")
#    ext_compile_args.append("-mmacosx-version-min=10.9")
#    ext_link_args.append("-stdlib=libc++")
#    ext_link_args.append("-mmacosx-version-min=10.9")

# `tiny_obj_loader.cc` contains implementation of tiny_obj_loader.
m = setuptools.Extension(
    "tinyobjloader",
    # extra_compile_args=ext_compile_args,
    # extra_link_args=ext_link_args,
    sources=["bindings.cc", "tiny_obj_loader.cc"],
    include_dirs=[
        # Support `build_ext` finding tinyobjloader (without first running
        # `sdist`).
        "..",
        # Support `build_ext` finding pybind 11 (provided it's permanently
        # installed).
        get_pybind_include(),
        get_pybind_include(user=True),
        # Support building from a source distribution finding pybind11 from
        # a PEP 517 temporary install.
        get_pybind_include(pep517=True),
    ],
    language="c++",
)


setuptools.setup(
    name="tinyobjloader",
    version="2.0.0rc9",
    description="Tiny but powerful Wavefront OBJ loader",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Syoyo Fujita",
    author_email="syoyo@lighttransport.com",
    url="https://github.com/tinyobjloader/tinyobjloader",
    project_urls={
        "Issue Tracker": "https://github.com/tinyobjloader/tinyobjloader/issues",
    },
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Manufacturing",
        "Topic :: Artistic Software",
        "Topic :: Multimedia :: Graphics :: 3D Modeling",
        "Topic :: Scientific/Engineering :: Visualization",
        "License :: OSI Approved :: MIT License",
        "License :: OSI Approved :: ISC License (ISCL)",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
    ],
    packages=setuptools.find_packages(),
    ext_modules=[m],
    cmdclass={"build_ext": build_ext_subclass},
)
