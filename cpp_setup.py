from setuptools import setup, Extension
import pybind11

ext_modules = [
    Extension(
        "cpptools",
        ["cpptools.cpp"],
        include_dirs=[pybind11.get_include()],
        language="c++",
        extra_compile_args=["/std:c++17", "/O2"],
        extra_link_args=["/MACHINE:X64"],
    )
]

setup(
    name="cpptools",
    version="0.0.1",
    ext_modules=ext_modules,
)