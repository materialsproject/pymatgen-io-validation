# Copyright (c) Materials Virtual Lab
# Distributed under the terms of the Modified BSD License.

from setuptools import setup, find_namespace_packages

import os

SETUP_PTH = os.path.dirname(os.path.abspath(__file__))

with open(os.path.join(SETUP_PTH, "README.md")) as f:
    desc = f.read()


setup(
    name="pymatgen-io-validation",
    packages=find_namespace_packages(include=["pymatgen.io.*"]),
    version="0.0.3",
    install_requires=["pymatgen>=2024.5.1", "emmet-core>=0.83.6", "pydantic>=2.4.2", "requests>=2.28.1"],
    extras_require={},
    package_data={"pymatgen.io.validation": ["*.yaml"]},
    author="Matthew Kuner, Janosh Riebesell, Jason Munro, Aaron Kaplan",
    author_email="matthewkuner@berkeley.edu",
    maintainer="Matthew Kuner",
    url="https://github.com/materialsproject/pymatgen-io-validation",
    license="BSD",
    description="A comprehensive I/O validator for electronic structure calculations",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    keywords=["pymatgen"],
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: BSD License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Information Analysis",
        "Topic :: Scientific/Engineering :: Physics",
        "Topic :: Scientific/Engineering :: Chemistry",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
)
