"""
pymatgen-io-validation provides validation tools for inputs and outputs of computational
simulations and calculations. That is, it checks calculation details against a reference
to ensure that data is compatible with some standard.
"""

from pymatgen.io.validation.validation import VaspValidator  # noqa: F401

from pymatgen.io.validation.settings import IOValidationSettings as _settings

if _settings().CHECK_PYPI_AT_LOAD:
    # Only check version at module load time, if specified in module settings.
    from pymatgen.io.validation.check_package_versions import package_version_check

    package_version_check()
