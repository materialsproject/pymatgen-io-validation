"""Optionally check whether package versions are up to date."""

from __future__ import annotations
from importlib.metadata import version
import requests  # type: ignore[import-untyped]
import warnings


def package_version_check() -> None:
    """Warn the user if pymatgen / pymatgen-io-validation is not up-to-date."""

    packages = {
        "pymatgen": "Hence, if any pymatgen input sets have been updated, this validator will be outdated.",
        "pymatgen-io-validation": "Hence, if any checks in this package have been updated, the validator you use will be outdated.",
    }

    for package, context_msg in packages.items():
        if not is_package_is_up_to_date(package):
            warnings.warn(
                "We *STRONGLY* recommend you to update your "
                f"`{package}` package, which is behind the most "
                f"recent version. {context_msg}"
            )


def is_package_is_up_to_date(package_name: str) -> bool:
    """Check if a package is up-to-date with the PyPI version."""

    try:
        cur_version = version(package_name)
    except Exception:
        raise ImportError(f"Package `{package_name}` is not installed!")

    try:
        response = requests.get(f"https://pypi.org/pypi/{package_name}/json")
        latest_version = response.json()["info"]["version"]
    except Exception:
        raise ImportError(f"Package `{package_name}` does not exist in PyPI!")

    return cur_version == latest_version
