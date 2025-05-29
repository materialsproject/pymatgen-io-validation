# mypy: ignore-errors

"""
Settings for pymatgen-io-validation. Used to be part of EmmetSettings.
"""

import json
from pathlib import Path
from typing import Dict, Type, TypeVar, Union

import requests
from monty.json import MontyDecoder
from pydantic import field_validator, model_validator, Field, ImportString
from pydantic_settings import BaseSettings, SettingsConfigDict

DEFAULT_CONFIG_FILE_PATH = str(Path.home().joinpath(".emmet.json"))


S = TypeVar("S", bound="IOValidationSettings")


class IOValidationSettings(BaseSettings):
    """
    Settings for pymatgen-io-validation
    """

    config_file: str = Field(DEFAULT_CONFIG_FILE_PATH, description="File to load alternative defaults from")

    CHECK_PYPI_AT_LOAD: bool = Field(
        False,
        description=(
            "Whether to do a version check when this module is loaded. "
            "Helps user ensure most recent parameter checks are used."
        ),
    )

    VASP_KPTS_TOLERANCE: float = Field(
        0.9,
        description="Relative tolerance for kpt density to still be a valid task document",
    )

    VASP_ALLOW_KPT_SHIFT: bool = Field(
        False,
        description="Whether to consider a task valid if kpoints are shifted by the user",
    )

    VASP_ALLOW_EXPLICIT_KPT_MESH: Union[str, bool] = Field(
        "auto",
        description="Whether to consider a task valid if the user defines an explicit kpoint mesh",
    )

    VASP_FFT_GRID_TOLERANCE: float = Field(
        0.9,
        description="Relative tolerance for FFT grid parameters to still be a valid",
    )

    VASP_DEFAULT_INPUT_SETS: Dict[str, ImportString] = Field(
        {
            "GGA Structure Optimization": "pymatgen.io.vasp.sets.MPRelaxSet",
            "GGA+U Structure Optimization": "pymatgen.io.vasp.sets.MPRelaxSet",
            "r2SCAN Structure Optimization": "pymatgen.io.vasp.sets.MPScanRelaxSet",
            "SCAN Structure Optimization": "pymatgen.io.vasp.sets.MPScanRelaxSet",
            "PBESol Structure Optimization": "pymatgen.io.vasp.sets.MPScanRelaxSet",
            "GGA Static": "pymatgen.io.vasp.sets.MPStaticSet",
            "GGA+U Static": "pymatgen.io.vasp.sets.MPStaticSet",
            "PBE Static": "pymatgen.io.vasp.sets.MPStaticSet",
            "PBE+U Static": "pymatgen.io.vasp.sets.MPStaticSet",
            "r2SCAN Static": "pymatgen.io.vasp.sets.MPScanStaticSet",
            "SCAN Static": "pymatgen.io.vasp.sets.MPScanStaticSet",
            "PBESol Static": "pymatgen.io.vasp.sets.MPScanStaticSet",
            "HSE06 Static": "pymatgen.io.vasp.sets.MPScanStaticSet",
            "GGA NSCF Uniform": "pymatgen.io.vasp.sets.MPNonSCFSet",
            "GGA+U NSCF Uniform": "pymatgen.io.vasp.sets.MPNonSCFSet",
            "GGA NSCF Line": "pymatgen.io.vasp.sets.MPNonSCFSet",
            "GGA+U NSCF Line": "pymatgen.io.vasp.sets.MPNonSCFSet",
            "GGA NMR Electric Field Gradient": "pymatgen.io.vasp.sets.MPNMRSet",
            "GGA NMR Nuclear Shielding": "pymatgen.io.vasp.sets.MPNMRSet",
            "GGA+U NMR Electric Field Gradient": "pymatgen.io.vasp.sets.MPNMRSet",
            "GGA+U NMR Nuclear Shielding": "pymatgen.io.vasp.sets.MPNMRSet",
            "GGA Deformation": "pymatgen.io.vasp.sets.MPStaticSet",
            "GGA+U Deformation": "pymatgen.io.vasp.sets.MPStaticSet",
            "GGA DFPT Dielectric": "pymatgen.io.vasp.sets.MPStaticSet",
            "GGA+U DFPT Dielectric": "pymatgen.io.vasp.sets.MPStaticSet",
        },
        description="Default input sets for task validation",
    )

    VASP_MAX_SCF_GRADIENT: float = Field(
        1000,
        description="Maximum upward gradient in the last SCF for any VASP calculation",
    )

    VASP_NUM_IONIC_STEPS_FOR_DRIFT: int = Field(
        3,
        description="Number of ionic steps to average over when validating drift forces",
    )

    VASP_MAX_POSITIVE_ENERGY: float = Field(
        50.0, description="Maximum allowable positive energy at the end of a calculation."
    )

    model_config = SettingsConfigDict(env_prefix="pymatgen_io_validation_", extra="ignore")

    FAST_VALIDATION: bool = Field(
        default=False,
        description=(
            "Whether to attempt to find all reasons a calculation fails (False), "
            "or stop validation if any single check fails."
        ),
    )

    @model_validator(mode="before")
    @classmethod
    def load_default_settings(cls, values):
        """
        Loads settings from a root file if available and uses that as defaults in
        place of built in defaults
        """
        config_file_path: str = values.get("config_file", DEFAULT_CONFIG_FILE_PATH)

        new_values = {}

        if config_file_path.startswith("http"):
            new_values = requests.get(config_file_path).json()
        elif Path(config_file_path).exists():
            with open(config_file_path, encoding="utf8") as f:
                new_values = json.load(f)

        new_values.update(values)

        return new_values

    @classmethod
    def autoload(cls: Type[S], settings: Union[None, dict, S]) -> S:  # noqa
        if settings is None:
            return cls()
        elif isinstance(settings, dict):
            return cls(**settings)
        return settings

    @field_validator("VASP_DEFAULT_INPUT_SETS", mode="before")
    @classmethod
    def convert_input_sets(cls, value):  # noqa
        if isinstance(value, dict):
            return {k: MontyDecoder().process_decoded(v) for k, v in value.items()}
        return value

    def as_dict(self):
        """
        HotPatch to enable serializing IOValidationSettings via Monty
        """
        return self.model_dump(exclude_unset=True, exclude_defaults=True)
