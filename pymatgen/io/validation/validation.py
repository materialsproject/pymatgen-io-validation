"""Define core validation schema."""

from __future__ import annotations
from pydantic import BaseModel, Field
from typing import TYPE_CHECKING

from pymatgen.io.validation.common import VaspFiles
from pymatgen.io.validation.check_common_errors import CheckStructureProperties, CheckCommonErrors
from pymatgen.io.validation.check_kpoints_kspacing import CheckKpointsKspacing
from pymatgen.io.validation.check_potcar import CheckPotcar
from pymatgen.io.validation.check_incar import CheckIncar

if TYPE_CHECKING:
    import os

DEFAULT_CHECKS = [CheckStructureProperties, CheckPotcar, CheckCommonErrors, CheckKpointsKspacing, CheckIncar]

# TODO: check for surface/slab calculations. Especially necessary for external calcs.
# TODO: implement check to make sure calcs are within some amount (e.g. 250 meV) of the convex hull in the MPDB


class VaspValidator(BaseModel):

    reasons: list[str] = Field([], description="List of deprecation tags detailing why this task isn't valid")
    warnings: list[str] = Field([], description="List of warnings about this calculation")
    vasp_files: VaspFiles = Field(description="The VASP I/O.")

    @property
    def is_valid(self) -> bool:
        return len(self.reasons) == 0

    @property
    def has_warnings(self) -> bool:
        return len(self.warnings) > 0

    @classmethod
    def from_vasp_input(
        cls,
        vasp_file_paths: dict[str, os.PathLike[str]] | None = None,
        vasp_files: VaspFiles | None = None,
        fast: bool = False,
        check_potcar: bool = True,
    ):

        if vasp_files:
            vf: VaspFiles = vasp_files
        elif vasp_file_paths:
            vf = VaspFiles.from_paths(**vasp_file_paths)

        config: dict[str, list[str]] = {
            "reasons": [],
            "warnings": [],
        }

        if check_potcar:
            checkers = DEFAULT_CHECKS
        else:
            checkers = [c for c in DEFAULT_CHECKS if c.__name__ != "CheckPotcar"]

        for check in checkers:
            check(fast=fast).check(vf, config["reasons"], config["warnings"])  # type: ignore[arg-type]
            if fast and len(config["reasons"]) > 0:
                break
        return cls(**config, vasp_files=vf)
