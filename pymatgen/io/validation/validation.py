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
    from pathlib import Path

DEFAULT_CHECKS = [CheckStructureProperties, CheckCommonErrors, CheckKpointsKspacing, CheckPotcar, CheckIncar]


class VaspValidator(BaseModel):

    reasons: list[str] = Field([], description="List of deprecation tags detailing why this task isn't valid")
    warnings: list[str] = Field([], description="List of warnings about this calculation")
    vasp_files: VaspFiles = Field(description="The VASP I/O.")

    @property
    def is_valid(self) -> bool:
        return len(self.reasons) > 0

    @classmethod
    def from_paths(
        cls,
        vasp_file_paths: dict[str, str | Path],
        fast: bool = False,
    ):

        config = {
            **{
                k: []
                for k in (
                    "reasons",
                    "warnings",
                )
            },
            "vasp_files": VaspFiles.from_paths(**vasp_file_paths),
        }

        for check in DEFAULT_CHECKS:
            check(fast=fast).check(config["vasp_files"], config["reasons"], config["warnings"])
            if fast and len(config["reasons"]) > 0:
                break
        return cls(**config)
