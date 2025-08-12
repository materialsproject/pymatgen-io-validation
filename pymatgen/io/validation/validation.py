"""Define core validation schema."""

from __future__ import annotations

from pathlib import Path
from pydantic import BaseModel, Field, PrivateAttr, computed_field
from typing import TYPE_CHECKING

from monty.os.path import zpath

from pymatgen.io.validation.common import VaspFiles
from pymatgen.io.validation.check_common_errors import CheckStructureProperties, CheckCommonErrors
from pymatgen.io.validation.check_kpoints_kspacing import CheckKpointsKspacing
from pymatgen.io.validation.check_potcar import CheckPotcar
from pymatgen.io.validation.check_incar import CheckIncar

if TYPE_CHECKING:
    from collections.abc import Mapping
    import os
    from typing_extensions import Self


DEFAULT_CHECKS = [CheckStructureProperties, CheckPotcar, CheckCommonErrors, CheckKpointsKspacing, CheckIncar]
REQUIRED_VASP_FILES: set[str] = {"INCAR", "KPOINTS", "POSCAR", "POTCAR", "OUTCAR", "vasprun.xml"}

# TODO: check for surface/slab calculations. Especially necessary for external calcs.
# TODO: implement check to make sure calcs are within some amount (e.g. 250 meV) of the convex hull in the MPDB


class VaspValidator(BaseModel):
    """Validate a VASP calculation."""

    vasp_files: VaspFiles = Field(description="The VASP I/O.")
    reasons: list[str] = Field([], description="List of deprecation tags detailing why this task isn't valid")
    warnings: list[str] = Field([], description="List of warnings about this calculation")

    _validated_md5: str | None = PrivateAttr(None)

    @computed_field  # type: ignore[misc]
    @property
    def valid(self) -> bool:
        """Determine if the calculation is valid after ensuring inputs have not changed."""
        self.recheck()
        return len(self.reasons) == 0

    @property
    def has_warnings(self) -> bool:
        """Determine if any warnings were incurred."""
        return len(self.warnings) > 0

    def recheck(self) -> None:
        """Rerun validation, prioritizing speed."""
        new_md5 = None
        if (self._validated_md5 is None) or (new_md5 := self.vasp_files.md5) != self._validated_md5:
            self.reasons = []
            self.warnings = []

            if self.vasp_files.user_input.potcar:
                check_list = DEFAULT_CHECKS
            else:
                check_list = [c for c in DEFAULT_CHECKS if c.__name__ != "CheckPotcar"]
            self.reasons, self.warnings = self.run_checks(self.vasp_files, check_list=check_list, fast=True)

            self._validated_md5 = new_md5 or self.vasp_files.md5

    @staticmethod
    def run_checks(
        vasp_files: VaspFiles,
        check_list: list | tuple = DEFAULT_CHECKS,
        fast: bool = False,
    ) -> tuple[list[str], list[str]]:
        """Perform validation.

        Parameters
        -----------
        vasp_files : VaspFiles
            The VASP I/O to validate.
        check_list : list or tuple of BaseValidator.
            The list of checks to perform. Defaults to `DEFAULT_CHECKS`.
        fast : bool (default = False)
            Whether to stop validation at the first validation failure (True)
            or compile a list of all failure reasons.

        Returns
        -----------
        tuple of list of str
            The first list are all reasons for validation failure,
            the second list contains all warnings.
        """

        if vasp_files.validation_errors:
            # Cannot validate the calculation, immediate failure
            return vasp_files.validation_errors, []

        reasons: list[str] = []
        warnings: list[str] = []
        for check in check_list:
            check(fast=fast).check(vasp_files, reasons, warnings)  # type: ignore[arg-type]
            if fast and len(reasons) > 0:
                break
        return reasons, warnings

    @classmethod
    def from_vasp_input(
        cls,
        vasp_file_paths: Mapping[str, str | Path | os.PathLike[str]] | None = None,
        vasp_files: VaspFiles | None = None,
        fast: bool = False,
        check_potcar: bool = True,
        **kwargs,
    ) -> Self:
        """
        Validate a VASP calculation from VASP files or their object representation.

        Parameters
        -----------
        vasp_file_paths : dict of str to os.PathLike, optional
            If specified, a dict of the form:
                {
                    "incar": < path to INCAR>,
                    "poscar": < path to POSCAR>,
                    ...
                }
            where keys are taken by `VaspFiles.from_paths`.
        vasp_files : VaspFiles, optional
            This takes higher precendence than `vasp_file_paths`, and
            allows the user to specify VASP input/output from a VaspFiles
            object.
        fast : bool (default = False)
            Whether to stop validation at the first failure (True)
            or to list all reasons why a calculation failed (False)
        check_potcar : bool (default = True)
            Whether to check the POTCAR for validity.
        **kwargs
            kwargs to pass to `VaspValidator`
        """

        if vasp_files:
            vf: VaspFiles = vasp_files
        elif vasp_file_paths:
            vf = VaspFiles.from_paths(**vasp_file_paths)
        else:
            raise ValueError("You must specify either a VaspFiles object or a dict of paths.")

        config: dict[str, list[str]] = {
            "reasons": [],
            "warnings": [],
        }

        if check_potcar:
            check_list = DEFAULT_CHECKS
        else:
            check_list = [c for c in DEFAULT_CHECKS if c.__name__ != "CheckPotcar"]

        config["reasons"], config["warnings"] = cls.run_checks(vf, check_list=check_list, fast=fast)
        validated = cls(**config, vasp_files=vf, **kwargs)
        validated._validated_md5 = vf.md5
        return validated

    @classmethod
    def from_directory(cls, dir_name: str | Path, **kwargs) -> Self:
        """Convenience method to validate a calculation from a directory.

        This method is intended solely for use cases where VASP input/output
        files are not renamed, beyond the compression methods supported by
        monty.os.zpath.

        Thus, INCAR, INCAR.gz, INCAR.bz2, INCAR.lzma are all acceptable, but
        INCAR.relax1.gz is not.

        For finer-grained control of which files are validated, explicitly
        pass file names to `VaspValidator.from_vasp_input`.

        Parameters
        -----------
        dir_name : str or Path
            The path to the calculation directory.
        **kwargs
            kwargs to pass to `VaspValidator`
        """
        dir_name = Path(dir_name)
        vasp_file_paths = {}
        for file_name in REQUIRED_VASP_FILES:
            if (file_path := Path(zpath(str(dir_name / file_name)))).exists():
                vasp_file_paths[file_name.lower().split(".")[0]] = file_path
        if not vasp_file_paths:
            raise ValueError("No valid VASP files were found.")
        return cls.from_vasp_input(vasp_file_paths=vasp_file_paths, **kwargs)
