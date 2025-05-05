"""Define core validation schema."""

from __future__ import annotations
from pathlib import Path
from pydantic import BaseModel, Field
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

# TODO: check for surface/slab calculations. Especially necessary for external calcs.
# TODO: implement check to make sure calcs are within some amount (e.g. 250 meV) of the convex hull in the MPDB


class VaspValidator(BaseModel):
    """Validate a VASP calculation."""

    reasons: list[str] = Field([], description="List of deprecation tags detailing why this task isn't valid")
    warnings: list[str] = Field([], description="List of warnings about this calculation")
    vasp_files: VaspFiles = Field(description="The VASP I/O.")

    @property
    def is_valid(self) -> bool:
        """Determine if the calculation is valid."""
        return len(self.reasons) == 0

    @property
    def has_warnings(self) -> bool:
        """Determine if any warnings were incurred."""
        return len(self.warnings) > 0

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
        return cls(**config, vasp_files=vf, **kwargs)

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
        for file_name in ("INCAR", "KPOINTS", "POSCAR", "POTCAR", "OUTCAR", "vasprun.xml"):
            if (file_path := Path(zpath(str(dir_name / file_name)))).exists():
                vasp_file_paths[file_name.lower().split(".")[0]] = file_path
        return cls.from_vasp_input(vasp_file_paths=vasp_file_paths, **kwargs)
