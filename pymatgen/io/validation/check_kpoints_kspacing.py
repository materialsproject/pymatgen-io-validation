"""Validate VASP KPOINTS files or the KSPACING/KGAMMA INCAR settings."""

from __future__ import annotations
from pydantic import Field
from typing import TYPE_CHECKING
import numpy as np

from pymatgen.io.validation.common import BaseValidator
from pymatgen.io.validation.settings import IOValidationSettings

SETTINGS = IOValidationSettings()

if TYPE_CHECKING:
    from pymatgen.io.validation.common import VaspFiles


class CheckKpointsKspacing(BaseValidator):
    """Check that k-point density is sufficiently high and is compatible with lattice symmetry."""

    name: str = "Check k-point density"
    kpts_tolerance: float | None = Field(
        SETTINGS.VASP_KPTS_TOLERANCE,
        description="Tolerance for evaluating k-point density, to accommodate different the k-point generation schemes across VASP versions.",
    )
    allow_explicit_kpoint_mesh: bool = Field(
        SETTINGS.VASP_ALLOW_EXPLICIT_KPT_MESH,
        description="Whether to permit explicit generation of k-points (as for a bandstructure calculation).",
    )
    allow_kpoint_shifts: bool = Field(
        SETTINGS.VASP_ALLOW_KPT_SHIFT,
        description="Whether to permit shifting the origin of the k-point mesh from Gamma.",
    )

    def auto_fail(self, vasp_files: VaspFiles, reasons: list[str], warnings: list[str]) -> bool:
        """Quick stop if actual k-points are missing."""
        if vasp_files.actual_kpoints is None:
            reasons.append("Missing actual k-points: please specify an IBZKPT or vasprun.xml in VaspFiles.")
        return vasp_files.actual_kpoints is None

    def _get_valid_num_kpts(
        self,
        vasp_files: VaspFiles,
    ) -> int:
        """
        Get the minimum permitted number of k-points for a structure according to an input set.

        Returns
        -----------
        int, the minimum permitted number of k-points, consistent with self.kpts_tolerance
        """
        # If MP input set specifies KSPACING in the INCAR
        if ("KSPACING" in vasp_files.valid_input_set.incar.keys()) and (vasp_files.valid_input_set.kpoints is None):
            valid_kspacing = vasp_files.valid_input_set.incar.get("KSPACING", self.vasp_defaults["KSPACING"].value)
            # number of kpoints along each of the three lattice vectors
            nk = [
                max(1, np.ceil(vasp_files.user_input.structure.lattice.reciprocal_lattice.abc[ik] / valid_kspacing))
                for ik in range(3)
            ]
            valid_num_kpts = np.prod(nk)
        # If MP input set specifies a KPOINTS file
        else:
            valid_num_kpts = vasp_files.valid_input_set.kpoints.num_kpts or np.prod(
                vasp_files.valid_input_set.kpoints.kpts[0]
            )

        return int(np.floor(int(valid_num_kpts) * self.kpts_tolerance))

    def _check_user_shifted_mesh(self, vasp_files: VaspFiles, reasons: list[str], warnings: list[str]) -> None:
        # Check for user shifts
        if (not self.allow_kpoint_shifts) and any(shift_val != 0 for shift_val in vasp_files.actual_kpoints.kpts_shift):
            reasons.append("INPUT SETTINGS --> KPOINTS: shifting the kpoint mesh is not currently allowed.")

    def _check_explicit_mesh_permitted(self, vasp_files: VaspFiles, reasons: list[str], warnings: list[str]) -> None:
        # Check for explicit kpoint meshes

        if (not self.allow_explicit_kpoint_mesh) and len(vasp_files.actual_kpoints.kpts) > 1:
            reasons.append(
                "INPUT SETTINGS --> KPOINTS: explicitly defining "
                "the k-point mesh is not currently allowed. "
                "Automatic k-point generation is required."
            )

    def _check_kpoint_density(self, vasp_files: VaspFiles, reasons: list[str], warnings: list[str]) -> None:
        """
        Check that k-point density is sufficiently high and is compatible with lattice symmetry.
        """

        # Check number of kpoints used
        valid_num_kpts = self._get_valid_num_kpts(vasp_files)

        cur_num_kpts = max(
            vasp_files.actual_kpoints.num_kpts,
            np.prod(vasp_files.actual_kpoints.kpts),
            len(vasp_files.actual_kpoints.kpts),
        )
        if cur_num_kpts < valid_num_kpts:
            reasons.append(
                f"INPUT SETTINGS --> KPOINTS or KSPACING: {cur_num_kpts} kpoints were "
                f"used, but it should have been at least {valid_num_kpts}."
            )

    def _check_kpoint_mesh_symmetry(self, vasp_files: VaspFiles, reasons: list[str], warnings: list[str]) -> None:
        # check for valid kpoint mesh (which depends on symmetry of the structure)

        cur_kpoint_style = vasp_files.actual_kpoints.style.name.lower()
        is_hexagonal = vasp_files.user_input.structure.lattice.is_hexagonal()
        is_face_centered = vasp_files.user_input.structure.get_space_group_info()[0][0] == "F"
        monkhorst_mesh_is_invalid = is_hexagonal or is_face_centered
        if (
            cur_kpoint_style == "monkhorst"
            and monkhorst_mesh_is_invalid
            and any(x % 2 == 0 for x in vasp_files.actual_kpoints.kpts[0])
        ):
            # only allow Monkhorst with all odd number of subdivisions per axis.
            kx, ky, kz = vasp_files.actual_kpoints.kpts[0]
            reasons.append(
                f"INPUT SETTINGS --> KPOINTS or KGAMMA: ({kx}x{ky}x{kz}) "
                "Monkhorst-Pack kpoint mesh was used."
                "To be compatible with the symmetry of the lattice, "
                "a Monkhorst-Pack mesh should have only odd number of "
                "subdivisions per axis."
            )
