"""Validate VASP KPOINTS files or the KSPACING/KGAMMA INCAR settings."""

from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from pymatgen.io.vasp import Kpoints

from pymatgen.io.validation.common import BaseValidator

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pymatgen.core import Structure
    from pymatgen.io.vasp.sets import VaspInputSet


@dataclass
class CheckKpointsKspacing(BaseValidator):
    """
    Check that k-point density is sufficiently high and is compatible with lattice symmetry.

    Parameters
    -----------
    reasons : list[str]
        A list of error strings to update if a check fails. These are higher
        severity and would deprecate a calculation.
    warnings : list[str]
        A list of warning strings to update if a check fails. These are lower
        severity and would flag a calculation for possible review.
    valid_input_set: VaspInputSet
        Valid input set to compare user INCAR parameters to.
    kpoints : Kpoints or dict
        Kpoints object or its .as_dict() representation used in the calculation.
    structure : pymatgen.core.Structure
        The structure used in the calculation
    name : str = "Check k-point density"
        Name of the validator class
    fast : bool = False
        Whether to perform quick check.
        True: stop validation if any check fails.
        False: perform all checks.
    defaults : dict
        Dict of default parameters
    kpts_tolerance : float
        Tolerance for evaluating k-point density, as the k-point generation
        scheme is inconsistent across VASP versions
    allow_explicit_kpoint_mesh : str | bool
        Whether to permit explicit generation of k-points (as for a bandstructure calculation).
    allow_kpoint_shifts : bool
        Whether to permit shifting the origin of the k-point mesh from Gamma.
    """

    reasons: list[str]
    warnings: list[str]
    name: str = "Check k-point density"
    valid_input_set: VaspInputSet = None
    kpoints: Kpoints | dict = None
    structure: Structure = None
    defaults: dict | None = None
    kpts_tolerance: float | None = None
    allow_explicit_kpoint_mesh: str | bool = False
    allow_kpoint_shifts: bool = False

    def _get_valid_num_kpts(self) -> int:
        """
        Get the minimum permitted number of k-points for a structure according to an input set.

        Returns
        -----------
        int, the minimum permitted number of k-points, consistent with self.kpts_tolerance
        """
        # If MP input set specifies KSPACING in the INCAR
        if ("KSPACING" in self.valid_input_set.incar.keys()) and (self.valid_input_set.kpoints is None):
            valid_kspacing = self.valid_input_set.incar.get("KSPACING", self.defaults["KSPACING"]["value"])
            latt_cur_anorm = self.structure.lattice.abc
            # number of kpoints along each of the three lattice vectors
            nk = [max(1, np.ceil(2 * np.pi / (valid_kspacing * latt_cur_anorm[ik]))) for ik in range(3)]
            valid_num_kpts = np.prod(nk)
        # If MP input set specifies a KPOINTS file
        else:
            valid_num_kpts = self.valid_input_set.kpoints.num_kpts or np.prod(self.valid_input_set.kpoints.kpts[0])

        return int(np.floor(int(valid_num_kpts) * self.kpts_tolerance))

    def _check_user_shifted_mesh(self) -> None:
        # Check for user shifts
        if (not self.allow_kpoint_shifts) and any(shift_val != 0 for shift_val in self.kpoints["usershift"]):
            self.reasons.append("INPUT SETTINGS --> KPOINTS: shifting the kpoint mesh is not currently allowed.")

    def _check_explicit_mesh_permitted(self) -> None:
        # Check for explicit kpoint meshes

        if (not self.allow_explicit_kpoint_mesh) and len(self.kpoints["kpoints"]) > 1:
            self.reasons.append(
                "INPUT SETTINGS --> KPOINTS: explicitly defining "
                "the k-point mesh is not currently allowed. "
                "Automatic k-point generation is required."
            )

    def _check_kpoint_density(self) -> None:
        """
        Check that k-point density is sufficiently high and is compatible with lattice symmetry.
        """

        # Check number of kpoints used
        valid_num_kpts = self._get_valid_num_kpts()

        if isinstance(self.kpoints, Kpoints):
            self.kpoints = self.kpoints.as_dict()

        cur_num_kpts = max(
            self.kpoints.get("nkpoints", 0), np.prod(self.kpoints.get("kpoints")), len(self.kpoints.get("kpoints"))
        )
        if cur_num_kpts < valid_num_kpts:
            self.reasons.append(
                f"INPUT SETTINGS --> KPOINTS or KSPACING: {cur_num_kpts} kpoints were "
                f"used, but it should have been at least {valid_num_kpts}."
            )

    def _check_kpoint_mesh_symmetry(self) -> None:
        # check for valid kpoint mesh (which depends on symmetry of the structure)

        cur_kpoint_style = self.kpoints.get("generation_style").lower()
        is_hexagonal = self.structure.lattice.is_hexagonal()
        is_face_centered = self.structure.get_space_group_info()[0][0] == "F"
        monkhorst_mesh_is_invalid = is_hexagonal or is_face_centered
        if (
            cur_kpoint_style == "monkhorst"
            and monkhorst_mesh_is_invalid
            and any(x % 2 == 0 for x in self.kpoints.get("kpoints")[0])
        ):
            # only allow Monkhorst with all odd number of subdivisions per axis.
            kx, ky, kz = self.kpoints.get("kpoints")[0]
            self.reasons.append(
                f"INPUT SETTINGS --> KPOINTS or KGAMMA: ({kx}x{ky}x{kz}) "
                "Monkhorst-Pack kpoint mesh was used."
                "To be compatible with the symmetry of the lattice, "
                "a Monkhorst-Pack mesh should have only odd number of "
                "subdivisions per axis."
            )
