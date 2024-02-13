"""Validate VASP KPOINTS files or the KSPACING/KGAMMA INCAR settings."""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from pymatgen.io.vasp import Kpoints

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pymatgen.core import Structure
    from pymatgen.io.vasp.sets import VaspInputSet


@dataclass
class CheckKpointsKspacing:
    """
    Check that k-point density is sufficiently high and is compatible with lattice symmetry.

    Parameters
    -----------
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

    defaults: dict | None = None
    kpts_tolerance: float | None = None
    allow_explicit_kpoint_mesh: str | bool = False
    allow_kpoint_shifts: bool = False

    def _get_valid_num_kpts(self, valid_input_set: VaspInputSet, structure: Structure) -> int:
        """
        Get the minimum permitted number of k-points for a structure according to an input set.

        Parameters
        -----------
        valid_input_set: VaspInputSet
            Valid input set to compare user INCAR parameters to.
        structure : pymatgen.core.Structure
            The structure used in the calculation

        Returns
        -----------
        int, the minimum permitted number of k-points.
        """
        # If MP input set specifies KSPACING in the INCAR
        if ("KSPACING" in valid_input_set.incar.keys()) and (valid_input_set.kpoints is None):
            valid_kspacing = valid_input_set.incar.get("KSPACING", self.defaults["KSPACING"]["value"])
            latt_cur_anorm = structure.lattice.abc
            # number of kpoints along each of the three lattice vectors
            nk = [max(1, np.ceil(2 * np.pi / (valid_kspacing * latt_cur_anorm[ik]))) for ik in range(3)]
            valid_num_kpts = np.prod(nk)
        # If MP input set specifies a KPOINTS file
        else:
            valid_num_kpts = valid_input_set.kpoints.num_kpts or np.prod(valid_input_set.kpoints.kpts[0])

        return int(valid_num_kpts)

    def check(
        self, reasons: list[str], valid_input_set: VaspInputSet, kpoints: Kpoints | dict, structure: Structure
    ) -> None:
        """
        Check that k-point density is sufficiently high and is compatible with lattice symmetry.

        Parameters
        -----------
        reasons : list[str]
            A list of error strings to update if a check fails. These are higher
            severity and would deprecate a calculation.
        valid_input_set: VaspInputSet
            Valid input set to compare user INCAR parameters to.
        kpoints : Kpoints or dict
            Kpoints object or its .as_dict() representation used in the calculation.
        structure : pymatgen.core.Structure
            The structure used in the calculation
        """

        # Check number of kpoints used
        valid_num_kpts = self._get_valid_num_kpts(valid_input_set, structure)
        valid_num_kpts = int(np.floor(valid_num_kpts * self.kpts_tolerance))

        if isinstance(kpoints, Kpoints):
            kpoints = kpoints.as_dict()

        cur_num_kpts = max(kpoints.get("nkpoints", 0), np.prod(kpoints.get("kpoints")), len(kpoints.get("kpoints")))
        if cur_num_kpts < valid_num_kpts:
            reasons.append(
                f"INPUT SETTINGS --> KPOINTS or KSPACING: {cur_num_kpts} kpoints were "
                f"used, but it should have been at least {valid_num_kpts}."
            )

        # check for valid kpoint mesh (which depends on symmetry of the structure)
        cur_kpoint_style = kpoints.get("generation_style").lower()
        is_hexagonal = structure.lattice.is_hexagonal()
        is_face_centered = structure.get_space_group_info()[0][0] == "F"
        monkhorst_mesh_is_invalid = is_hexagonal or is_face_centered
        if (
            cur_kpoint_style == "monkhorst"
            and monkhorst_mesh_is_invalid
            and any(x % 2 == 0 for x in kpoints.get("kpoints")[0])
        ):
            # only allow Monkhorst with all odd number of subdivisions per axis.
            kx, ky, kz = kpoints.get("kpoints")[0]
            reasons.append(
                f"INPUT SETTINGS --> KPOINTS or KGAMMA: ({kx}x{ky}x{kz}) "
                "Monkhorst-Pack kpoint mesh was used."
                "To be compatible with the symmetry of the lattice, "
                "a Monkhorst-Pack mesh should have only odd number of "
                "subdivisions per axis."
            )

        # Check for explicit kpoint meshes
        if (not self.allow_explicit_kpoint_mesh) and len(kpoints["kpoints"]) > 1:
            reasons.append(
                "INPUT SETTINGS --> KPOINTS: explicitly defining "
                "the k-point mesh is not currently allowed. "
                "Automatic k-point generation is required."
            )

        # Check for user shifts
        if (not self.allow_kpoint_shifts) and any(shift_val != 0 for shift_val in kpoints["usershift"]):
            reasons.append("INPUT SETTINGS --> KPOINTS: shifting the kpoint mesh is not currently allowed.")
