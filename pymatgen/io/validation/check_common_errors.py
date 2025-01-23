"""Check common issues with VASP calculations."""

from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np

from typing import TYPE_CHECKING

from emmet.core.vasp.calc_types.enums import TaskType
from pymatgen.core import Structure

from pymatgen.io.validation.common import BaseValidator

if TYPE_CHECKING:
    from emmet.core.tasks import TaskDoc
    from emmet.core.vasp.calc_types.enums import RunType
    from emmet.core.vasp.task_valid import TaskDocument
    from pymatgen.io.vasp.inputs import Incar
    from typing import Sequence
    from numpy.typing import ArrayLike


@dataclass
class CheckCommonErrors(BaseValidator):
    """
    Check for common calculation errors.

    Parameters
    -----------
    reasons : list[str]
        A list of error strings to update if a check fails. These are higher
        severity and would deprecate a calculation.
    warnings : list[str]
        A list of warning strings to update if a check fails. These are lower
        severity and would flag a calculation for possible review.
    task_doc : emmet.core TaskDoc | TaskDocument
        Task document parsed from the calculation directory.
    parameters : dict[str,Any]
        Dict of user-supplied/-parsed INCAR parameters.
    structure: Pymatgen Structure
        Structure used in the calculation.
    run_type: RunType
        Run type of the calculation
    name : str = "Check common errors"
        Name of the validator
    fast : bool = False
        True: stop validation when any single check fails
    defaults : dict
        Dict of default parameters
    valid_max_magmoms : dict[str,float]
        Dict of maximum magmoms corresponding to a given element.
    exclude_elements : set[str]
        Set of elements that cannot be added to the Materials Project's hull.
    valid_max_allowed_scf_gradient : float
        Largest permitted change in total energies between two SCF cycles.
    num_ionic_steps_to_avg_drift_over : int
        Number of ionic steps to average over to yield the drift in total energy.
    """

    reasons: list[str]
    warnings: list[str]
    task_doc: TaskDoc | TaskDocument = None
    parameters: dict = None
    structure: Structure = None
    run_type: RunType = None
    name: str = "Check common errors"
    fast: bool = False
    defaults: dict | None = None
    # TODO: make this also work for elements Gd and Eu, which have magmoms >5 in at least one of their pure structures
    valid_max_magmoms: dict[str, float] = field(default_factory=lambda: {"Gd": 10.0, "Eu": 10.0})
    exclude_elements: set[str] = field(default_factory=lambda: {"Am", "Po"})
    valid_max_allowed_scf_gradient: float | None = None
    num_ionic_steps_to_avg_drift_over: int | None = None

    def __post_init__(self):
        self.incar = self.task_doc["calcs_reversed"][0]["input"]["incar"]
        self.ionic_steps = self.task_doc["calcs_reversed"][0]["output"]["ionic_steps"]

    def _check_run_type(self) -> None:
        if f"{self.run_type}".upper() not in {"GGA", "GGA+U", "PBE", "PBE+U", "R2SCAN"}:
            self.reasons.append(f"FUNCTIONAL --> Functional {self.run_type} not currently accepted.")

    def _check_parse(self) -> None:
        if self.parameters == {} or self.parameters is None:
            self.reasons.append(
                "CAN NOT PROPERLY PARSE CALCULATION --> Issue parsing input parameters from the vasprun.xml file."
            )

    def _check_gga_and_metagga(self) -> None:
        # Check for cases where both GGA and METAGGA are set. This should *not* be allowed, as it can erroneously change
        # the outputted energy significantly. See https://github.com/materialsproject/atomate2/issues/453#issuecomment-1699605867
        # for more details.
        if self.incar.get("GGA", "--") != "--" and str(self.incar.get("METAGGA", None)).lower() not in {"--", "none"}:
            self.reasons.append(
                "KNOWN BUG --> GGA and METAGGA should never be specified together, as this can cause major errors in the "
                "outputted energy. See https://github.com/materialsproject/atomate2/issues/453#issuecomment-1699605867 "
                "for more information."
            )

    def _check_electronic_convergence(self) -> None:
        # check if structure electronically converged

        if self.incar.get("ALGO", self.defaults["ALGO"]["value"]).lower() != "chi":
            # Response function calculations are non-self-consistent: only one ionic step, no electronic SCF
            if self.parameters.get("LEPSILON", self.defaults["LEPSILON"]["value"]):
                final_esteps = self.ionic_steps[-1]["electronic_steps"]
                to_check = {"e_wo_entrp", "e_fr_energy", "e_0_energy"}

                for i in range(len(final_esteps)):
                    if set(final_esteps[i]) != to_check:
                        break
                    i += 1

                is_converged = i + 1 < self.parameters.get("NELM", self.defaults["NELM"]["value"])
                n_non_conv = 1

            else:
                conv_steps = [
                    len(self.ionic_steps[i]["electronic_steps"])
                    < self.parameters.get("NELM", self.defaults["NELM"]["value"])
                    for i in range(len(self.ionic_steps))
                ]
                is_converged = all(conv_steps)
                n_non_conv = len([step for step in conv_steps if not step])

            if not is_converged:
                self.reasons.append(
                    f"CONVERGENCE --> Did not achieve electronic convergence in {n_non_conv} ionic step(s). NELM should be increased."
                )

    def _check_drift_forces(self) -> None:
        # Check if drift force is too large
        try:
            all_drift_forces = self.task_doc["calcs_reversed"][0]["output"]["outcar"]["drift"]
            if len(all_drift_forces) < self.num_ionic_steps_to_avg_drift_over:
                drift_forces_to_avg_over = all_drift_forces
            else:
                drift_forces_to_avg_over = all_drift_forces[::-1][: self.num_ionic_steps_to_avg_drift_over]

            drift_mags_to_avg_over = [np.linalg.norm(drift_forces) for drift_forces in drift_forces_to_avg_over]
            cur_avg_drift_mag = np.average(drift_mags_to_avg_over)

            valid_max_drift = 0.05
            if cur_avg_drift_mag > valid_max_drift:
                self.reasons.append(
                    f"CONVERGENCE --> Excessive drift of {round(cur_avg_drift_mag,4)} eV/A is greater than allowed "
                    f"value of {valid_max_drift} eV/A."
                )
        except Exception:
            self.warnings.append("Drift forces not contained in calcs_reversed! Can not check for excessive drift.")

    def _check_positive_energy(self) -> None:
        # Check for excessively positive final energies (which usually indicates a bad structure)
        valid_max_energy_per_atom = 50
        cur_final_energy_per_atom = self.task_doc["output"]["energy_per_atom"]
        if cur_final_energy_per_atom > valid_max_energy_per_atom:
            self.reasons.append(
                f"LARGE POSITIVE FINAL ENERGY --> Final energy is {round(cur_final_energy_per_atom,4)} eV/atom, which is "
                f"greater than the maximum allowed value of {valid_max_energy_per_atom} eV/atom."
            )

    def _check_large_magmoms(self) -> None:
        # Check for excessively large final magnetic moments
        cur_magmoms = [
            abs(mag["tot"]) for mag in self.task_doc["calcs_reversed"][0]["output"]["outcar"]["magnetization"]
        ]
        bad_site_magmom_msgs = []
        if len(cur_magmoms) > 0:
            for site_num in range(0, len(self.structure)):
                cur_site_ele = self.structure.sites[site_num].species_string
                cur_site_magmom = cur_magmoms[site_num]
                cur_site_max_allowed_magmom = self.valid_max_magmoms.get(cur_site_ele, 5.0)

                if cur_site_magmom > cur_site_max_allowed_magmom:
                    bad_site_magmom_msgs.append(
                        f"at least one {cur_site_ele} site with magmom greater than {cur_site_max_allowed_magmom}"
                    )

        if len(bad_site_magmom_msgs) > 0:
            self.reasons.append(
                "MAGNETISM --> Final structure contains sites with magnetic moments "
                "that are very likely erroneous. This includes: "
                f"{'; '.join(set(bad_site_magmom_msgs))}."
            )

    def _check_scf_grad(self) -> None:
        # Check for a SCF gradient that is too large (usually indicates unstable calculations)
        # NOTE: do NOT use `e_0_energy`, as there is a bug in the vasprun.xml when printing that variable
        # (see https://www.vasp.at/forum/viewtopic.php?t=16942 for more details).
        skip = abs(self.parameters.get("NELMDL", self.defaults["NELMDL"]["value"])) - 1
        energies = [d["e_fr_energy"] for d in self.ionic_steps[-1]["electronic_steps"]]
        if len(energies) > skip:
            cur_max_gradient = np.max(np.gradient(energies)[skip:])
            cur_max_gradient_per_atom = cur_max_gradient / self.structure.num_sites
            if cur_max_gradient_per_atom > self.valid_max_allowed_scf_gradient:
                self.warnings.append(
                    f"STABILITY --> The max SCF gradient is {round(cur_max_gradient_per_atom,4)} eV/atom, "
                    "which is larger than the typical max expected value of "
                    f"{self.valid_max_allowed_scf_gradient} eV/atom. "
                    f"This sometimes indicates an unstable calculation."
                )
        else:
            self.warnings.append(
                "Not enough electronic steps to compute valid gradient and compare with max SCF gradient tolerance."
            )

    def _check_unused_elements(self) -> None:
        # Check for Am and Po elements. These currently do not have proper elemental entries
        # and will not get treated properly by the thermo builder.
        elements = set(self.task_doc["chemsys"].split("-"))
        if excluded_elements := self.exclude_elements.intersection(elements):
            self.reasons.append(
                f"COMPOSITION --> Your structure contains the elements {' '.join(excluded_elements)}, "
                "which are not currently being accepted."
            )


@dataclass
class CheckVaspVersion(BaseValidator):
    """
    Check for common errors related to the version of VASP used.

    Parameters
    -----------
    reasons : list[str]
        A list of error strings to update if a check fails. These are higher
        severity and would deprecate a calculation.
    warnings : list[str]
        A list of warning strings to update if a check fails. These are lower
        severity and would flag a calculation for possible review.
    vasp_version: Sequence[int]
        Vasp version, e.g., 6.4.1 could be represented as (6,4,1)
    parameters : dict[str,Any]
        Dict of user-supplied/-parsed INCAR parameters.
    incar : dict | Incar
        INCAR corresponding to the calculation.
    name : str = "Base validator class"
        Name of the validator class
    fast : bool = False
        Whether to perform quick check.
        True: stop validation if any check fails.
        False: perform all checks.
    defaults : dict
        Dict of default parameters
    """

    reasons: list[str]
    warnings: list[str]
    vasp_version: Sequence[int] = None
    parameters: dict = None
    incar: dict | Incar = None
    name: str = "VASP version validator"
    defaults: dict | None = None

    def _check_vasp_version(self) -> None:
        """
        Check for common errors related to the version of VASP used.

        reasons : list[str]
            A list of error strings to update if a check fails. These are higher
            severity and would deprecate a calculation.
        warnings : list[str]
            A list of warning strings to update if a check fails. These are lower
            severity and would flag a calculation for possible review.
        """
        if (
            self.vasp_version[0] == 5
            and (self.incar.get("METAGGA", self.defaults["METAGGA"]["value"]) not in [None, "--", "None"])
            and self.parameters.get("ISPIN", self.defaults["ISPIN"]["value"]) == 2
        ):
            self.reasons.append(
                "POTENTIAL BUG --> We believe that there may be a bug with spin-polarized calculations for METAGGAs "
                "in some versions of VASP 5. Please create a new GitHub issue if you believe this "
                "is not the case and we will consider changing this check!"
            )
        elif (list(self.vasp_version) != [5, 4, 4]) and (self.vasp_version[0] < 6):
            vasp_version_str = ".".join([str(x) for x in self.vasp_version])
            self.reasons.append(
                f"VASP VERSION --> This calculation is using VASP version {vasp_version_str}, "
                "but we only allow versions 5.4.4 and >=6.0.0 (as of July 2023)."
            )


@dataclass
class CheckStructureProperties(BaseValidator):
    """Check structure for options that are not suitable for thermodynamic calculations."""

    reasons: list[str]
    warnings: list[str]
    structures: list[dict | Structure | None] = None
    task_type: TaskType = None
    name: str = "VASP POSCAR properties validator"
    site_properties_to_check: tuple[str, ...] = ("selective_dynamics", "velocities")

    def __post_init__(self) -> None:
        """Extract required structure site properties."""

        for idx, struct in enumerate(self.structures):
            if isinstance(struct, dict):
                self.structures[idx] = Structure.from_dict(struct)

        self._site_props = {
            k: [struct.site_properties.get(k) for struct in self.structures if struct]  # type: ignore[union-attr]
            for k in self.site_properties_to_check
        }

    @staticmethod
    def _has_frozen_degrees_of_freedom(selective_dynamics_array: ArrayLike[bool] | None) -> bool:
        """Check selective dynamics array for False values."""
        if selective_dynamics_array is None:
            return False
        return not np.all(selective_dynamics_array)

    def _check_selective_dynamics(self) -> None:
        """Check structure for inappropriate site properties."""

        if (selec_dyn := self._site_props.get("selective_dynamics")) is not None and self.task_type in {
            TaskType.Structure_Optimization,
            TaskType.Deformation,
        }:
            if any(self._has_frozen_degrees_of_freedom(sd_array) for sd_array in selec_dyn):
                self.reasons.append(
                    "Selective dynamics: certain degrees of freedom in the structure "
                    "were not permitted to relax. To correctly place entries on the convex "
                    "hull, all degrees of freedom should be allowed to relax."
                )

    @staticmethod
    def _has_nonzero_velocities(velocities: ArrayLike | None, tol: float = 1.0e-8) -> bool:
        if velocities is None:
            return False
        return np.any(np.abs(velocities) > tol)

    def _check_velocities(self) -> None:
        """Check structure for non-zero velocities."""

        if (velos := self._site_props.get("velocities")) is not None and self.task_type != TaskType.Molecular_Dynamics:
            if any(self._has_nonzero_velocities(velo) for velo in velos):
                self.warnings.append(
                    "At least one of the structures had non-zero velocities. "
                    f"While these are ignored by VASP for {self.task_type} "
                    "calculations, please ensure that you intended to run a "
                    "non-molecular dynamics calculation."
                )
