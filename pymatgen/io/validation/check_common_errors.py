"""Check common issues with VASP calculations."""
from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from emmet.core.tasks import TaskDoc
    from emmet.core.vasp.task_valid import TaskDocument
    from pymatgen.core import Structure
    from pymatgen.io.vasp import Incar
    from typing import Sequence


@dataclass
class CheckCommonErrors:
    """
    Check for common calculation errors.

    Parameters
    -----------
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

    defaults: dict | None = None
    # TODO: make this also work for elements Gd and Eu, which have magmoms >5 in at least one of their pure structures
    valid_max_magmoms: dict[str, float] = field(default_factory=lambda: {"Gd": 10.0, "Eu": 10.0})
    exclude_elements: set[str] = field(default_factory=lambda: {"Am", "Po"})
    valid_max_allowed_scf_gradient: float | None = None
    num_ionic_steps_to_avg_drift_over: int | None = None

    def check(
        self,
        reasons: list[str],
        warnings: list[str],
        task_doc: TaskDoc | TaskDocument,
        parameters: dict,
        structure: Structure,
    ) -> None:
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
        """

        task_doc = task_doc.model_dump()
        incar = task_doc["calcs_reversed"][0]["input"]["incar"]
        # Check for cases where both GGA and METAGGA are set. This should *not* be allowed, as it can erroneously change
        # the outputted energy significantly. See https://github.com/materialsproject/atomate2/issues/453#issuecomment-1699605867
        # for more details.
        if incar.get("GGA", "--") != "--" and str(incar.get("METAGGA", None)).lower() not in ["--", "none"]:
            reasons.append(
                "KNOWN BUG --> GGA and METAGGA should never be specified together, as this can cause major errors in the "
                "outputted energy. See https://github.com/materialsproject/atomate2/issues/453#issuecomment-1699605867 "
                "for more information."
            )

        # check if structure electronically converged
        ionic_steps = task_doc["calcs_reversed"][0]["output"]["ionic_steps"]
        final_esteps = (
            ionic_steps[-1]["electronic_steps"]
            if incar.get("ALGO", self.defaults["ALGO"]["value"]).lower() != "chi"
            else 0
        )
        # In a response function run there is no ionic steps, there is no scf step
        if parameters.get("LEPSILON", self.defaults["LEPSILON"]["value"]):
            i = 1
            to_check = {"e_wo_entrp", "e_fr_energy", "e_0_energy"}
            while set(final_esteps[i]) == to_check:
                i += 1
            is_converged = i + 1 != parameters.get("NELM", self.defaults["NELM"]["value"])
        else:
            is_converged = len(final_esteps) < parameters.get("NELM", self.defaults["NELM"]["value"])

        if not is_converged:
            reasons.append(
                "CONVERGENCE --> Did not achieve electronic convergence in the final ionic step. NELM should be increased."
            )

        # Check if drift force is too large
        try:
            all_drift_forces = task_doc["calcs_reversed"][0]["output"]["outcar"]["drift"]
            if len(all_drift_forces) < self.num_ionic_steps_to_avg_drift_over:
                drift_forces_to_avg_over = all_drift_forces
            else:
                drift_forces_to_avg_over = all_drift_forces[::-1][: self.num_ionic_steps_to_avg_drift_over]

            drift_mags_to_avg_over = [np.linalg.norm(drift_forces) for drift_forces in drift_forces_to_avg_over]
            cur_avg_drift_mag = np.average(drift_mags_to_avg_over)

            valid_max_drift = 0.05
            if cur_avg_drift_mag > valid_max_drift:
                reasons.append(
                    f"CONVERGENCE --> Excessive drift of {round(cur_avg_drift_mag,4)} eV/A is greater than allowed "
                    f"value of {valid_max_drift} eV/A."
                )
        except Exception:
            warnings.append("Drift forces not contained in calcs_reversed! Can not check for excessive drift.")

        # Check for excessively positive final energies (which usually indicates a bad structure)
        valid_max_energy_per_atom = 50
        cur_final_energy_per_atom = task_doc["output"]["energy_per_atom"]
        if cur_final_energy_per_atom > valid_max_energy_per_atom:
            reasons.append(
                f"LARGE POSITIVE FINAL ENERGY --> Final energy is {round(cur_final_energy_per_atom,4)} eV/atom, which is "
                f"greater than the maximum allowed value of {valid_max_energy_per_atom} eV/atom."
            )

        # Check for excessively large final magnetic moments
        cur_magmoms = [abs(mag["tot"]) for mag in task_doc["calcs_reversed"][0]["output"]["outcar"]["magnetization"]]
        bad_site_magmom_msgs = []
        if len(cur_magmoms) > 0:
            for site_num in range(0, len(structure)):
                cur_site_ele = structure.sites[site_num].species_string
                cur_site_magmom = cur_magmoms[site_num]
                cur_site_max_allowed_magmom = self.valid_max_magmoms.get(cur_site_ele, 5.0)

                if cur_site_magmom > cur_site_max_allowed_magmom:
                    bad_site_magmom_msgs.append(
                        f"at least one {cur_site_ele} site with magmom greater than {cur_site_max_allowed_magmom}"
                    )
        if len(bad_site_magmom_msgs) > 0:
            reasons.append(
                "MAGNETISM --> Final structure contains sites with magnetic moments "
                "that are very likely erroneous. This includes: "
                f"{'; '.join(set(bad_site_magmom_msgs))}."
            )

        # Check for a SCF gradient that is too large (usually indicates unstable calculations)
        # NOTE: do NOT use `e_0_energy`, as there is a bug in the vasprun.xml when printing that variable
        # (see https://www.vasp.at/forum/viewtopic.php?t=16942 for more details).
        skip = abs(parameters.get("NELMDL", self.defaults["NELMDL"]["value"])) - 1
        energies = [d["e_fr_energy"] for d in ionic_steps[-1]["electronic_steps"]]
        if len(energies) > skip:
            cur_max_gradient = np.max(np.gradient(energies)[skip:])
            cur_max_gradient_per_atom = cur_max_gradient / structure.num_sites
            if cur_max_gradient_per_atom > self.valid_max_allowed_scf_gradient:
                warnings.append(
                    f"STABILITY --> The max SCF gradient is {round(cur_max_gradient_per_atom,4)} eV/atom, "
                    "which is larger than the typical max expected value of "
                    f"{self.valid_max_allowed_scf_gradient} eV/atom. "
                    f"This sometimes indicates an unstable calculation."
                )
        else:
            warnings.append(
                "Not enough electronic steps to compute valid gradient" " and compare with max SCF gradient tolerance."
            )

        # Check for Am and Po elements. These currently do not have proper elemental entries
        # and will not get treated properly by the thermo builder.
        elements = set(task_doc["chemsys"].split("-"))
        if excluded_elements := self.exclude_elements.intersection(elements):
            reasons.append(
                f"COMPOSITION --> Your structure contains the elements {' '.join(excluded_elements)}, "
                "which are not currently being accepted."
            )


@dataclass
class CheckVaspVersion:
    """
    Check for common errors related to the version of VASP used.

    Parameters
    -----------
    defaults : dict
        Dict of default parameters
    """

    defaults: dict | None = None

    def check(self, reasons: list[str], vasp_version: Sequence[int], parameters: dict, incar: dict | Incar) -> None:
        """
        Check for common errors related to the version of VASP used.

        reasons : list[str]
            A list of error strings to update if a check fails. These are higher
            severity and would deprecate a calculation.
        vasp_version: Sequence[int]
            Vasp version, e.g., 6.4.1 could be represented as (6,4,1)
        parameters : dict[str,Any]
            Dict of user-supplied/-parsed INCAR parameters.
        incar : dict | Incar
            INCAR corresponding to the calculation.
        """
        if (
            vasp_version[0] == 5
            and (incar.get("METAGGA", self.defaults["METAGGA"]["value"]) not in [None, "--", "None"])
            and parameters.get("ISPIN", self.defaults["ISPIN"]["value"]) == 2
        ):
            reasons.append(
                "POTENTIAL BUG --> We believe that there may be a bug with spin-polarized calculations for METAGGAs "
                "in some versions of VASP 5. Please create a new GitHub issue if you believe this "
                "is not the case and we will consider changing this check!"
            )
        elif (list(vasp_version) != [5, 4, 4]) and (vasp_version[0] < 6):
            vasp_version_str = ".".join([str(x) for x in vasp_version])
            reasons.append(
                f"VASP VERSION --> This calculation is using VASP version {vasp_version_str}, "
                "but we only allow versions 5.4.4 and >=6.0.0 (as of July 2023)."
            )
