"""Check common issues with VASP calculations."""

from __future__ import annotations
from pydantic import Field
import numpy as np
from typing import TYPE_CHECKING

from pymatgen.io.validation.common import SETTINGS, BaseValidator

if TYPE_CHECKING:
    from collections.abc import Sequence
    from numpy.typing import ArrayLike

    from pymatgen.io.validation.common import VaspFiles


class CheckCommonErrors(BaseValidator):
    """
    Check for common calculation errors.
    """

    name: str = "Check common errors"
    valid_max_magmoms: dict[str, float] = Field(
        default_factory=lambda: {"Gd": 10.0, "Eu": 10.0},
        description="Dict of maximum magmoms corresponding to a given element.",
    )
    exclude_elements: set[str] = Field(
        default_factory=lambda: {"Am", "Po"},
        description="Set of elements that cannot be added to the Materials Project's hull.",
    )
    valid_max_allowed_scf_gradient: float | None = Field(
        SETTINGS.VASP_MAX_SCF_GRADIENT, description="Largest permitted change in total energies between two SCF cycles."
    )
    num_ionic_steps_to_avg_drift_over: int | None = Field(
        SETTINGS.VASP_NUM_IONIC_STEPS_FOR_DRIFT,
        description="Number of ionic steps to average over to yield the drift in total energy.",
    )
    valid_max_energy_per_atom: float | None = Field(
        SETTINGS.VASP_MAX_POSITIVE_ENERGY,
        description="The maximum permitted, self-consistent positive energy in eV/atom.",
    )

    def _check_vasp_version(self, vasp_files: VaspFiles, reasons: list[str], warnings: list[str]) -> None:
        """
        Check for common errors related to the version of VASP used.

        reasons : list[str]
            A list of error strings to update if a check fails. These are higher
            severity and would deprecate a calculation.
        warnings : list[str]
            A list of warning strings to update if a check fails. These are lower
            severity and would flag a calculation for possible review.
        """

        if not vasp_files.vasp_version:
            # Skip if vasprun.xml not specified
            return

        if (
            vasp_files.vasp_version[0] == 5
            and (
                vasp_files.user_input.incar.get("METAGGA", self.vasp_defaults["METAGGA"].value)
                not in [None, "--", "None"]
            )
            and vasp_files.user_input.incar.get("ISPIN", self.vasp_defaults["ISPIN"].value) == 2
        ):
            reasons.append(
                "POTENTIAL BUG --> We believe that there may be a bug with spin-polarized calculations for METAGGAs "
                "in some versions of VASP 5. Please create a new GitHub issue if you believe this "
                "is not the case and we will consider changing this check!"
            )
        elif (list(vasp_files.vasp_version) != [5, 4, 4]) and (vasp_files.vasp_version[0] < 6):
            vasp_version_str = ".".join([str(x) for x in vasp_files.vasp_version])
            reasons.append(
                f"VASP VERSION --> This calculation is using VASP version {vasp_version_str}, "
                "but we only allow versions 5.4.4 and >=6.0.0 (as of July 2023)."
            )

    def _check_electronic_convergence(self, vasp_files: VaspFiles, reasons: list[str], warnings: list[str]) -> None:
        # check if structure electronically converged

        if (
            vasp_files.user_input.incar.get("ALGO", self.vasp_defaults["ALGO"].value).lower() != "chi"
            and vasp_files.vasprun
        ):
            # Response function calculations are non-self-consistent: only one ionic step, no electronic SCF
            if vasp_files.user_input.incar.get("LEPSILON", self.vasp_defaults["LEPSILON"].value):
                final_esteps = vasp_files.vasprun.ionic_steps[-1]["electronic_steps"]
                to_check = {"e_wo_entrp", "e_fr_energy", "e_0_energy"}

                for i in range(len(final_esteps)):
                    if set(final_esteps[i]) != to_check:
                        break
                    i += 1

                is_converged = i + 1 < vasp_files.user_input.incar.get("NELM", self.vasp_defaults["NELM"].value)
                n_non_conv = 1

            else:
                conv_steps = [
                    len(ionic_step["electronic_steps"])
                    < vasp_files.user_input.incar.get("NELM", self.vasp_defaults["NELM"].value)
                    for ionic_step in vasp_files.vasprun.ionic_steps
                ]
                is_converged = all(conv_steps)
                n_non_conv = len([step for step in conv_steps if not step])

            if not is_converged:
                reasons.append(
                    f"CONVERGENCE --> Did not achieve electronic convergence in {n_non_conv} ionic step(s). NELM should be increased."
                )

    def _check_drift_forces(self, vasp_files: VaspFiles, reasons: list[str], warnings: list[str]) -> None:
        # Check if drift force is too large

        if not self.num_ionic_steps_to_avg_drift_over or not vasp_files.outcar:
            return

        if all_drift_forces := vasp_files.outcar.drift:
            if len(all_drift_forces) < self.num_ionic_steps_to_avg_drift_over:
                drift_forces_to_avg_over = all_drift_forces
            else:
                drift_forces_to_avg_over = all_drift_forces[::-1][: self.num_ionic_steps_to_avg_drift_over]

            drift_mags_to_avg_over = [np.linalg.norm(drift_forces) for drift_forces in drift_forces_to_avg_over]
            cur_avg_drift_mag = np.average(drift_mags_to_avg_over)

            valid_max_drift = 0.05
            if cur_avg_drift_mag > valid_max_drift:
                warnings.append(
                    f"CONVERGENCE --> Excessive drift of {round(cur_avg_drift_mag,4)} eV/A is greater than allowed "
                    f"value of {valid_max_drift} eV/A."
                )
        else:
            warnings.append(
                "Could not determine drift forces from OUTCAR, and thus could not check for excessive drift."
            )

    def _check_positive_energy(self, vasp_files: VaspFiles, reasons: list[str], warnings: list[str]) -> None:
        # Check for excessively positive final energies (which usually indicates a bad structure)
        if (
            vasp_files.vasprun
            and self.valid_max_energy_per_atom
            and (cur_final_energy_per_atom := vasp_files.vasprun.final_energy / len(vasp_files.user_input.structure))
            > self.valid_max_energy_per_atom
        ):
            reasons.append(
                f"LARGE POSITIVE FINAL ENERGY --> Final energy is {round(cur_final_energy_per_atom,4)} eV/atom, which is "
                f"greater than the maximum allowed value of {self.valid_max_energy_per_atom} eV/atom."
            )

    def _check_large_magmoms(self, vasp_files: VaspFiles, reasons: list[str], warnings: list[str]) -> None:
        # Check for excessively large final magnetic moments

        if (
            not vasp_files.outcar
            or not vasp_files.outcar.magnetization
            or any(mag.get("tot") is None for mag in vasp_files.outcar.magnetization)
        ):
            warnings.append("MAGNETISM --> No OUTCAR file specified or data missing.")
            return

        cur_magmoms = [abs(mag["tot"]) for mag in vasp_files.outcar.magnetization]
        bad_site_magmom_msgs = []
        if len(cur_magmoms) > 0:
            for site_num in range(0, len(vasp_files.user_input.structure)):
                cur_site_ele = vasp_files.user_input.structure.sites[site_num].species_string
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

    def _check_scf_grad(self, vasp_files: VaspFiles, reasons: list[str], warnings: list[str]) -> None:
        # Check for a SCF gradient that is too large (usually indicates unstable calculations)
        # NOTE: do NOT use `e_0_energy`, as there is a bug in the vasprun.xml when printing that variable
        # (see https://www.vasp.at/forum/viewtopic.php?t=16942 for more details).

        if not vasp_files.vasprun or not self.valid_max_allowed_scf_gradient:
            return

        skip = abs(vasp_files.user_input.incar.get("NELMDL", self.vasp_defaults["NELMDL"].value)) - 1

        energies = [d["e_fr_energy"] for d in vasp_files.vasprun.ionic_steps[-1]["electronic_steps"]]
        if len(energies) > skip:
            cur_max_gradient = np.max(np.gradient(energies)[skip:])
            cur_max_gradient_per_atom = cur_max_gradient / vasp_files.user_input.structure.num_sites
            if self.valid_max_allowed_scf_gradient and cur_max_gradient_per_atom > self.valid_max_allowed_scf_gradient:
                warnings.append(
                    f"STABILITY --> The max SCF gradient is {round(cur_max_gradient_per_atom,4)} eV/atom, "
                    "which is larger than the typical max expected value of "
                    f"{self.valid_max_allowed_scf_gradient} eV/atom. "
                    f"This sometimes indicates an unstable calculation."
                )
        else:
            warnings.append(
                "Not enough electronic steps to compute valid gradient and compare with max SCF gradient tolerance."
            )

    def _check_unused_elements(self, vasp_files: VaspFiles, reasons: list[str], warnings: list[str]) -> None:
        # Check for Am and Po elements. These currently do not have proper elemental entries
        # and will not get treated properly by the thermo builder.
        elements = set(vasp_files.user_input.structure.composition.chemical_system.split("-"))
        if excluded_elements := self.exclude_elements.intersection(elements):
            reasons.append(
                f"COMPOSITION --> Your structure contains the elements {' '.join(excluded_elements)}, "
                "which are not currently being accepted."
            )


class CheckStructureProperties(BaseValidator):
    """Check structure for options that are not suitable for thermodynamic calculations."""

    name: str = "VASP POSCAR properties validator"
    site_properties_to_check: tuple[str, ...] = Field(
        ("selective_dynamics", "velocities"), description="Which site properties to check on a structure."
    )

    @staticmethod
    def _has_frozen_degrees_of_freedom(selective_dynamics_array: Sequence[bool] | None) -> bool:
        """Check selective dynamics array for False values."""
        if selective_dynamics_array is None:
            return False
        return not np.all(selective_dynamics_array)

    def _check_selective_dynamics(self, vasp_files: VaspFiles, reasons: list[str], warnings: list[str]) -> None:
        """Check structure for inappropriate site properties."""
        if (
            selec_dyn := vasp_files.user_input.structure.site_properties.get("selective_dynamics")
        ) is not None and vasp_files.run_type == "relax":
            if any(self._has_frozen_degrees_of_freedom(sd_array) for sd_array in selec_dyn):
                reasons.append(
                    "Selective dynamics: certain degrees of freedom in the structure "
                    "were not permitted to relax. To correctly place entries on the convex "
                    "hull, all degrees of freedom should be allowed to relax."
                )

    @staticmethod
    def _has_nonzero_velocities(velocities: ArrayLike | None, tol: float = 1.0e-8) -> bool:
        if velocities is None:
            return False
        return np.any(np.abs(velocities) > tol)  # type: ignore [return-value]

    def _check_velocities(self, vasp_files: VaspFiles, reasons: list[str], warnings: list[str]) -> None:
        """Check structure for non-zero velocities."""
        if (
            velos := vasp_files.user_input.structure.site_properties.get("velocities")
        ) is not None and vasp_files.run_type != "md":
            if any(self._has_nonzero_velocities(velo) for velo in velos):
                warnings.append(
                    "At least one of the structures had non-zero velocities. "
                    f"While these are ignored by VASP for {vasp_files.run_type} "
                    "calculations, please ensure that you intended to run a "
                    "non-molecular dynamics calculation."
                )
