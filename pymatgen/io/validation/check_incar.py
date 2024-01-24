"""Module for validating VASP INCAR files"""
from __future__ import annotations
import copy
from importlib.resources import files as import_res_files
from monty.serialization import loadfn
from math import isclose
import numpy as np
from emmet.core.vasp.calc_types.enums import TaskType

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import Any, Sequence
    from pymatgen.core import Structure
    from pymatgen.io.vasp import Incar
    from pymatgen.io.vasp.sets import VaspInputSet

_vasp_defaults = loadfn(import_res_files("pymatgen.io.validation") / "vasp_defaults.yaml")


def _check_incar(
    reasons,
    warnings,
    valid_input_set,
    structure,
    task_doc,
    calcs_reversed,
    ionic_steps,
    nionic_steps,
    parameters,
    incar,
    potcar,
    vasp_version: Sequence[int],
    task_type,
    fft_grid_tolerance,
):
    """
    note that all changes to `reasons` and `warnings` can be done in-place
    (and hence there is no need to return those variables after every function call).
    Any cases where that is not done is just to make the code more readable.
    # MK: A better description of this class should be included above the current docstring. 
        # I think writing clearly how the whole function operates, in chronological order, would be best.
    """

    working_params = GetParams( # MK: unclear
        parameters=parameters,
        defaults=_vasp_defaults,
        input_set=valid_input_set,
        structure=structure,
        task_doc=task_doc,
        calcs_reversed=calcs_reversed,
        ionic_steps=ionic_steps,
        nionic_steps=nionic_steps,
        incar=incar,
        potcar=potcar,
        vasp_version=vasp_version,
        task_type=task_type,
        fft_grid_tolerance=fft_grid_tolerance,
    )

    simple_validator = BasicValidator()
    for key in working_params.defaults:
        simple_validator.check_parameter(
            reasons=reasons,
            warnings=warnings,
            input_tag=working_params.defaults[key].get("alias", key),
            current_values=working_params.parameters[key],
            reference_values=working_params.valid_values[key],
            operations=working_params.defaults[key]["operation"],
            tolerance=working_params.defaults[key]["tolerance"],
            append_comments=working_params.defaults[key]["comment"],
            severity=working_params.defaults[key]["severity"],
        )

    return reasons


class GetParams:
    """Initialize current params and update defaults as needed.""" # MK: unclear, expand

    _default_defaults = {
        "value": None,
        "tag": None,
        "operation": None,
        "comment": None,
        "tolerance": 1.0e-4,
        "severity": "reason",
    }

    def __init__(
        self,
        parameters: dict,
        defaults: dict,
        input_set: VaspInputSet,
        structure: Structure,
        task_doc,
        calcs_reversed: list,
        ionic_steps,
        nionic_steps,
        incar: Incar,
        potcar,
        vasp_version: Sequence[int],
        task_type,
        fft_grid_tolerance: float,
    ) -> None: 
        # MK: unclear. I think a thorough docstring describing how this class init 
        # operates, in chronological order, would be best.
        self.parameters = copy.deepcopy(parameters)
        self.defaults = copy.deepcopy(defaults)
        self.input_set = input_set
        self.incar = incar
        self.vasp_version = vasp_version
        self.calcs_reversed = calcs_reversed
        self.structure = structure
        self.valid_values: dict[str, Any] = {}

        self.task_doc = task_doc
        self._fft_grid_tolerance = fft_grid_tolerance
        self._task_type = task_type
        self._ionic_steps = ionic_steps
        self._nionic_steps = nionic_steps

        self.categories: dict[str, list[str]] = {}
        for key in self.defaults:
            if self.defaults[key]["tag"] not in self.categories:
                self.categories[self.defaults[key]["tag"]] = []
            self.categories[self.defaults[key]["tag"]].append(key)

        self.add_defaults_to_parameters(valid_values_source=self.input_set.incar)

        self.parameter_updates: dict[str, Any] = {
            "dft+u": self.update_u_params,
            "symmetry": self.update_symmetry_params,
            "startup": self.update_startup_params,
            "precision": self.update_precision_params,
            "misc special": self.update_misc_params,
            "hybrid": self.update_hybrid_functional_params,
            "fft": self.update_fft_params,
            "density mixing": self.update_lmaxmix_and_lmaxtau,
            "smearing": self.update_smearing,
            "electronic": self.update_electronic_params,
            "ionic": self.update_ionic_params,
        }

        for key in self.parameter_updates:
            self.parameter_updates[key]()

        self.add_defaults_to_parameters()

        for key in self.defaults:
            for attr in self._default_defaults:
                self.defaults[key][attr] = self.defaults[key].get(attr, self._default_defaults[attr])

    def add_defaults_to_parameters(self, valid_values_source: dict | None = None) -> None:
        # update parameters with initial defaults
        valid_values_source = valid_values_source or self.valid_values

        for key in self.defaults:
            self.parameters[key] = self.parameters.get(key, self.defaults[key]["value"])
            self.valid_values[key] = valid_values_source.get(key, self.defaults[key]["value"])

    def update_u_params(self) -> None:
        if not self.parameters["LDAU"]:
            return

        for key in self.categories["dft+u"]:
            valid_value = self.input_set.incar.get(key, self.defaults[key]["value"])

            # TODO: ADK: is LDAUTYPE usually specified as a list??
            if key == "LDAUTYPE":
                self.parameters[key] = (
                    self.parameters[key][0] if isinstance(self.parameters[key], list) else self.parameters[key]
                )
                self.valid_values[key] = valid_value[0] if isinstance(valid_value, list) else valid_value
            else:
                self.parameters[key] = self.incar.get(key, self.defaults[key]["value"])
            self.defaults[key]["operation"] = "=="

    def update_symmetry_params(self) -> None:
        # ISYM.
        if self.parameters["LHFCALC"]:
            self.defaults["ISYM"]["value"] = 3

        # allow ISYM as good or better than what is specified in the valid_input_set.
        if "ISYM" in self.input_set.incar.keys():
            self.valid_values["ISYM"] = list(range(-1, self.input_set.incar.get("ISYM") + 1, 1))
        else:  # otherwise let ISYM = -1, 0, or 2
            self.valid_values["ISYM"] = [-1, 0, 2]
        self.defaults["ISYM"]["operation"] = "in"

        # SYMPREC.
        # custodian will set SYMPREC to a maximum of 1e-3 (as of August 2023)
        self.valid_values["SYMPREC"] = 1e-3
        self.defaults["SYMPREC"].update(
            {
                "operation": ">=",
                "comment": (
                    "If you believe that this SYMPREC value is necessary "
                    "(perhaps this calculation has a very large cell), please create "
                    "a GitHub issue and we will consider to admit your calculations."
                ),
            }
        )

    def update_startup_params(self) -> None:
        self.valid_values["ISTART"] = [0, 1, 2]

        # ICHARG.
        if self.input_set.incar.get("ICHARG", self.defaults["ICHARG"]["value"]) < 10:
            self.valid_values["ICHARG"] = 9  # should be <10 (SCF calcs)
            self.defaults["ICHARG"]["operation"] = ">="
        else:
            self.valid_values["ICHARG"] = self.input_set.incar.get("ICHARG")
            self.defaults["ICHARG"]["operation"] = "=="

    def update_precision_params(self) -> None:
        # LREAL.
        # Do NOT use the value for LREAL from the `Vasprun.parameters` object, as VASP changes these values
        # relative to the INCAR. Rather, check the LREAL value in the `Vasprun.incar` object.
        if str(self.input_set.incar.get("LREAL")).upper() in ["AUTO", "A"]:
            self.valid_values["LREAL"] = ["FALSE", "AUTO", "A"]
        elif str(self.input_set.incar.get("LREAL")).upper() in ["FALSE"]:
            self.valid_values["LREAL"] = ["FALSE"]

        self.parameters["LREAL"] = str(self.incar.get("LREAL", self.defaults["LREAL"]["value"])).upper()
        # PREC.
        if self.input_set.incar.get("PREC", self.defaults["PREC"]["value"]).upper() in ["ACCURATE", "HIGH"]:
            self.valid_values["PREC"] = ["ACCURATE", "ACCURA", "HIGH"]
        else:
            raise ValueError("Validation code check for PREC tag needs to be updated to account for a new input set!")
        self.defaults["PREC"]["operation"] = "in"

        # ROPT. Should be better than or equal to default for the PREC level.
        # This only matters if projectors are done in real-space.
        # Note that if the user sets LREAL = Auto in their Incar, it will show
        # up as "True" in the `parameters` object (hence we use the `parameters` object)
        # According to VASP wiki (https://www.vasp.at/wiki/index.php/ROPT), only
        # the magnitude of ROPT is relevant for precision.
        if self.parameters["LREAL"] == "TRUE":
            # this only matters if projectors are done in real-space.
            cur_prec = self.parameters["PREC"].upper()
            ropt_default = {
                "NORMAL": -5e-4,
                "ACCURATE": -2.5e-4,
                "ACCURA": -2.5e-4,
                "LOW": -0.01,
                "MED": -0.002,
                "HIGH": -4e-4,
            }
            self.parameters["ROPT"] = [abs(value) for value in self.parameters.get("ROPT", [ropt_default[cur_prec]])]
            self.defaults["ROPT"] = {
                "value": [abs(ropt_default[cur_prec]) for _ in self.parameters["ROPT"]],
                "tag": "startup",
                "operation": [">=" for _ in self.parameters["ROPT"]],
            }

    def update_misc_params(self) -> None:
        # EFERMI. Only available for VASP >= 6.4. Should not be set to a numerical
        # value, as this may change the number of electrons.
        # self.vasp_version = (major, minor, patch)
        if (self.vasp_version[0] >= 6) and (self.vasp_version[1] >= 4):
            # Must check EFERMI in the *incar*, as it is saved as a numerical
            # value after VASP guesses it in the vasprun.xml `parameters`
            # (which would always cause this check to fail, even if the user
            # set EFERMI properly in the INCAR).
            self.parameters["EFERMI"] = self.incar.get("EFERMI", self.defaults["EFERMI"]["value"])
            self.valid_values["EFERMI"] = ["LEGACY", "MIDGAP"]
            self.defaults["EFERMI"]["operation"] = "in"

        # IWAVPR.
        if self.incar.get("IWAVPR"):
            self.parameters["IWAVPR"] = self.incar["IWAVPR"] if self.incar["IWAVPR"] is not None else 0
            self.defaults["IWAVPR"].update(
                {"operation": "==", "comment": "VASP discourages users from setting the IWAVPR tag (as of July 2023)."}
            )

        # LCORR.
        if self.parameters["IALGO"] != 58:
            self.defaults["LCORR"].update(
                {
                    "operation": "==",
                }
            )

        if (
            self.parameters["ISPIN"] == 2
            and len(self.calcs_reversed[0]["output"]["outcar"]["magnetization"]) != self.structure.num_sites
        ):
            self.defaults["LORBIT"].update(
                {
                    "operation": "auto fail",
                    "comment": (
                        "Magnetization values were not written "
                        "to the OUTCAR. This is usually due to LORBIT being set to None or "
                        "False for calculations with ISPIN=2."
                    ),
                }
            )

        if self.parameters["LORBIT"] >= 11 and self.parameters["ISYM"] and (self.vasp_version[0] < 6):
            self.defaults["LORBIT"]["warning"] = (
                "For LORBIT >= 11 and ISYM = 2 the partial charge densities are not correctly symmetrized and can result "
                "in different charges for symmetrically equivalent partial charge densities. This issue is fixed as of version "
                ">=6. See the vasp wiki page for LORBIT for more details."
            )

        # RWIGS and VCA - do not set
        for key in ["RWIGS", "VCA"]:
            aux_str = ""
            if key == "RWIGS":
                aux_str = " This is because it will change some outputs like the magmom on each site."
            self.defaults[key].update(
                {
                    "value": [self.defaults[key] for _ in self.parameters[key]],
                    "operation": ["==" for _ in self.parameters[key]],
                    "comment": f"{key} should not be set. {aux_str}",
                }
            )

    def update_hybrid_functional_params(self) -> None:
        self.valid_values["LHFCALC"] = self.input_set.incar.get("LHFCALC", self.defaults["LHFCALC"]["value"])

        if self.valid_values["LHFCALC"]:
            self.defaults["AEXX"] = 0.25
            self.parameters["AEXX"] = self.parameters.get("AEXX", self.defaults["AEXX"])
            self.defaults["AGGAC"] = 0.0
            for key in ("AGGAX", "ALDAX", "AMGGAX"):
                self.defaults[key] = 1.0 - self.parameters["AEXX"]

            if self.parameters.get("AEXX", self.defaults["AEXX"]) == 1.0:
                self.defaults["ALDAC"] = 0.0
                self.defaults["AMGGAC"] = 0.0

        for key in self.categories["hybrid"]:
            self.defaults[key].update(
                {
                    "operation": "==" if isinstance(self.defaults[key]["value"], bool) else "approx",
                }
            )

    def update_fft_params(self) -> None:
        # NGX/Y/Z and NGXF/YF/ZF. Not checked if not in INCAR file (as this means the VASP default was used).
        if any(i for i in ["NGX", "NGY", "NGZ", "NGXF", "NGYF", "NGZF"] if i in self.incar.keys()):
            self.valid_values["ENMAX"] = max(
                self.parameters["ENMAX"], self.input_set.incar.get("ENCUT", self.defaults["ENMAX"])
            )
            (
                [self.valid_values["NGX"], self.valid_values["NGY"], self.valid_values["NGZ"]],
                [self.valid_values["NGXF"], self.valid_values["NGYF"], self.valid_values["NGZF"]],
            ) = self.input_set.calculate_ng(custom_encut=self.valid_values["ENMAX"])

            for direction in ["X", "Y", "Z"]:
                for mod in ["", "F"]:
                    key = f"NG{direction}{mod}"
                    self.valid_values[key] = int(self.valid_values[key] * self._fft_grid_tolerance)

                    self.defaults[key] = {
                        "value": self.valid_values[key],
                        "tag": "fft",
                        "operation": "<=",
                        "comment": (
                            "This likely means the number FFT grid points was modified by the user. "
                            "If not, please create a GitHub issue."
                        ),
                    }

    def update_lmaxmix_and_lmaxtau(self) -> None:
        """
        Check that LMAXMIX and LMAXTAU are above the required value. Also ensure that they are not greater than 6,
        as that is inadvisable according to the VASP development team (as of writing this in August 2023).
        """

        self.valid_values["LMAXMIX"] = self.input_set.incar.get("LMAXMIX", self.defaults["LMAXMIX"]["value"])
        self.valid_values["LMAXTAU"] = min(self.valid_values["LMAXMIX"] + 2, 6)
        self.parameters["LMAXTAU"] = self.incar.get("LMAXTAU", self.defaults["LMAXTAU"]["value"])

        for key in ["LMAXMIX", "LMAXTAU"]:
            if key == "LMAXTAU" and (
                self.incar.get("METAGGA", self.defaults["METAGGA"]["value"]) in ["--", None, "None"]
            ):
                continue

            if self.parameters[key] > 6:
                self.defaults[key]["comment"] = (
                    f"From empirical testing, using {key} > 6 appears "
                    "to introduce computational instabilities, and is currently inadvisable "
                    "according to the VASP development team."
                )

            # Either add to reasons or warnings depending on task type (as this affects NSCF calcs the most)
            # @ Andrew Rosen, is this an adequate check? Or should we somehow also be checking for cases where
            # a previous SCF calc used the wrong LMAXMIX too?
            if (
                not any(
                    [
                        self._task_type == TaskType.NSCF_Uniform,
                        self._task_type == TaskType.NSCF_Line,
                        self.parameters["ICHARG"] >= 10,
                    ]
                )
                and key == "LMAXMIX"
            ):
                self.defaults[key]["severity"] = "warning"

            if self.valid_values[key] < 6:
                self.valid_values[key] = [self.valid_values[key], 6]
                self.defaults[key]["operation"] = ["<=", ">="]
                self.parameters[key] = [self.parameters[key], self.parameters[key]]
            else:
                self.defaults[key]["operation"] = "=="

    def update_smearing(self, bandgap_tol=1.0e-4) -> None:
        bandgap = self.task_doc.output.bandgap

        smearing_comment = f"This is flagged as incorrect because this calculation had a bandgap of {round(bandgap,3)}"

        # bandgap_tol taken from
        # https://github.com/materialsproject/pymatgen/blob/1f98fa21258837ac174105e00e7ac8563e119ef0/pymatgen/io/vasp/sets.py#L969
        if bandgap > bandgap_tol:
            self.valid_values["ISMEAR"] = [-5, 0]
            self.valid_values["SIGMA"] = 0.05
        else:
            self.valid_values["ISMEAR"] = [0, 1, 2]
            if self.parameters["NSW"] == 0:
                # ISMEAR = -5 is valid for metals *only* when doing static calc
                self.valid_values["ISMEAR"].append(-5)
                smearing_comment += " and is a static calculation"
            else:
                smearing_comment += " and is a non-static calculation"
            self.valid_values["SIGMA"] = 0.2

        smearing_comment += "."

        for key in ["ISMEAR", "SIGMA"]:
            self.defaults[key]["comment"] = smearing_comment

        # TODO: improve logic for SIGMA reasons given in the case where you have a material that should have been relaxed with ISMEAR in [-5, 0], but used ISMEAR in [1,2].
        # Because in such cases, the user wouldn't need to update the SIGMA if they use tetrahedron smearing.
        if self.parameters["ISMEAR"] in [-5, -4, -2]:
            self.defaults["SIGMA"]["warning"] = (
                f"SIGMA is not being directly checked, as an ISMEAR of {self.parameters['ISMEAR']} "
                f"is being used. However, given the bandgap of {round(bandgap,3)}, "
                f"the maximum SIGMA used should be {self.valid_values['ISMEAR']} "
                "if using an ISMEAR *not* in [-5, -4, -2]."
            )

        else:
            self.defaults["SIGMA"]["operation"] = ">="

        # Also check if SIGMA is too large according to the VASP wiki,
        # which occurs when the entropy term in the energy is greater than 1 meV/atom.
        self.parameters["electronic entropy"] = -1e20
        for ionic_step in self._ionic_steps:
            electronic_steps = ionic_step["electronic_steps"]
            for elec_step in electronic_steps:
                if elec_step.get("eentropy", None):
                    self.parameters["electronic entropy"] = max(
                        self.parameters["electronic entropy"], abs(elec_step["eentropy"] / self.structure.num_sites)
                    )

        self.valid_values["electronic entropy"] = 0.001
        self.defaults["electronic entropy"] = {
            "value": 0.0,
            "tag": "smearing",
            "comment": (
                "The entropy term (T*S) in the energy was "
                f"{round(1000 * self.parameters['electronic entropy'], 3)} meV/atom, "
                " which is greater than the "
                f"{round(1000 * self.valid_values['electronic entropy'], 1)} meV/atom "
                f"maximum suggested in the VASP wiki. Thus, SIGMA should be decreased."
            ),
            "alias": "SIGMA",
            "operation": ">=",
        }

    def _get_default_nbands(self):
        """
        This method is copied from the `estimate_nbands` function in pymatgen.io.vasp.sets.py.
        The only noteworthy changes (should) be that there is no reliance on the user setting
        up the psp_resources for pymatgen.
        """
        nions = len(self.structure.sites)

        if self.parameters["ISPIN"] == 1:
            nmag = 0
        else:
            nmag = sum(self.parameters.get("MAGMOM", [0]))
            nmag = np.floor((nmag + 1) / 2)

        possible_val_1 = np.floor((self._NELECT + 2) / 2) + max(np.floor(nions / 2), 3)
        possible_val_2 = np.floor(self._NELECT * 0.6)

        default_nbands = max(possible_val_1, possible_val_2) + nmag

        if self.parameters.get("LNONCOLLINEAR"):
            default_nbands = default_nbands * 2

        if self.parameters.get("NPAR"):
            default_nbands = (
                np.floor((default_nbands + self.parameters["NPAR"] - 1) / self.parameters["NPAR"])
            ) * self.parameters["NPAR"]

        return int(default_nbands)

    def update_electronic_params(self):
        # ENINI. Only check for IALGO = 48 / ALGO = VeryFast, as this is the only algo that uses this tag.
        if self.parameters["IALGO"] == 48:
            self.valid_values["ENINI"] = self.valid_values["ENMAX"]
            self.defaults["ENINI"]["operation"] = "<="

        # ENAUG. Should only be checked for calculations where the relevant MP input set specifies ENAUG.
        # In that case, ENAUG should be the same or greater than in valid_input_set.
        if self.input_set.incar.get("ENAUG"):
            self.defaults["ENAUG"]["operation"] = "<="

        # IALGO.
        self.valid_values["IALGO"] = [38, 58, 68, 90]
        # TODO: figure out if 'normal' algos every really affect results other than convergence

        # NELECT.
        self._NELECT = self.parameters.get("NELECT")
        # Do not check for non-neutral NELECT if NELECT is not in the INCAR
        if self.incar.get("NELECT"):
            self.valid_values["NELECT"] = 0.0
            try:
                self.parameters["NELECT"] = float(self.calcs_reversed[0]["output"]["structure"]._charge)
                self.defaults["NELECT"].update(
                    {
                        "operation": "approx",
                        "comment": (
                            f"This causes the structure to have a charge of {self.parameters['NELECT']}. "
                            f"NELECT should be set to {self._NELECT + self.parameters['NELECT']} instead."
                        ),
                    }
                )
            except Exception:
                self.defaults["NELECT"].update(
                    {
                        "operation": "auto fail",
                        "alias": "NELECT / POTCAR",
                        "comment": "Issue checking whether NELECT was changed to make "
                        "the structure have a non-zero charge. This is likely due to the "
                        "directory not having a POTCAR file.",
                    }
                )

        # NBANDS.
        min_nbands = int(np.ceil(self._NELECT / 2) + 1)
        self.defaults["NBANDS"] = {
            "value": self._get_default_nbands(),
            "operation": ["<=", ">="],
            "tag": "electronic",
            "comment": (
                "Too many or too few bands can lead to unphysical electronic structure "
                "(see https://github.com/materialsproject/custodian/issues/224 "
                "for more context.)"
            ),
        }
        self.valid_values["NBANDS"] = [min_nbands, 4 * self.defaults["NBANDS"]["value"]]
        self.parameters["NBANDS"] = [self.parameters["NBANDS"] for _ in range(2)]

    def update_ionic_params(self):
        # IBRION.
        self.valid_values["IBRION"] = [-1, 1, 2]
        if self.input_set.incar.get("IBRION"):
            self.valid_values["IBRION"] = [self.input_set.incar["IBRION"]]

        # POTIM.
        if self.parameters["IBRION"] in [1, 2, 3, 5, 6]:
            # POTIM is only used for some IBRION values
            self.valid_values["POTIM"] = 5
            self.defaults["POTIM"].update(
                {
                    "operation": ">=",
                    "comment": "POTIM being so high will likely lead to erroneous results.",
                }
            )

            # Check for large changes in energy between ionic steps (usually indicates too high POTIM)
            if self._nionic_steps > 1:
                # Do not use `e_0_energy`, as there is a bug in the vasprun.xml when printing that variable
                # (see https://www.vasp.at/forum/viewtopic.php?t=16942 for more details).
                cur_ionic_step_energies = [ionic_step["e_fr_energy"] for ionic_step in self._ionic_steps]
                cur_ionic_step_energy_gradient = np.diff(cur_ionic_step_energies)
                self.parameters["max gradient"] = max(np.abs(cur_ionic_step_energy_gradient)) / self.structure.num_sites
                self.valid_values["max gradient"] = 1
                self.defaults["max gradient"] = {
                    "value": None,
                    "tag": "ionic",
                    "alias": "POTIM",
                    "operation": ">=",
                    "comment": (
                        f"The energy changed by a maximum of {self.valid_values['max gradient']} eV/atom "
                        "between ionic steps, which is greater than the maximum "
                        f"allowed of {self.valid_values['max gradient']} eV/atom. "
                        "This indicates that POTIM is too high."
                    ),
                }

        # EDIFFG.
        # Should be the same or smaller than in valid_input_set. Force-based cutoffs (not in every
        # every MP-compliant input set, but often have comparable or even better results) will also be accepted
        # I am **NOT** confident that this should be the final check. Perhaps I need convincing (or perhaps it does indeed need to be changed...)
        # TODO:    -somehow identify if a material is a vdW structure, in which case force-convergence should maybe be more strict?
        self.defaults["EDIFFG"] = {
            "value": 10 * self.valid_values["EDIFF"],
            "category": "ionic",
            "comment": "CONVERGENCE --> Structure is not converged according to EDIFFG.",
        }

        self.valid_values["EDIFFG"] = self.input_set.incar.get("EDIFFG", self.defaults["EDIFFG"]["value"])

        if self.task_doc.output.forces is None:
            self.defaults["EDIFFG"]["warning"] = "TaskDoc does not contain output forces!"
            self.defaults["EDIFFG"]["operation"] = "auto fail"

        elif self.parameters["EDIFFG"] < 0.0:
            self.parameters["EDIFFG"] = [np.linalg.norm(force_on_atom) for force_on_atom in self.task_doc.output.forces]
            self.valid_values["EDIFFG"] = [abs(self.valid_values["EDIFFG"]) for _ in range(self.structure.num_sites)]
            self.defaults["EDIFFG"].update(
                {
                    "value": [self.defaults["EDIFFG"]["value"] for _ in range(self.structure.num_sites)],
                    "operation": [">=" for _ in range(self.structure.num_sites)],
                }
            )

        elif self.parameters["EDIFFG"] > 0.0 and self.parameters["NSW"] > 0 and self._nionic_steps > 1:
            energy_of_last_step = self.calcs_reversed[0]["output"]["ionic_steps"][-1]["e_0_energy"]
            energy_of_second_to_last_step = self.calcs_reversed[0]["output"]["ionic_steps"][-2]["e_0_energy"]
            self.parameters["EDIFFG"] = abs(energy_of_last_step - energy_of_second_to_last_step)
            self.defaults["EDIFFG"]["operation"] = ">="


class BasicValidator:
    """Lightweight validator class to handle majority of parameter checking."""
     # MK: unclear. Is the above docstring accurate? It seems like all checks use this, right?

    # avoiding dunder methods because these raise too many NotImplemented's
    operations: tuple[str, ...] = ("==", ">", ">=", "<", "<=", "in", "approx", "auto fail")

    def __init__(self, global_tolerance=1.0e-4) -> None:
        self.tolerance = global_tolerance

    def _comparator(self, x: Any, operation: str, y: Any, **kwargs) -> bool:
        if operation == "auto fail":
            c = False
        elif operation == "==":
            c = x == y
        elif operation == ">":
            c = x > y
        elif operation == ">=":
            c = x >= y
        elif operation == "<":
            c = x < y
        elif operation == "<=":
            c = x <= y
        elif operation == "in":
            c = y in x
        elif operation == "approx":
            c = isclose(x, y, **kwargs)
        return c

    def _check_parameter(
        self,
        error_list: list[str],
        input_tag: str,
        current_value: Any,
        reference_value: Any,
        operation: str,
        tolerance: float | None = None,
        append_comments: str | None = None,
    ) -> None:
        """Determine validity of parameter subject to specified operation."""

        # Allow for printing different tag than the one used to access values
        # For example, the user sets ENCUT via INCAR, but the value of ENCUT is stored
        # by VASP as ENMAX  
        # MK: is the above comment in the best place?

        append_comments = append_comments or ""

        kwargs: dict[str, Any] = {}
        if operation == "approx" and isinstance(current_value, float):
            kwargs.update({"rel_tol": tolerance or self.tolerance, "abs_tol": 0.0})
        valid_value = self._comparator(reference_value, operation, current_value, **kwargs)

        if not valid_value:
            # reverse the inequality sign because of ordering of input and expected
            # values in reason string
            flipped_operation = operation
            if ">" in operation:
                flipped_operation.replace(">", "<")
            elif "<" in operation:
                flipped_operation.replace("<", ">")

            error_list.append(
                f"INPUT SETTINGS --> {input_tag}: set to {current_value}, but should be "
                f"{flipped_operation} {reference_value}. {append_comments}"
            )

    def check_parameter(
        self,
        reasons: list[str],
        warnings: list[str],
        input_tag: str,
        current_values: Any,
        reference_values: Any,
        operations: str | list[str],
        tolerance: float = None,
        append_comments: str | None = None,
        severity: str = "reason",
    ):
        """Determine validity of parameter subject to possible multiple operations."""

        severity_to_list = {"reason": reasons, "warning": warnings}

        if not isinstance(operations, list):
            operations = [operations]
            current_values = [current_values]
            reference_values = [reference_values]

        if not all(operation in self.operations for operation in operations):
            # MK: is this actually the best way to handle this? Why not raise an error in such cases?
            # Do not validate
            return

        for iop in range(len(operations)):
            self._check_parameter(
                error_list=severity_to_list[severity],
                input_tag=input_tag,
                current_value=current_values[iop],
                reference_value=reference_values[iop],
                operation=operations[iop],
                tolerance=tolerance,
                append_comments=append_comments,
            )
