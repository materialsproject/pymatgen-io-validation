"""Validate VASP INCAR files."""

from __future__ import annotations
import copy
from dataclasses import dataclass
from math import isclose
import numpy as np
from emmet.core.vasp.calc_types.enums import TaskType

from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from emmet.core.vasp.task_valid import TaskDocument
    from emmet.core.tasks import TaskDoc
    from typing import Any, Sequence
    from pymatgen.core import Structure
    from pymatgen.io.vasp.sets import VaspInputSet

# TODO: fix ISIF getting overwritten by MP input set.


@dataclass
class CheckIncar:
    """
    Check calculation parameters related to INCAR input tags.

    defaults : dict
        Dict of default parameters.
    fft_grid_tolerance: float
        Directly calculating the FFT grid defaults from VASP is actually impossible
        without information on how VASP was compiled. This is because the FFT
        params generated depend on whatever fft library used. So instead, we do our
        best to calculate the FFT grid defaults and then lower it artificially by
        `fft_grid_tolerance`. So if the user's FFT grid parameters are greater than
        (fft_grid_tolerance x slightly-off defaults), the FFT params are marked
        as valid.
    """

    defaults: dict | None = None
    fft_grid_tolerance: float | None = None

    def check(
        self,
        reasons: list[str],
        warnings: list[str],
        valid_input_set: VaspInputSet,
        task_doc: TaskDoc | TaskDocument,
        parameters: dict[str, Any],
        structure: Structure,
        vasp_version: Sequence[int],
        task_type: TaskType,
    ) -> None:
        """
        Check calculation parameters related to INCAR input tags.

        This first updates any parameter with a specified update method.
        In practice, each INCAR tag in `vasp_defaults.yaml` has a "tag"
        attribute. If there is an update method
        `UpdateParameterValues.update_{tag.replace(" ","_")}_params`,
        all parameters with that tag will be updated.

        Then after all missing values in the supplied parameters (padding
        implicit values with their defaults), this checks whether the user-
        supplied/-parsed parameters satisfy a set of operations against the
        reference valid input set.

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
        task_doc : emmet.core TaskDoc | TaskDocument
            Task document parsed from the calculation directory.
        parameters : dict[str,Any]
            Dict of user-supplied/-parsed INCAR parameters.
        structure: Pymatgen Structure
            Structure used in the calculation.
        vasp_version: Sequence[int]
            Vasp version, e.g., 6.4.1 could be represented as (6,4,1)
        task_type : TaskType
            Task type of the calculation.
        """

        # Instantiate class that updates "dynamic" INCAR tags
        # (like NBANDS, or hybrid-related parameters)

        working_params = UpdateParameterValues(
            parameters=parameters,
            defaults=self.defaults,
            input_set=valid_input_set,
            structure=structure,
            task_doc=task_doc,
            vasp_version=vasp_version,
            task_type=task_type,
            fft_grid_tolerance=self.fft_grid_tolerance,
        )
        # Update values in the working parameters by adding
        # defaults to unspecified INCAR tags, and by updating
        # any INCAR tag that has a specified update method
        working_params.update_parameters_and_defaults()

        # Validate each parameter in the set of working parameters
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
            # if key == "ISIF":
            #    print("batz",key,working_params.parameters[key], parameters.get("ISIF"),working_params.valid_values[key],working_params.defaults[key])


class UpdateParameterValues:
    """
    Update a set of parameters according to supplied rules and defaults.

    While many of the parameters in VASP need only a simple check to determine
    validity with respect to Materials Project parameters, a few are updated
    by VASP when other conditions are met.

    For example, if LDAU is set to False, none of the various LDAU* (LDAUU, LDAUJ,
    LDAUL) tags need validation. But if LDAU is set to true, these all need validation.

    Another example is NBANDS, which VASP computes from a set of input tags.
    This class allows one to mimic the VASP NBANDS functionality for computing
    NBANDS dynamically, and update both the current and reference values for NBANDs.

    To do this in a simple, automatic fashion, each parameter in `vasp_defaults.yaml` has
    a "tag" field. To update a set of parameters with a given tag, one then adds a function
    to `GetParams` called `update_{tag.replace(" ","_")}_params`. For example, the "dft plus u"
    tag has an update function called `update_dft_plus_u_params`. If no such update method
    exists, that tag is skipped.

    Attrs
    ---------
    _default_schema : dict[str,Any]
        The schema of an entry in the dict of default values (`self.defaults`).
        This pads any missing entries in the set of parameters defaults with
        sensible default values.
    """

    _default_schema: dict[str, Any] = {
        "value": None,
        "tag": None,
        "operation": None,
        "comment": None,
        "tolerance": 1.0e-4,
        "severity": "reason",
    }

    def __init__(
        self,
        parameters: dict[str, Any],
        defaults: dict[str, dict],
        input_set: VaspInputSet,
        structure: Structure,
        task_doc: TaskDoc | TaskDocument,
        vasp_version: Sequence[int],
        task_type: TaskType,
        fft_grid_tolerance: float,
    ) -> None:
        """
        Given a set of user parameters, a valid input set, and defaults, update certain tagged parameters.

        Parameters
        -----------
        parameters: dict[str,Any]
            Dict of user-supplied parameters.
        defaults: dict
            Dict of default values for parameters, tags for parameters, and the operation to check them.
        input_set: VaspInputSet
            Valid input set to compare parameters to.
        structure: Pymatgen Structure
            Structure used in the calculation.
        task_doc : emmet.core TaskDoc | TaskDocument
            Task document parsed from the calculation directory.
        vasp_version: Sequence[int]
            Vasp version, e.g., 6.4.1 could be represented as (6,4,1)
        task_type : TaskType
            Task type of the calculation.
        fft_grid_tolerance: float
            See docstr for `_check_incar`. The FFT grid generation has been udpated frequently
            in VASP, and determining the grid density with absolute certainty is not possible.
            This tolerance allows for "reasonable" discrepancies from the ideal FFT grid density.
        """

        self.parameters = copy.deepcopy(parameters)
        self.defaults = copy.deepcopy(defaults)
        self.input_set = input_set
        self.vasp_version = vasp_version
        self.structure = structure
        self.valid_values: dict[str, Any] = {}

        # convert to dict for consistent handling of attrs
        self.task_doc = task_doc.model_dump()
        # Add some underscored values for convenience
        self._fft_grid_tolerance = fft_grid_tolerance
        self._calcs_reversed = self.task_doc["calcs_reversed"]
        self._incar = self._calcs_reversed[0]["input"]["incar"]
        self._ionic_steps = self._calcs_reversed[0]["output"]["ionic_steps"]
        self._nionic_steps = len(self._ionic_steps)
        self._potcar = self._calcs_reversed[0]["input"]["potcar_spec"]
        self._task_type = task_type

    def update_parameters_and_defaults(self) -> None:
        """Update user parameters and defaults for tags with a specified update method."""

        self.categories: dict[str, list[str]] = {}
        for key in self.defaults:
            if self.defaults[key]["tag"] not in self.categories:
                self.categories[self.defaults[key]["tag"]] = []
            self.categories[self.defaults[key]["tag"]].append(key)

        tag_order = [key.replace(" ", "_") for key in self.categories if key != "post_init"] + ["post_init"]
        # add defaults to parameters from the incar as needed
        self.add_defaults_to_parameters(valid_values_source=self.input_set.incar)
        # collect list of tags in parameter defaults
        for tag in tag_order:
            # check to see if update method for that tag exists, and if so, run it
            update_method_str = f"update_{tag}_params"
            if hasattr(self, update_method_str):
                self.__getattribute__(update_method_str)()

        # add defaults to parameters from the defaults as needed
        self.add_defaults_to_parameters()

        for key in self.defaults:
            for attr in self._default_schema:
                self.defaults[key][attr] = self.defaults[key].get(attr, self._default_schema[attr])

    def add_defaults_to_parameters(self, valid_values_source: dict | None = None) -> None:
        """
        Update parameters with initial defaults.

        Parameters
        -----------
        valid_values_source : dict or None (default)
            If None, update missing values in `self.parameters` and `self.valid_values`
            using self.defaults. If a dict, update from that dict.
        """
        valid_values_source = valid_values_source or self.valid_values

        for key in self.defaults:
            self.parameters[key] = self.parameters.get(key, self.defaults[key]["value"])
            self.valid_values[key] = valid_values_source.get(key, self.defaults[key]["value"])

    def update_dft_plus_u_params(self) -> None:
        """Update DFT+U params."""
        if not self.parameters["LDAU"]:
            return

        for key in self.categories["dft plus u"]:
            valid_value = self.input_set.incar.get(key, self.defaults[key]["value"])

            # TODO: ADK: is LDAUTYPE usually specified as a list??
            if key == "LDAUTYPE":
                self.parameters[key] = (
                    self.parameters[key][0] if isinstance(self.parameters[key], list) else self.parameters[key]
                )
                self.valid_values[key] = valid_value[0] if isinstance(valid_value, list) else valid_value
            else:
                self.parameters[key] = self._incar.get(key, self.defaults[key]["value"])
            self.defaults[key]["operation"] = "=="

    def update_symmetry_params(self) -> None:
        """Update symmetry-related parameters."""
        # ISYM.
        self.valid_values["ISYM"] = [-1, 0, 1, 2]
        if self.parameters["LHFCALC"]:
            self.defaults["ISYM"]["value"] = 3
            self.valid_values["ISYM"].append(3)
        self.defaults["ISYM"]["operation"] = "in"

        # SYMPREC.
        # custodian will set SYMPREC to a maximum of 1e-3 (as of August 2023)
        self.valid_values["SYMPREC"] = 1e-3
        self.defaults["SYMPREC"].update(
            {
                "operation": "<=",
                "comment": (
                    "If you believe that this SYMPREC value is necessary "
                    "(perhaps this calculation has a very large cell), please create "
                    "a GitHub issue and we will consider to admit your calculations."
                ),
            }
        )

    def update_startup_params(self) -> None:
        """Update VASP initialization parameters."""
        self.valid_values["ISTART"] = [0, 1, 2]

        # ICHARG.
        if self.input_set.incar.get("ICHARG", self.defaults["ICHARG"]["value"]) < 10:
            self.valid_values["ICHARG"] = 9  # should be <10 (SCF calcs)
            self.defaults["ICHARG"]["operation"] = "<="
        else:
            self.valid_values["ICHARG"] = self.input_set.incar.get("ICHARG")
            self.defaults["ICHARG"]["operation"] = "=="

    def update_precision_params(self) -> None:
        """Update VASP parameters related to precision."""
        # LREAL.
        # Do NOT use the value for LREAL from the `Vasprun.parameters` object, as VASP changes these values
        # relative to the INCAR. Rather, check the LREAL value in the `Vasprun.incar` object.
        if str(self.input_set.incar.get("LREAL")).upper() in ["AUTO", "A"]:
            self.valid_values["LREAL"] = ["FALSE", "AUTO", "A"]
        elif str(self.input_set.incar.get("LREAL")).upper() in ["FALSE"]:
            self.valid_values["LREAL"] = ["FALSE"]

        self.parameters["LREAL"] = str(self._incar.get("LREAL", self.defaults["LREAL"]["value"])).upper()
        # PREC.
        self.parameters["PREC"] = self.parameters["PREC"].upper()
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
                "operation": ["<=" for _ in self.parameters["ROPT"]],
            }

    def update_misc_special_params(self) -> None:
        """Update miscellaneous parameters that do not fall into another category."""
        # EFERMI. Only available for VASP >= 6.4. Should not be set to a numerical
        # value, as this may change the number of electrons.
        # self.vasp_version = (major, minor, patch)
        if (self.vasp_version[0] >= 6) and (self.vasp_version[1] >= 4):
            # Must check EFERMI in the *incar*, as it is saved as a numerical
            # value after VASP guesses it in the vasprun.xml `parameters`
            # (which would always cause this check to fail, even if the user
            # set EFERMI properly in the INCAR).
            self.parameters["EFERMI"] = self._incar.get("EFERMI", self.defaults["EFERMI"]["value"])
            self.valid_values["EFERMI"] = ["LEGACY", "MIDGAP"]
            self.defaults["EFERMI"]["operation"] = "in"

        # IWAVPR.
        if self._incar.get("IWAVPR"):
            self.parameters["IWAVPR"] = self._incar["IWAVPR"] if self._incar["IWAVPR"] is not None else 0
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
            and len(self._calcs_reversed[0]["output"]["outcar"]["magnetization"]) != self.structure.num_sites
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
                    "value": [self.defaults[key]["value"][0] for _ in self.parameters[key]],
                    "operation": ["==" for _ in self.parameters[key]],
                    "comment": f"{key} should not be set. {aux_str}",
                }
            )
            self.valid_values[key] = self.defaults[key]["value"].copy()

    def update_hybrid_params(self) -> None:
        """Update params related to hybrid functionals."""
        self.valid_values["LHFCALC"] = self.input_set.incar.get("LHFCALC", self.defaults["LHFCALC"]["value"])

        if self.valid_values["LHFCALC"]:
            self.defaults["AEXX"]["value"] = 0.25
            self.parameters["AEXX"] = self.parameters.get("AEXX", self.defaults["AEXX"]["value"])
            self.defaults["AGGAC"]["value"] = 0.0
            for key in ("AGGAX", "ALDAX", "AMGGAX"):
                self.defaults[key]["value"] = 1.0 - self.parameters["AEXX"]

            if self.parameters.get("AEXX", self.defaults["AEXX"]["value"]) == 1.0:
                self.defaults["ALDAC"]["value"] = 0.0
                self.defaults["AMGGAC"]["value"] = 0.0

        for key in self.categories["hybrid"]:
            self.defaults[key]["operation"] = "==" if isinstance(self.defaults[key]["value"], bool) else "approx"

    def update_fft_params(self) -> None:
        """Update ENCUT and parameters related to the FFT grid."""

        # ensure that ENCUT is appropriately updated
        self.valid_values["ENMAX"] = self.input_set.incar.get("ENCUT", self.defaults["ENMAX"])

        grid_keys = {"NGX", "NGXF", "NGY", "NGYF", "NGZ", "NGZF"}
        # NGX/Y/Z and NGXF/YF/ZF. Not checked if not in INCAR file (as this means the VASP default was used).
        if any(i for i in grid_keys if i in self._incar.keys()):
            self.valid_values["ENMAX"] = max(self.parameters["ENMAX"], self.valid_values["ENMAX"])

            (
                [self.valid_values["NGX"], self.valid_values["NGY"], self.valid_values["NGZ"]],
                [self.valid_values["NGXF"], self.valid_values["NGYF"], self.valid_values["NGZF"]],
            ) = self.input_set.calculate_ng(custom_encut=self.valid_values["ENMAX"])

            for key in grid_keys:
                self.valid_values[key] = int(self.valid_values[key] * self._fft_grid_tolerance)

                self.defaults[key] = {
                    "value": self.valid_values[key],
                    "tag": "fft",
                    "operation": ">=",
                    "comment": (
                        "This likely means the number FFT grid points was modified by the user. "
                        "If not, please create a GitHub issue."
                    ),
                }

    def update_density_mixing_params(self) -> None:
        """
        Check that LMAXMIX and LMAXTAU are above the required value.

        Also ensure that they are not greater than 6, as that is inadvisable
        according to the VASP development team (as of August 2023).
        """

        self.valid_values["LMAXMIX"] = self.input_set.incar.get("LMAXMIX", self.defaults["LMAXMIX"]["value"])
        self.valid_values["LMAXTAU"] = min(self.valid_values["LMAXMIX"] + 2, 6)
        self.parameters["LMAXTAU"] = self._incar.get("LMAXTAU", self.defaults["LMAXTAU"]["value"])

        for key in ["LMAXMIX", "LMAXTAU"]:
            if key == "LMAXTAU" and (
                self._incar.get("METAGGA", self.defaults["METAGGA"]["value"]) in ["--", None, "None"]
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
                self.defaults[key]["operation"] = [">=", "<="]
                self.parameters[key] = [self.parameters[key], self.parameters[key]]
            else:
                self.defaults[key]["operation"] = "=="

    def update_smearing_params(self, bandgap_tol=1.0e-4) -> None:
        """
        Update parameters related to Fermi-level smearing.

        This is based on the final bandgap obtained in the calc.
        """
        bandgap = self.task_doc["output"]["bandgap"]

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

        # TODO: improve logic for SIGMA reasons given in the case where you
        # have a material that should have been relaxed with ISMEAR in [-5, 0],
        # but used ISMEAR in [1,2]. Because in such cases, the user wouldn't
        # need to update the SIGMA if they use tetrahedron smearing.
        if self.parameters["ISMEAR"] in [-5, -4, -2]:
            self.defaults["SIGMA"]["warning"] = (
                f"SIGMA is not being directly checked, as an ISMEAR of {self.parameters['ISMEAR']} "
                f"is being used. However, given the bandgap of {round(bandgap,3)}, "
                f"the maximum SIGMA used should be {self.valid_values['ISMEAR']} "
                "if using an ISMEAR *not* in [-5, -4, -2]."
            )

        else:
            self.defaults["SIGMA"]["operation"] = "<="

        # Also check if SIGMA is too large according to the VASP wiki,
        # which occurs when the entropy term in the energy is greater than 1 meV/atom.
        self.parameters["ELECTRONIC ENTROPY"] = -1e20
        for ionic_step in self._ionic_steps:
            electronic_steps = ionic_step["electronic_steps"]
            for elec_step in electronic_steps:
                if elec_step.get("eentropy", None):
                    self.parameters["ELECTRONIC ENTROPY"] = max(
                        self.parameters["ELECTRONIC ENTROPY"], abs(elec_step["eentropy"] / self.structure.num_sites)
                    )

        convert_eV_to_meV = 1000
        self.parameters["ELECTRONIC ENTROPY"] = round(self.parameters["ELECTRONIC ENTROPY"] * convert_eV_to_meV, 3)
        self.valid_values["ELECTRONIC ENTROPY"] = 0.001 * convert_eV_to_meV

        self.defaults["ELECTRONIC ENTROPY"] = {
            "value": 0.0,
            "tag": "smearing",
            "comment": (
                "The entropy term (T*S) in the energy is suggested to be less than "
                f"{round(self.valid_values['ELECTRONIC ENTROPY'], 1)} meV/atom "
                f"in the VASP wiki. Thus, SIGMA should be decreased."
            ),
            "operation": "<=",
        }

    def _get_default_nbands(self):
        """
        Estimate number of bands used in calculation.

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
        """Update electronic self-consistency parameters."""
        # ENINI. Only check for IALGO = 48 / ALGO = VeryFast, as this is the only algo that uses this tag.
        if self.parameters["IALGO"] == 48:
            self.valid_values["ENINI"] = self.valid_values["ENMAX"]
            self.defaults["ENINI"]["operation"] = ">="

        # ENAUG. Should only be checked for calculations where the relevant MP input set specifies ENAUG.
        # In that case, ENAUG should be the same or greater than in valid_input_set.
        if self.input_set.incar.get("ENAUG"):
            self.defaults["ENAUG"]["operation"] = ">="

        # IALGO.
        self.valid_values["IALGO"] = [38, 58, 68, 90]
        # TODO: figure out if 'normal' algos every really affect results other than convergence

        # NELECT.
        self._NELECT = self.parameters.get("NELECT")
        # Do not check for non-neutral NELECT if NELECT is not in the INCAR
        if self._incar.get("NELECT"):
            self.valid_values["NELECT"] = 0.0
            try:
                self.parameters["NELECT"] = float(self._calcs_reversed[0]["output"]["structure"]._charge)
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
            "operation": [">=", "<="],
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
        """Update parameters related to ionic relaxation."""

        self.valid_values["ISIF"] = 2

        # IBRION.
        self.valid_values["IBRION"] = [-1, 1, 2]
        if self.input_set.incar.get("IBRION"):
            if self.input_set.incar.get("IBRION") not in self.valid_values["IBRION"]:
                self.valid_values["IBRION"] = [self.input_set.incar["IBRION"]]

        # POTIM.
        if self.parameters["IBRION"] in [1, 2, 3, 5, 6]:
            # POTIM is only used for some IBRION values
            self.valid_values["POTIM"] = 5
            self.defaults["POTIM"].update(
                {
                    "operation": "<=",
                    "comment": "POTIM being so high will likely lead to erroneous results.",
                }
            )

            # Check for large changes in energy between ionic steps (usually indicates too high POTIM)
            if self._nionic_steps > 1:
                # Do not use `e_0_energy`, as there is a bug in the vasprun.xml when printing that variable
                # (see https://www.vasp.at/forum/viewtopic.php?t=16942 for more details).
                cur_ionic_step_energies = [ionic_step["e_fr_energy"] for ionic_step in self._ionic_steps]
                cur_ionic_step_energy_gradient = np.diff(cur_ionic_step_energies)
                self.parameters["MAX ENERGY GRADIENT"] = round(
                    max(np.abs(cur_ionic_step_energy_gradient)) / self.structure.num_sites, 3
                )
                self.valid_values["MAX ENERGY GRADIENT"] = 1
                self.defaults["MAX ENERGY GRADIENT"] = {
                    "value": None,
                    "tag": "ionic",
                    "operation": "<=",
                    "comment": (
                        f"The energy changed by a maximum of {self.parameters['MAX ENERGY GRADIENT']} eV/atom "
                        "between ionic steps; this indicates that POTIM is too high."
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
        }

        self.valid_values["EDIFFG"] = self.input_set.incar.get("EDIFFG", self.defaults["EDIFFG"]["value"])
        self.defaults["EDIFFG"][
            "comment"
        ] = f"Hence, structure is not converged according to EDIFFG, which should be {self.valid_values['EDIFFG']} or better."

        if self.task_doc["output"]["forces"] is None:
            self.defaults["EDIFFG"]["warning"] = "TaskDoc does not contain output forces!"
            self.defaults["EDIFFG"]["operation"] = "auto fail"

        elif self.valid_values["EDIFFG"] < 0.0:
            self.parameters["EDIFFG"] = round(
                max([np.linalg.norm(force_on_atom) for force_on_atom in self.task_doc["output"]["forces"]]), 3
            )

            self.valid_values["EDIFFG"] = abs(self.valid_values["EDIFFG"])
            self.defaults["EDIFFG"].update(
                {
                    "value": self.defaults["EDIFFG"]["value"],
                    "operation": "<=",
                    "alias": "MAX FINAL FORCE MAGNITUDE",
                }
            )

        # the latter two checks just ensure the code does not error by indexing out of range
        elif self.valid_values["EDIFFG"] > 0.0 and self._nionic_steps > 1:
            energy_of_last_step = self._calcs_reversed[0]["output"]["ionic_steps"][-1]["e_0_energy"]
            energy_of_second_to_last_step = self._calcs_reversed[0]["output"]["ionic_steps"][-2]["e_0_energy"]
            self.parameters["EDIFFG"] = abs(energy_of_last_step - energy_of_second_to_last_step)
            self.defaults["EDIFFG"]["operation"] = "<="
            self.defaults["EDIFFG"]["alias"] = "ENERGY CHANGE BETWEEN LAST TWO IONIC STEPS"

    def update_post_init_params(self):
        """Update any params that depend on other params being set/updated."""

        # EBREAK
        # vasprun includes default EBREAK value, so we check ionic steps
        # to see if the user set a value for EBREAK.
        # Note that the NBANDS estimation differs from VASP's documentation,
        # so we can't check the vasprun value directly
        if self._incar.get("EBREAK"):
            self.defaults["EBREAK"]["value"] = self.defaults["EDIFF"]["value"] / (
                4.0 * self.defaults["NBANDS"]["value"]
            )
            self.defaults["EBREAK"]["operation"] = "auto fail"


class BasicValidator:
    """
    Compare test and reference values according to one or more operations.

    Parameters
    -----------
    global_tolerance : float = 1.e-4
        Default tolerance for assessing approximate equality via math.isclose

    Attrs
    -----------
    operations : set[str]
        List of acceptable operations, such as "==" for strict equality, or "in" to
        check if a Sequence contains an element
    """

    # avoiding dunder methods because these raise too many NotImplemented's
    operations: set[str | None] = {"==", ">", ">=", "<", "<=", "in", "approx", "auto fail", None}

    def __init__(self, global_tolerance=1.0e-4) -> None:
        """Set math.isclose tolerance"""
        self.tolerance = global_tolerance

    def _comparator(self, lhs: Any, operation: str, rhs: Any, **kwargs) -> bool:
        """
        Compare different values using one of a set of supported operations in self.operations.

        Parameters
        -----------
        lhs : Any
            Left-hand side of the operation.
        operation : str
            Operation acting on rhs from lhs. For example, if operation is ">",
            this returns (lhs > rhs).
        rhs : Any
            Right-hand of the operation.
        kwargs
            If needed, kwargs to pass to operation.
        """
        if operation is None:
            c = True
        elif operation == "auto fail":
            c = False
        elif operation == "==":
            c = lhs == rhs
        elif operation == ">":
            c = lhs > rhs
        elif operation == ">=":
            c = lhs >= rhs
        elif operation == "<":
            c = lhs < rhs
        elif operation == "<=":
            c = lhs <= rhs
        elif operation == "in":
            c = lhs in rhs
        elif operation == "approx":
            c = isclose(lhs, rhs, **kwargs)
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
        """
        Determine validity of parameter subject to a single specified operation.

        Parameters
        -----------
        error_list : list[str]
            A list of error/warning strings to update if a check fails.
        input_tag : str
            The name of the input tag which is being checked.
        current_value : Any
            The test value.
        reference_value : Any
            The value to compare the test value to.
        operation : str
            A valid operation in self.operations. For example, if operation = "<=",
            this checks `current_value <= reference_value` (note order of values).
        tolerance : float or None (default)
            If None and operation == "approx", default tolerance to self.tolerance.
            Otherwise, use the user-supplied tolerance.
        append_comments : str or None (default)
            Additional comments that may be helpful for the user to understand why
            a check failed.
        """

        append_comments = append_comments or ""

        kwargs: dict[str, Any] = {}
        if operation == "approx" and isinstance(current_value, float):
            kwargs.update({"rel_tol": tolerance or self.tolerance, "abs_tol": 0.0})
        valid_value = self._comparator(current_value, operation, reference_value, **kwargs)

        if not valid_value:
            error_list.append(
                f"INPUT SETTINGS --> {input_tag}: is {current_value}, but should be "
                f"{'' if operation == 'auto fail' else operation + ' '}{reference_value}."
                f"{' ' if len(append_comments) > 0 else ''}{append_comments}"
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
        severity: Literal["reason", "warning"] = "reason",
    ) -> None:
        """
        Determine validity of parameter according to one or more operations.

        Parameters
        -----------
        reasons : list[str]
            A list of error strings to update if a check fails. These are higher
            severity and would deprecate a calculation.
        warnings : list[str]
            A list of warning strings to update if a check fails. These are lower
            severity and would flag a calculation for possible review.
        input_tag : str
            The name of the input tag which is being checked.
        current_values : Any
            The test value(s). If multiple operations are specified, must be a Sequence
            of test values.
        reference_values : Any
            The value(s) to compare the test value(s) to. If multiple operations are
            specified, must be a Sequence of reference values.
        operations : str
            One or more valid operations in self.operations. For example, if operations = "<=",
            this checks `current_values <= reference_values` (note order of values).
            Or, if operations == ["<=", ">"], this checks
            ```
            (
                (current_values[0] <= reference_values[0])
                and (current_values[1] > reference_values[1])
            )
            ```
        tolerance : float or None (default)
            Tolerance to use in math.isclose if any of operations is "approx". Defaults
            to self.tolerance.
        append_comments : str or None (default)
            Additional comments that may be helpful for the user to understand why
            a check failed.
        severity : Literal["reason", "warning"]
            If a calculation fails, the severity of failure. Directs output to
            either reasons or warnings.
        """

        severity_to_list = {"reason": reasons, "warning": warnings}

        if not isinstance(operations, list):
            operations = [operations]
            current_values = [current_values]
            reference_values = [reference_values]

        unknown_operations = {operation for operation in operations if operation not in self.operations}
        if len(unknown_operations) > 0:
            raise ValueError("Unknown operations:\n  " + ", ".join([f"{uo}" for uo in unknown_operations]))

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
