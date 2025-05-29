"""Validate VASP INCAR files."""

from __future__ import annotations
import numpy as np
from pydantic import Field

from pymatgen.io.validation.common import SETTINGS, BaseValidator
from pymatgen.io.validation.vasp_defaults import InputCategory, VaspParam

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import Any
    from pymatgen.io.validation.common import VaspFiles

# TODO: fix ISIF getting overwritten by MP input set.


class CheckIncar(BaseValidator):
    """
    Check calculation parameters related to INCAR input tags.

    Because this class checks many INCAR tags in sequence, while it
    inherits from the `pymatgen.io.validation.common.BaseValidator`
    class, it also defines a custom `check` method.

    Note about `fft_grid_tolerance`:
    Directly calculating the FFT grid defaults from VASP is actually impossible
    without information on how VASP was compiled. This is because the FFT
    params generated depend on whatever fft library used. So instead, we do our
    best to calculate the FFT grid defaults and then lower it artificially by
    `fft_grid_tolerance`. So if the user's FFT grid parameters are greater than
    (fft_grid_tolerance x slightly-off defaults), the FFT params are marked
    as valid.
    """

    name: str = "Check INCAR tags"
    fft_grid_tolerance: float | None = Field(
        SETTINGS.VASP_FFT_GRID_TOLERANCE, description="Tolerance for determining sufficient density of FFT grid."
    )
    bandgap_tol: float = Field(1.0e-4, description="Tolerance for assuming a material has no gap.")

    def check(self, vasp_files: VaspFiles, reasons: list[str], warnings: list[str]) -> None:
        """
        Check calculation parameters related to INCAR input tags.

        This first updates any parameter with a specified update method.
        In practice, each INCAR tag in `VASP` has a "tag" attribute.
        If there is an update method
        `UpdateParameterValues.update_{tag}_params`,
        all parameters with that tag will be updated.

        Then after all missing values in the supplied parameters (padding
        implicit values with their defaults), this checks whether the user-
        supplied/-parsed parameters satisfy a set of operations against the
        reference valid input set.
        """

        # Instantiate class that updates "dynamic" INCAR tags
        # (like NBANDS, or hybrid-related parameters)

        user_incar_params, valid_incar_params = self.update_parameters_and_defaults(vasp_files)
        msgs = {
            "reason": reasons,
            "warning": warnings,
        }
        # Validate each parameter in the set of working parameters
        for vasp_param in self.vasp_defaults.values():
            if self.fast and len(reasons) > 0:
                # fast check: stop checking whenever a single check fails
                break
            resp = vasp_param.check(user_incar_params[vasp_param.name], valid_incar_params[vasp_param.name])
            msgs[vasp_param.severity].extend(resp.get(vasp_param.severity, []))

    def update_parameters_and_defaults(self, vasp_files: VaspFiles) -> tuple[dict[str, Any], dict[str, Any]]:
        """Update a set of parameters according to supplied rules and defaults.

        While many of the parameters in VASP need only a simple check to determine
        validity with respect to Materials Project parameters, a few are updated
        by VASP when other conditions are met.

        For example, if LDAU is set to False, none of the various LDAU* (LDAUU, LDAUJ,
        LDAUL) tags need validation. But if LDAU is set to true, these all need validation.

        Another example is NBANDS, which VASP computes from a set of input tags.
        This class allows one to mimic the VASP NBANDS functionality for computing
        NBANDS dynamically, and update both the current and reference values for NBANDs.

        To do this in a simple, automatic fashion, each parameter in `VASP_DEFAULTS` has
        a "tag" field. To update a set of parameters with a given tag, one then adds a function
        to `GetParams` called `update_{tag}_params`. For example, the "dft plus u"
        tag has an update function called `update_dft_plus_u_params`. If no such update method
        exists, that tag is skipped.
        """

        # Note: we cannot make these INCAR objects because INCAR checks certain keys
        # Like LREAL and forces them to bool when the validator expects them to be str
        user_incar = {k: v for k, v in vasp_files.user_input.incar.as_dict().items() if not k.startswith("@")}
        ref_incar = {k: v for k, v in vasp_files.valid_input_set.incar.as_dict().items() if not k.startswith("@")}

        self.add_defaults_to_parameters(user_incar, ref_incar)
        # collect list of tags in parameter defaults
        for tag in InputCategory.__members__:
            # check to see if update method for that tag exists, and if so, run it
            update_method_str = f"_update_{tag}_params"
            if hasattr(self, update_method_str):
                getattr(self, update_method_str)(user_incar, ref_incar, vasp_files)

        # add defaults to parameters from the defaults as needed
        self.add_defaults_to_parameters(user_incar, ref_incar)

        return user_incar, ref_incar

    def add_defaults_to_parameters(self, *incars) -> None:
        """
        Update parameters with initial defaults.
        """
        for key in self.vasp_defaults:
            for incar in incars:
                if (incar.get(key)) is None:
                    incar[key] = self.vasp_defaults[key].value

    def _update_dft_plus_u_params(self, user_incar: dict, ref_incar: dict, vasp_files: VaspFiles) -> None:
        """Update DFT+U params."""
        if not user_incar["LDAU"]:
            return

        for key in [v.name for v in self.vasp_defaults.values() if v.tag == "dft_plus_u"]:

            # TODO: ADK: is LDAUTYPE usually specified as a list??
            if key == "LDAUTYPE":
                user_incar[key] = user_incar[key][0] if isinstance(user_incar[key], list) else user_incar[key]
                if isinstance(ref_incar[key], list):
                    ref_incar[key] = ref_incar[key][0]

            self.vasp_defaults[key].operation = "=="

    def _update_symmetry_params(self, user_incar: dict, ref_incar: dict, vasp_files: VaspFiles) -> None:
        """Update symmetry-related parameters."""
        # ISYM.
        ref_incar["ISYM"] = [-1, 0, 1, 2]
        if user_incar["LHFCALC"]:
            self.vasp_defaults["ISYM"].value = 3
            ref_incar["ISYM"].append(3)
        self.vasp_defaults["ISYM"].operation = "in"

        # SYMPREC.
        # custodian will set SYMPREC to a maximum of 1e-3 (as of August 2023)
        ref_incar["SYMPREC"] = 1e-3
        self.vasp_defaults["SYMPREC"].operation = "<="
        self.vasp_defaults["SYMPREC"].comment = (
            "If you believe that this SYMPREC value is necessary "
            "(perhaps this calculation has a very large cell), please create "
            "a GitHub issue and we will consider to admit your calculations."
        )

    def _update_startup_params(self, user_incar: dict, ref_incar: dict, vasp_files: VaspFiles) -> None:
        """Update VASP initialization parameters."""
        ref_incar["ISTART"] = [0, 1, 2]

        # ICHARG.
        if ref_incar.get("ICHARG", self.vasp_defaults["ICHARG"].value) < 10:
            ref_incar["ICHARG"] = 9  # should be <10 (SCF calcs)
            self.vasp_defaults["ICHARG"].operation = "<="
        else:
            ref_incar["ICHARG"] = ref_incar.get("ICHARG")
            self.vasp_defaults["ICHARG"].operation = "=="

    def _update_precision_params(self, user_incar: dict, ref_incar: dict, vasp_files: VaspFiles) -> None:
        """Update VASP parameters related to precision."""
        # LREAL.
        # Do NOT use the value for LREAL from the `Vasprun.parameters` object, as VASP changes these values
        # relative to the INCAR. Rather, check the LREAL value in the `Vasprun.incar` object.
        if str(ref_incar.get("LREAL")).upper() in ["AUTO", "A"]:
            ref_incar["LREAL"] = ["FALSE", "AUTO", "A"]
        elif str(ref_incar.get("LREAL")).upper() in ["FALSE"]:
            ref_incar["LREAL"] = ["FALSE"]

        user_incar["LREAL"] = str(user_incar["LREAL"]).upper()
        # PREC.
        user_incar["PREC"] = user_incar["PREC"].upper()
        if ref_incar["PREC"].upper() in {"ACCURATE", "HIGH"}:
            ref_incar["PREC"] = ["ACCURATE", "ACCURA", "HIGH"]
        else:
            raise ValueError("Validation code check for PREC tag needs to be updated to account for a new input set!")
        self.vasp_defaults["PREC"].operation = "in"

        # ROPT. Should be better than or equal to default for the PREC level.
        # This only matters if projectors are done in real-space.
        # Note that if the user sets LREAL = Auto in their Incar, it will show
        # up as "True" in the `parameters` object (hence we use the `parameters` object)
        # According to VASP wiki (https://www.vasp.at/wiki/index.php/ROPT), only
        # the magnitude of ROPT is relevant for precision.
        if user_incar["LREAL"] == "TRUE":
            # this only matters if projectors are done in real-space.
            cur_prec = user_incar["PREC"].upper()
            ropt_default = {
                "NORMAL": -5e-4,
                "ACCURATE": -2.5e-4,
                "ACCURA": -2.5e-4,
                "LOW": -0.01,
                "MED": -0.002,
                "HIGH": -4e-4,
            }
            user_incar["ROPT"] = [abs(value) for value in user_incar.get("ROPT", [ropt_default[cur_prec]])]
            self.vasp_defaults["ROPT"] = VaspParam(
                name="ROPT",
                value=[abs(ropt_default[cur_prec]) for _ in user_incar["ROPT"]],
                tag="startup",
                operation=["<=" for _ in user_incar["ROPT"]],
            )

    def _update_misc_special_params(self, user_incar: dict, ref_incar: dict, vasp_files: VaspFiles) -> None:
        """Update miscellaneous parameters that do not fall into another category."""
        # EFERMI. Only available for VASP >= 6.4. Should not be set to a numerical
        # value, as this may change the number of electrons.
        # self.vasp_version = (major, minor, patch)
        if vasp_files.vasp_version and (vasp_files.vasp_version[0] >= 6) and (vasp_files.vasp_version[1] >= 4):
            # Must check EFERMI in the *incar*, as it is saved as a numerical
            # value after VASP guesses it in the vasprun.xml `parameters`
            # (which would always cause this check to fail, even if the user
            # set EFERMI properly in the INCAR).
            ref_incar["EFERMI"] = ["LEGACY", "MIDGAP"]
            self.vasp_defaults["EFERMI"].operation = "in"

        # IWAVPR.
        if user_incar.get("IWAVPR"):
            self.vasp_defaults["IWAVPR"].operation = "=="
            self.vasp_defaults["IWAVPR"].comment = (
                "VASP discourages users from setting the IWAVPR tag (as of July 2023)."
            )

        # LCORR.
        if user_incar["IALGO"] != 58:
            self.vasp_defaults["LCORR"].operation = "=="

        if (
            user_incar["ISPIN"] == 2
            and vasp_files.outcar
            and len(getattr(vasp_files.outcar, "magnetization", [])) != vasp_files.user_input.structure.num_sites
        ):
            self.vasp_defaults["LORBIT"].update(
                {
                    "operation": "auto fail",
                    "comment": (
                        "Magnetization values were not written "
                        "to the OUTCAR. This is usually due to LORBIT being set to None or "
                        "False for calculations with ISPIN=2."
                    ),
                }
            )

        if (
            vasp_files.vasp_version
            and (vasp_files.vasp_version[0] < 6)
            and user_incar["LORBIT"] >= 11
            and user_incar["ISYM"]
        ):
            self.vasp_defaults["LORBIT"]["warning"] = (
                "For LORBIT >= 11 and ISYM = 2 the partial charge densities are not correctly symmetrized and can result "
                "in different charges for symmetrically equivalent partial charge densities. This issue is fixed as of version "
                ">=6. See the vasp wiki page for LORBIT for more details."
            )

        # RWIGS and VCA - do not set
        for key in ["RWIGS", "VCA"]:
            aux_str = ""
            if key == "RWIGS":
                aux_str = " This is because it will change some outputs like the magmom on each site."
            self.vasp_defaults[key] = VaspParam(
                name=key,
                value=[self.vasp_defaults[key].value[0] for _ in user_incar[key]],
                tag="misc_special",
                operation=["==" for _ in user_incar[key]],
                comment=f"{key} should not be set. {aux_str}",
            )
            ref_incar[key] = self.vasp_defaults[key].value

    def _update_hybrid_params(self, user_incar: dict, ref_incar: dict, vasp_files: VaspFiles) -> None:
        """Update params related to hybrid functionals."""
        ref_incar["LHFCALC"] = ref_incar.get("LHFCALC", self.vasp_defaults["LHFCALC"].value)

        if ref_incar["LHFCALC"]:
            self.vasp_defaults["AEXX"].value = 0.25
            user_incar["AEXX"] = user_incar.get("AEXX", self.vasp_defaults["AEXX"].value)
            self.vasp_defaults["AGGAC"].value = 0.0
            for key in ("AGGAX", "ALDAX", "AMGGAX"):
                self.vasp_defaults[key].value = 1.0 - user_incar["AEXX"]

            if user_incar.get("AEXX", self.vasp_defaults["AEXX"].value) == 1.0:
                self.vasp_defaults["ALDAC"].value = 0.0
                self.vasp_defaults["AMGGAC"].value = 0.0

        for key in [v.name for v in self.vasp_defaults.values() if v.tag == "hybrid"]:
            self.vasp_defaults[key]["operation"] = "==" if isinstance(self.vasp_defaults[key].value, bool) else "approx"

    def _update_fft_params(self, user_incar: dict, ref_incar: dict, vasp_files: VaspFiles) -> None:
        """Update ENCUT and parameters related to the FFT grid."""

        # ensure that ENCUT is appropriately updated
        user_incar["ENMAX"] = user_incar.get("ENCUT", getattr(vasp_files.vasprun, "parameters", {}).get("ENMAX"))

        ref_incar["ENMAX"] = vasp_files.valid_input_set.incar.get("ENCUT", self.vasp_defaults["ENMAX"])

        grid_keys = {"NGX", "NGXF", "NGY", "NGYF", "NGZ", "NGZF"}
        # NGX/Y/Z and NGXF/YF/ZF. Not checked if not in INCAR file (as this means the VASP default was used).
        if any(i for i in grid_keys if i in user_incar.keys()):
            enmaxs = [user_incar["ENMAX"], ref_incar["ENMAX"]]
            ref_incar["ENMAX"] = max([v for v in enmaxs if v < float("inf")])

            if fft_grid := vasp_files.valid_input_set._calculate_ng(custom_encut=ref_incar["ENMAX"]):
                (
                    [
                        ref_incar["NGX"],
                        ref_incar["NGY"],
                        ref_incar["NGZ"],
                    ],
                    [
                        ref_incar["NGXF"],
                        ref_incar["NGYF"],
                        ref_incar["NGZF"],
                    ],
                ) = fft_grid

            for key in grid_keys:
                ref_incar[key] = int(ref_incar[key] * self.fft_grid_tolerance)

                self.vasp_defaults[key] = VaspParam(
                    name=key,
                    value=ref_incar[key],
                    tag="fft",
                    operation=">=",
                    comment=(
                        "This likely means the number FFT grid points was modified by the user. "
                        "If not, please create a GitHub issue."
                    ),
                )

    def _update_density_mixing_params(self, user_incar: dict, ref_incar: dict, vasp_files: VaspFiles) -> None:
        """
        Check that LMAXMIX and LMAXTAU are above the required value.

        Also ensure that they are not greater than 6, as that is inadvisable
        according to the VASP development team (as of August 2023).
        """

        ref_incar["LMAXTAU"] = min(ref_incar["LMAXMIX"] + 2, 6)

        for key in ["LMAXMIX", "LMAXTAU"]:
            if key == "LMAXTAU" and user_incar["METAGGA"] in ["--", None, "None"]:
                continue

            if user_incar[key] > 6:
                self.vasp_defaults[key].comment = (
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
                        vasp_files.run_type == "nonscf",
                        user_incar["ICHARG"] >= 10,
                    ]
                )
                and key == "LMAXMIX"
            ):
                self.vasp_defaults[key].severity = "warning"

            if ref_incar[key] < 6:
                ref_incar[key] = [ref_incar[key], 6]
                self.vasp_defaults[key].operation = [">=", "<="]
                user_incar[key] = [user_incar[key], user_incar[key]]
            else:
                self.vasp_defaults[key].operation = "=="

    def _update_smearing_params(self, user_incar: dict, ref_incar: dict, vasp_files: VaspFiles) -> None:
        """
        Update parameters related to Fermi-level smearing.

        This is based on the final bandgap obtained in the calc.
        """
        if vasp_files.bandgap is not None:

            smearing_comment = (
                f"This is flagged as incorrect because this calculation had a bandgap of {round(vasp_files.bandgap,3)}"
            )

            # bandgap_tol taken from
            # https://github.com/materialsproject/pymatgen/blob/1f98fa21258837ac174105e00e7ac8563e119ef0/pymatgen/io/vasp/sets.py#L969
            if vasp_files.bandgap > self.bandgap_tol:
                ref_incar["ISMEAR"] = [-5, 0]
                ref_incar["SIGMA"] = 0.05
            else:
                ref_incar["ISMEAR"] = [-1, 0, 1, 2]
                if user_incar["NSW"] == 0:
                    # ISMEAR = -5 is valid for metals *only* when doing static calc
                    ref_incar["ISMEAR"].append(-5)
                    smearing_comment += " and is a static calculation"
                else:
                    smearing_comment += " and is a non-static calculation"
                ref_incar["SIGMA"] = 0.2

            smearing_comment += "."

            for key in ["ISMEAR", "SIGMA"]:
                self.vasp_defaults[key].comment = smearing_comment

            if user_incar["ISMEAR"] not in [-5, -4, -2]:
                self.vasp_defaults["SIGMA"].operation = "<="

        else:
            # These are generally applicable in all cases. Loosen check to warning.
            ref_incar["ISMEAR"] = [-1, 0]
            if vasp_files.run_type == "static":
                ref_incar["ISMEAR"] += [-5]
            elif vasp_files.run_type == "relax":
                self.vasp_defaults["ISMEAR"].comment = (
                    "Performing relaxations in metals with the tetrahedron method "
                    "may lead to significant errors in forces. To enable this check, "
                    "supply a vasprun.xml file."
                )
                self.vasp_defaults["ISMEAR"].severity = "warning"

        # Also check if SIGMA is too large according to the VASP wiki,
        # which occurs when the entropy term in the energy is greater than 1 meV/atom.
        user_incar["ELECTRONIC ENTROPY"] = -1e20
        if vasp_files.vasprun:
            for ionic_step in vasp_files.vasprun.ionic_steps:
                if eentropy := ionic_step["electronic_steps"][-1].get("eentropy"):
                    user_incar["ELECTRONIC ENTROPY"] = max(
                        user_incar["ELECTRONIC ENTROPY"],
                        abs(eentropy / vasp_files.user_input.structure.num_sites),
                    )

            convert_eV_to_meV = 1000
            user_incar["ELECTRONIC ENTROPY"] = round(user_incar["ELECTRONIC ENTROPY"] * convert_eV_to_meV, 3)
            ref_incar["ELECTRONIC ENTROPY"] = 0.001 * convert_eV_to_meV

            self.vasp_defaults["ELECTRONIC ENTROPY"] = VaspParam(
                name="ELECTRONIC ENTROPY",
                value=0.0,
                tag="smearing",
                comment=(
                    "The entropy term (T*S) in the energy is suggested to be less than "
                    f"{round(ref_incar['ELECTRONIC ENTROPY'], 1)} meV/atom "
                    f"in the VASP wiki. Thus, SIGMA should be decreased."
                ),
                operation="<=",
            )

    def _get_default_nbands(self, nelect: float, user_incar: dict, vasp_files: VaspFiles):
        """
        Estimate number of bands used in calculation.

        This method is copied from the `estimate_nbands` function in pymatgen.io.vasp.sets.py.
        The only noteworthy changes (should) be that there is no reliance on the user setting
        up the psp_resources for pymatgen.
        """
        nions = len(vasp_files.user_input.structure.sites)

        if user_incar["ISPIN"] == 1:
            nmag = 0
        else:
            nmag = sum(user_incar.get("MAGMOM", [0]))
            nmag = np.floor((nmag + 1) / 2)

        possible_val_1 = np.floor((nelect + 2) / 2) + max(np.floor(nions / 2), 3)
        possible_val_2 = np.floor(nelect * 0.6)

        default_nbands = max(possible_val_1, possible_val_2) + nmag

        if user_incar.get("LNONCOLLINEAR"):
            default_nbands = default_nbands * 2

        if vasp_files.vasprun and (npar := vasp_files.vasprun.parameters.get("NPAR")):
            default_nbands = (np.floor((default_nbands + npar - 1) / npar)) * npar

        return int(default_nbands)

    def _update_electronic_params(self, user_incar: dict, ref_incar: dict, vasp_files: VaspFiles):
        """Update electronic self-consistency parameters."""
        # ENINI. Only check for IALGO = 48 / ALGO = VeryFast, as this is the only algo that uses this tag.
        if user_incar["IALGO"] == 48:
            ref_incar["ENINI"] = ref_incar["ENMAX"]
            self.vasp_defaults["ENINI"].operation = ">="

        # ENAUG. Should only be checked for calculations where the relevant MP input set specifies ENAUG.
        # In that case, ENAUG should be the same or greater than in valid_input_set.
        if ref_incar.get("ENAUG") and not np.isinf(ref_incar["ENAUG"]):
            self.vasp_defaults["ENAUG"].operation = ">="

        # IALGO.
        ref_incar["IALGO"] = [38, 58, 68, 90]
        # TODO: figure out if 'normal' algos every really affect results other than convergence

        # NELECT.
        # Do not check for non-neutral NELECT if NELECT is not in the INCAR
        if vasp_files.vasprun and (nelect := vasp_files.vasprun.parameters.get("NELECT")):
            ref_incar["NELECT"] = 0.0
            try:
                user_incar["NELECT"] = float(vasp_files.vasprun.final_structure._charge or 0.0)
                self.vasp_defaults["NELECT"].operation = "approx"
                self.vasp_defaults["NELECT"].comment = (
                    f"This causes the structure to have a charge of {user_incar['NELECT']}. "
                    f"NELECT should be set to {nelect + user_incar['NELECT']} instead."
                )
            except Exception:
                self.vasp_defaults["NELECT"] = VaspParam(
                    name="NELECT",
                    value=None,
                    tag="electronic",
                    operation="auto fail",
                    severity="warning",
                    alias="NELECT / POTCAR",
                    comment=(
                        "Issue checking whether NELECT was changed to make "
                        "the structure have a non-zero charge. This is likely due to the "
                        "directory not having a POTCAR file."
                    ),
                )

            # NBANDS.
            min_nbands = int(np.ceil(nelect / 2) + 1)
            self.vasp_defaults["NBANDS"] = VaspParam(
                name="NBANDS",
                value=self._get_default_nbands(nelect, user_incar, vasp_files),
                tag="electronic",
                operation=[">=", "<="],
                comment=(
                    "Too many or too few bands can lead to unphysical electronic structure "
                    "(see https://github.com/materialsproject/custodian/issues/224 "
                    "for more context.)"
                ),
            )
            ref_incar["NBANDS"] = [min_nbands, 4 * self.vasp_defaults["NBANDS"].value]
            user_incar["NBANDS"] = [vasp_files.vasprun.parameters.get("NBANDS") for _ in range(2)]

    def _update_ionic_params(self, user_incar: dict, ref_incar: dict, vasp_files: VaspFiles):
        """Update parameters related to ionic relaxation."""

        ref_incar["ISIF"] = 2

        # IBRION.
        ref_incar["IBRION"] = [-1, 1, 2]
        if (inp_set_ibrion := vasp_files.valid_input_set.incar.get("IBRION")) and inp_set_ibrion not in ref_incar[
            "IBRION"
        ]:
            ref_incar["IBRION"].append(inp_set_ibrion)

        ionic_steps = []
        if vasp_files.vasprun is not None:
            ionic_steps = vasp_files.vasprun.ionic_steps

        # POTIM.
        if user_incar["IBRION"] in [1, 2, 3, 5, 6]:
            # POTIM is only used for some IBRION values
            ref_incar["POTIM"] = 5
            self.vasp_defaults["POTIM"].operation = "<="
            self.vasp_defaults["POTIM"].comment = "POTIM being so high will likely lead to erroneous results."

            # Check for large changes in energy between ionic steps (usually indicates too high POTIM)
            if len(ionic_steps) > 1:
                # Do not use `e_0_energy`, as there is a bug in the vasprun.xml when printing that variable
                # (see https://www.vasp.at/forum/viewtopic.php?t=16942 for more details).
                cur_ionic_step_energies = [ionic_step["e_fr_energy"] for ionic_step in ionic_steps]
                cur_ionic_step_energy_gradient = np.diff(cur_ionic_step_energies)
                user_incar["MAX ENERGY GRADIENT"] = round(
                    max(np.abs(cur_ionic_step_energy_gradient)) / vasp_files.user_input.structure.num_sites,
                    3,
                )
                ref_incar["MAX ENERGY GRADIENT"] = 1
                self.vasp_defaults["MAX ENERGY GRADIENT"] = VaspParam(
                    name="MAX ENERGY GRADIENT",
                    value=None,
                    tag="ionic",
                    operation="<=",
                    comment=(
                        f"The energy changed by a maximum of {user_incar['MAX ENERGY GRADIENT']} eV/atom "
                        "between ionic steps; this indicates that POTIM is too high."
                    ),
                )

        if not ionic_steps:
            return

        # EDIFFG.
        # Should be the same or smaller than in valid_input_set. Force-based cutoffs (not in every
        # every MP-compliant input set, but often have comparable or even better results) will also be accepted
        # I am **NOT** confident that this should be the final check. Perhaps I need convincing (or perhaps it does indeed need to be changed...)
        # TODO:    -somehow identify if a material is a vdW structure, in which case force-convergence should maybe be more strict?
        self.vasp_defaults["EDIFFG"] = VaspParam(
            name="EDIFFG",
            value=10 * ref_incar["EDIFF"],
            tag="ionic",
            operation=None,
        )

        ref_incar["EDIFFG"] = ref_incar.get("EDIFFG", self.vasp_defaults["EDIFFG"].value)
        self.vasp_defaults["EDIFFG"].comment = (
            "The structure is not force-converged according "
            f"to |EDIFFG|={abs(ref_incar['EDIFFG'])} (or smaller in magnitude)."
        )

        if ionic_steps[-1].get("forces") is None:
            self.vasp_defaults["EDIFFG"].comment = (
                "vasprun.xml does not contain forces, cannot check force convergence."
            )
            self.vasp_defaults["EDIFFG"].severity = "warning"
            self.vasp_defaults["EDIFFG"].operation = "auto fail"

        elif ref_incar["EDIFFG"] < 0.0 and (vrun_forces := ionic_steps[-1].get("forces")) is not None:
            user_incar["EDIFFG"] = round(
                max([np.linalg.norm(force_on_atom) for force_on_atom in vrun_forces]),
                3,
            )

            ref_incar["EDIFFG"] = abs(ref_incar["EDIFFG"])
            self.vasp_defaults["EDIFFG"] = VaspParam(
                name="EDIFFG",
                value=self.vasp_defaults["EDIFFG"].value,
                tag="ionic",
                operation="<=",
                alias="MAX FINAL FORCE MAGNITUDE",
            )

        # the latter two checks just ensure the code does not error by indexing out of range
        elif ref_incar["EDIFFG"] > 0.0 and vasp_files.vasprun and len(ionic_steps) > 1:
            energy_of_last_step = ionic_steps[-1]["e_0_energy"]
            energy_of_second_to_last_step = ionic_steps[-2]["e_0_energy"]
            user_incar["EDIFFG"] = abs(energy_of_last_step - energy_of_second_to_last_step)
            self.vasp_defaults["EDIFFG"].operation = "<="
            self.vasp_defaults["EDIFFG"].alias = "ENERGY CHANGE BETWEEN LAST TWO IONIC STEPS"

    def _update_post_init_params(self, user_incar: dict, ref_incar: dict, vasp_files: VaspFiles):
        """Update any params that depend on other params being set/updated."""

        # EBREAK
        # vasprun includes default EBREAK value, so we check ionic steps
        # to see if the user set a value for EBREAK.
        # Note that the NBANDS estimation differs from VASP's documentation,
        # so we can't check the vasprun value directly
        if user_incar.get("EBREAK"):
            self.vasp_defaults["EBREAK"].value = self.vasp_defaults["EDIFF"].value / (
                4.0 * self.vasp_defaults["NBANDS"].value
            )
            self.vasp_defaults["EBREAK"].operation = "auto fail"
