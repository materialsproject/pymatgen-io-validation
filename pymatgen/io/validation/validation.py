"""Main module for VASP calculation validation"""
from __future__ import annotations

from datetime import datetime
from importlib.resources import files as import_resource_files
import numpy as np
from pydantic import Field
from pydantic.types import ImportString  # replacement for PyObject
from pathlib import Path
from monty.serialization import loadfn

from pymatgen.io.vasp.sets import VaspInputSet

# TODO: AK: why MPMetalRelaxSet
# TODO: MK: because more kpoints are needed for metals given the more complicated Fermi surfaces, and MPMetalRelaxSet uses more kpoints
from pymatgen.io.vasp.sets import MPMetalRelaxSet

from emmet.core.tasks import TaskDoc
from emmet.core.base import EmmetBaseModel
from emmet.core.mpid import MPID
from emmet.core.vasp.calc_types.enums import CalcType, TaskType
from emmet.core.vasp.calc_types import (
    RunType,
    calc_type as emmet_calc_type,
    run_type as emmet_run_type,
    task_type as emmet_task_type,
)

from pymatgen.io.validation.check_incar import _check_incar
from pymatgen.io.validation.check_common_errors import _check_common_errors
from pymatgen.io.validation.check_kpoints_kspacing import _check_kpoints_kspacing
from pymatgen.io.validation.settings import IOValidationSettings

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pymatgen.io.vasp.inputs import Potcar

SETTINGS = IOValidationSettings()

_pmg_potcar_summary_stats = loadfn(import_resource_files("pymatgen.io.vasp") / "potcar-summary-stats.json.bz2")

# TODO: check for surface/slab calculations. Especially necessary for external calcs.
# TODO: implement check to make sure calcs are within some amount (e.g. 250 meV) of the convex hull in the MPDB


class ValidationDoc(EmmetBaseModel):
    """
    Validation document for a VASP calculation
    """

    task_id: MPID = Field(..., description="The task_id for this validation document")

    valid: bool = Field(False, description="Whether this task is valid or not")

    last_updated: datetime = Field(
        description="Last updated date for this document",
        default_factory=datetime.utcnow,
    )

    reasons: list[str] = Field(None, description="List of deprecation tags detailing why this task isn't valid")

    warnings: list[str] = Field([], description="List of potential warnings about this calculation")

    # data: Dict = Field(
    #     description="Dictionary of data used to perform validation."
    #     " Useful for post-mortem analysis"
    # )

    class Config:  # noqa
        extra = "allow"

    @classmethod
    def from_task_doc(
        cls,
        task_doc: TaskDoc,
        input_sets: dict[str, ImportString] = SETTINGS.VASP_DEFAULT_INPUT_SETS,
        potcar_summary_stats: dict[str, ImportString] = _pmg_potcar_summary_stats,
        kpts_tolerance: float = SETTINGS.VASP_KPTS_TOLERANCE,
        allow_kpoint_shifts: bool = SETTINGS.VASP_ALLOW_KPT_SHIFT,
        allow_explicit_kpoint_mesh: str | bool = SETTINGS.VASP_ALLOW_EXPLICIT_KPT_MESH,
        fft_grid_tolerance: float = SETTINGS.VASP_FFT_GRID_TOLERANCE,
        num_ionic_steps_to_avg_drift_over: float = SETTINGS.VASP_NUM_IONIC_STEPS_FOR_DRIFT,
        max_allowed_scf_gradient: float = SETTINGS.VASP_MAX_SCF_GRADIENT,
    ) -> ValidationDoc:
        """
        Determines if a calculation is valid based on expected input parameters from a pymatgen inputset

        Args:
            task_doc: the task document to process
            input_sets: a dictionary of task_types -> pymatgen input set for validation
            potcar_summary_stats: Dictionary of potcar summary data. Mapping is calculation type -> potcar symbol -> summary data.
            kpts_tolerance: the tolerance to allow kpts to lag behind the input set settings
            allow_kpoint_shifts: Whether to consider a task valid if kpoints are shifted by the user
            allow_explicit_kpoint_mesh: Whether to consider a task valid if the user defines an explicit kpoint mesh
            fft_grid_tolerance: Relative tolerance for FFT grid parameters to still be a valid
            num_ionic_steps_to_avg_drift_over: Number of ionic steps to average over when validating drift forces
            max_allowed_scf_gradient: maximum uphill gradient allowed for SCF steps after the
                initial equillibriation period. Note this is in eV per atom.
        """

        bandgap = task_doc.output.bandgap
        calcs_reversed = task_doc.calcs_reversed
        calcs_reversed = [
            calc.model_dump() for calc in calcs_reversed
        ]  # convert to dictionary to use built-in `.get()` method

        parameters = (
            task_doc.input.parameters
        )  # used for most input tag checks (as this is more reliable than examining the INCAR file directly in most cases)
        incar = calcs_reversed[0]["input"][
            "incar"
        ]  # used for INCAR tag checks where you need to look at the actual INCAR (semi-rare)
        if task_doc.orig_inputs is None:
            orig_inputs = {}
        else:
            orig_inputs = task_doc.orig_inputs.model_dump()
            if orig_inputs["kpoints"] is not None:
                orig_inputs["kpoints"] = orig_inputs["kpoints"].as_dict()

        ionic_steps = calcs_reversed[0]["output"]["ionic_steps"]
        nionic_steps = len(ionic_steps)

        potcars = calcs_reversed[0]["input"]["potcar_spec"]

        calc_type = _get_calc_type(calcs_reversed, orig_inputs)
        task_type = _get_task_type(calcs_reversed, orig_inputs)
        run_type = _get_run_type(calcs_reversed)

        if allow_explicit_kpoint_mesh == "auto":
            if "NSCF" in calc_type.name:
                allow_explicit_kpoint_mesh = True
            else:
                allow_explicit_kpoint_mesh = False

        # Why was this lingering here?
        # task_doc.chemsys

        vasp_version = [int(x) for x in calcs_reversed[0]["vasp_version"].split(".")]

        if calcs_reversed[0].get("input", {}).get("structure", None):
            structure = calcs_reversed[0]["input"]["structure"]
        else:
            structure = task_doc.input.structure or task_doc.output.structure

        reasons = []
        # data = {}  # type: ignore
        warnings: list[str] = []

        if f"{run_type}".upper() not in {"GGA", "GGA+U", "PBE", "PBE+U", "R2SCAN"}:
            reasons.append(f"FUNCTIONAL --> Functional {run_type} not currently accepted.")

        try:
            valid_input_set = _get_input_set(run_type, task_type, calc_type, structure, input_sets, bandgap)
        except Exception as e:
            reasons.append(
                "NO MATCHING MP INPUT SET --> no matching MP input set was found. If you believe this to be a mistake, please create a GitHub issue."
            )
            valid_input_set = None
            print(f"Error while finding MP input set: {e}.")

        if parameters == {} or parameters is None:
            reasons.append(
                "CAN NOT PROPERLY PARSE CALCULATION --> Issue parsing input parameters from the vasprun.xml file."
            )
        elif valid_input_set:
            # Get subset of POTCAR summary stats to validate calculation

            psp_subset = potcar_summary_stats.get(valid_input_set._config_dict["POTCAR_FUNCTIONAL"], {})

            valid_potcar_summary_stats = {}  # type: ignore
            for element in structure.composition.remove_charges().as_dict():
                potcar_symbol = valid_input_set._config_dict["POTCAR"][element]
                for titel_no_spc in psp_subset:
                    for psp in psp_subset[titel_no_spc]:
                        if psp["symbol"] == potcar_symbol:
                            if titel_no_spc not in valid_potcar_summary_stats:
                                valid_potcar_summary_stats[titel_no_spc] = []
                            valid_potcar_summary_stats[titel_no_spc].append(psp)

            if potcar_summary_stats:
                _check_potcars(reasons, warnings, potcars, valid_potcar_summary_stats)

            # TODO: check for surface/slab calculations!!!!!!

            reasons = _check_vasp_version(reasons, vasp_version, parameters, incar)

            reasons = _check_common_errors(
                reasons,
                warnings,
                task_doc,
                calcs_reversed,
                parameters,
                incar,
                structure,
                max_allowed_scf_gradient,
                ionic_steps,
                num_ionic_steps_to_avg_drift_over,
            )

            reasons = _check_kpoints_kspacing(
                reasons,
                task_type,
                parameters,
                kpts_tolerance,
                valid_input_set,
                calcs_reversed,
                allow_explicit_kpoint_mesh,
                allow_kpoint_shifts,
                structure,
            )

            reasons = _check_incar(
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
                potcars,
                vasp_version,
                task_type,
                fft_grid_tolerance,
            )

        # Unsure about what might be a better way to do this...
        task_id = task_doc.task_id if (task_doc.task_id is not None) else -1

        validation_doc = ValidationDoc(
            task_id=task_id,
            calc_type=calc_type,
            run_type=run_type,
            task_type=task_type,
            valid=len(reasons) == 0,
            reasons=reasons,
            # data=data,
            warnings=warnings,
        )

        return validation_doc

    @classmethod
    def from_directory(
        cls,
        dir_name: Path | str,
        input_sets: dict[str, ImportString] = SETTINGS.VASP_DEFAULT_INPUT_SETS,
        potcar_summary_stats: dict[str, ImportString] = _pmg_potcar_summary_stats,
        kpts_tolerance: float = SETTINGS.VASP_KPTS_TOLERANCE,
        allow_kpoint_shifts: bool = SETTINGS.VASP_ALLOW_KPT_SHIFT,
        allow_explicit_kpoint_mesh: str | bool = SETTINGS.VASP_ALLOW_EXPLICIT_KPT_MESH,
        fft_grid_tolerance: float = SETTINGS.VASP_FFT_GRID_TOLERANCE,
        num_ionic_steps_to_avg_drift_over: float = SETTINGS.VASP_NUM_IONIC_STEPS_FOR_DRIFT,
        max_allowed_scf_gradient: float = SETTINGS.VASP_MAX_SCF_GRADIENT,
    ) -> ValidationDoc:
        """
        Determines if a calculation is valid based on expected input parameters from a pymatgen inputset

        Args:
            dir_name: the directory containing the calculation files to process
            input_sets: a dictionary of task_types -> pymatgen input set for validation
            potcar_summary_stats: Dictionary of potcar summary data. Mapping is calculation type -> potcar symbol -> summary data.
            kpts_tolerance: the tolerance to allow kpts to lag behind the input set settings
            allow_kpoint_shifts: Whether to consider a task valid if kpoints are shifted by the user
            allow_explicit_kpoint_mesh: Whether to consider a task valid if the user defines an explicit kpoint mesh
            fft_grid_tolerance: Relative tolerance for FFT grid parameters to still be a valid
            num_ionic_steps_to_avg_drift_over: Number of ionic steps to average over when validating drift forces
            max_allowed_scf_gradient: maximum uphill gradient allowed for SCF steps after the
                initial equillibriation period. Note this is in eV per atom.
        """
        try:
            task_doc = TaskDoc.from_directory(
                dir_name=dir_name,
                volumetric_files=(),
            )

            validation_doc = ValidationDoc.from_task_doc(
                task_doc=task_doc,
                input_sets=input_sets,
                potcar_summary_stats=potcar_summary_stats,
                kpts_tolerance=kpts_tolerance,
                allow_kpoint_shifts=allow_kpoint_shifts,
                allow_explicit_kpoint_mesh=allow_explicit_kpoint_mesh,
                fft_grid_tolerance=fft_grid_tolerance,
                num_ionic_steps_to_avg_drift_over=num_ionic_steps_to_avg_drift_over,
                max_allowed_scf_gradient=max_allowed_scf_gradient,
            )

            return validation_doc
        except Exception as e:
            print(e)
            if "no vasp files found" in str(e).lower():
                raise Exception(f"NO CALCULATION FOUND --> {dir_name} is not a VASP calculation directory.")
            else:
                raise Exception(
                    f"CANNOT PARSE CALCULATION --> Issue parsing results. This often means your calculation did not complete. The error stack reads: \n {e}"
                )


def _get_input_set(run_type, task_type, calc_type, structure, input_sets, bandgap):
    # TODO: For every input set key in emmet.core.settings.VASP_DEFAULT_INPUT_SETS,
    #       with "GGA" in it, create an equivalent dictionary item with "PBE" instead.
    # In the mean time, the below workaround is used.
    gga_pbe_structure_opt_calc_types = [
        CalcType.GGA_Structure_Optimization,
        CalcType.GGA_U_Structure_Optimization,
        CalcType.PBE_Structure_Optimization,
        CalcType.PBE_U_Structure_Optimization,
    ]

    # Ensure input sets get proper additional input values
    if "SCAN" in run_type.value:
        valid_input_set: VaspInputSet = input_sets[str(calc_type)](structure, bandgap=bandgap)  # type: ignore

    elif task_type == TaskType.NSCF_Uniform:
        valid_input_set = input_sets[str(calc_type)](structure, mode="uniform")
    elif task_type == TaskType.NSCF_Line:
        valid_input_set = input_sets[str(calc_type)](structure, mode="line")

    elif "dielectric" in str(task_type).lower():
        valid_input_set = input_sets[str(calc_type)](structure, lepsilon=True)

    elif task_type == TaskType.NMR_Electric_Field_Gradient:
        valid_input_set = input_sets[str(calc_type)](structure, mode="efg")
    elif task_type == TaskType.NMR_Nuclear_Shielding:
        valid_input_set = input_sets[str(calc_type)](
            structure, mode="cs"
        )  # Is this correct? Someone more knowledgeable either fix this or remove this comment if it is correct please!

    elif calc_type in gga_pbe_structure_opt_calc_types:
        if bandgap == 0:
            valid_input_set = MPMetalRelaxSet(structure)
        else:
            valid_input_set = input_sets[str(calc_type)](structure)

    else:
        valid_input_set = input_sets[str(calc_type)](structure)

    return valid_input_set


def _check_potcars(
    reasons,
    warnings,
    potcars: Potcar,
    valid_potcar_summary_stats: dict[str, ImportString],
    data_match_tol: float = 1e-6,
):
    """
    Checks to make sure the POTCAR is equivalent to the correct POTCAR from the
    pymatgen input set.
    """

    if potcars is None:
        reasons.append(
            "PSEUDOPOTENTIALS --> Missing POTCAR files. "
            "Alternatively, our potcar checker may have an issue--please create a GitHub issue if you "
            "know your POTCAR exists and can be read by Pymatgen."
        )
        return

    try:
        incorrect_potcars = []
        for potcar in potcars:
            reference_summary_stats = valid_potcar_summary_stats.get(potcar["titel"].replace(" ", ""), [])

            if len(reference_summary_stats) == 0:
                incorrect_potcars.append(potcar["titel"].split(" ")[1])
                continue

            key_match = False
            data_match = False
            for ref_psp in reference_summary_stats:
                key_match = all(
                    set(ref_psp["keywords"][key]) == set(potcar["summary_stats"]["keywords"][key])  # type: ignore
                    for key in ["header", "data"]
                )

                data_diff = [
                    abs(ref_psp["stats"][key][stat] - potcar["summary_stats"]["stats"][key][stat])  # type: ignore
                    for stat in ["MEAN", "ABSMEAN", "VAR", "MIN", "MAX"]
                    for key in ["header", "data"]
                ]
                data_match = all(np.array(data_diff) < data_match_tol)
                if key_match and data_match:
                    break

            if (not key_match) or (not data_match):
                incorrect_potcars.append(potcar["titel"].split(" ")[1])

        if len(incorrect_potcars) > 0:
            # format error string
            incorrect_potcars = [potcar.split("_")[0] for potcar in incorrect_potcars]
            if len(incorrect_potcars) == 2:
                incorrect_potcars = ", ".join(incorrect_potcars[:-1]) + f" and {incorrect_potcars[-1]}"  # type: ignore
            elif len(incorrect_potcars) >= 3:
                incorrect_potcars = ", ".join(incorrect_potcars[:-1]) + "," + f" and {incorrect_potcars[-1]}"  # type: ignore

            reasons.append(
                f"PSEUDOPOTENTIALS --> Incorrect POTCAR files were used for {incorrect_potcars}. "
                "Alternatively, our potcar checker may have an issue--please create a GitHub issue if you "
                "believe the POTCARs used are correct."
            )

    except KeyError as e:
        print(e)
        # Assume it is an old calculation without potcar_spec data and treat it as failing the POTCAR check
        reasons.append(
            "Issue validating POTCARS --> Likely due to an old version of Emmet (wherein potcar summary_stats is not saved in TaskDoc), though other errors have been seen. Hence, it is marked as invalid"
        )


def _check_vasp_version(reasons, vasp_version, parameters, incar):
    if (
        vasp_version[0] == 5
        and (incar.get("METAGGA", None) not in [None, "--", "None"])
        and parameters.get("ISPIN", 1) == 2
    ):
        reasons.append(
            "POTENTIAL BUG --> We believe that there may be a bug with spin-polarized calculations for METAGGAs "
            "in some versions of VASP 5. Please create a new GitHub issue if you believe this "
            "is not the case and we will consider changing this check!"
        )
    elif list(vasp_version) != [5, 4, 4] or vasp_version[0] < 6:
        vasp_version_str = ".".join([str(x) for x in vasp_version])
        reasons.append(
            f"VASP VERSION --> This calculation is using VASP version {vasp_version_str}, "
            "but we only allow versions 5.4.4 and >=6.0.0 (as of July 2023)."
        )
    return reasons


def _get_run_type(calcs_reversed) -> RunType:
    params = calcs_reversed[0].get("input", {}).get("parameters", {})
    incar = calcs_reversed[0].get("input", {}).get("incar", {})
    return emmet_run_type({**params, **incar})


def _get_task_type(calcs_reversed, orig_inputs):
    inputs = calcs_reversed[0].get("input", {}) if len(calcs_reversed) > 0 else orig_inputs
    return emmet_task_type(inputs)


def _get_calc_type(calcs_reversed, orig_inputs):
    inputs = calcs_reversed[0].get("input", {}) if len(calcs_reversed) > 0 else orig_inputs
    params = calcs_reversed[0].get("input", {}).get("parameters", {})
    incar = calcs_reversed[0].get("input", {}).get("incar", {})

    return emmet_calc_type(inputs, {**params, **incar})
