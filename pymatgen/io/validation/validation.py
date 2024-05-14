"""Validate VASP calculations using emmet."""

from __future__ import annotations

from datetime import datetime
from pydantic import Field
from pydantic.types import ImportString  # replacement for PyObject
from pathlib import Path
from monty.serialization import loadfn

from pymatgen.io.vasp.sets import VaspInputSet

# TODO: AK: why MPMetalRelaxSet
# TODO: MK: because more kpoints are needed for metals given the more complicated Fermi surfaces, and MPMetalRelaxSet uses more kpoints
from pymatgen.io.vasp.sets import MPMetalRelaxSet

from emmet.core.tasks import TaskDoc
from emmet.core.vasp.task_valid import TaskDocument
from emmet.core.base import EmmetBaseModel
from emmet.core.mpid import MPID
from emmet.core.utils import jsanitize
from emmet.core.vasp.calc_types.enums import CalcType, TaskType
from emmet.core.vasp.calc_types import (
    RunType,
    calc_type as emmet_calc_type,
    run_type as emmet_run_type,
    task_type as emmet_task_type,
)
from pymatgen.core import Structure
from pymatgen.io.validation.check_incar import CheckIncar
from pymatgen.io.validation.check_common_errors import (
    CheckCommonErrors,
    CheckVaspVersion,
)
from pymatgen.io.validation.check_kpoints_kspacing import CheckKpointsKspacing
from pymatgen.io.validation.check_potcar import CheckPotcar
from pymatgen.io.validation.settings import IOValidationSettings

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import Any

SETTINGS = IOValidationSettings()
_vasp_defaults = loadfn(SETTINGS.VASP_DEFAULTS_FILENAME)

# TODO: check for surface/slab calculations. Especially necessary for external calcs.
# TODO: implement check to make sure calcs are within some amount (e.g. 250 meV) of the convex hull in the MPDB


class ValidationDoc(EmmetBaseModel):
    """
    Validation document for a VASP calculation
    """

    task_id: MPID | None = Field(None, description="The task_id for this validation document")

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

    def model_post_init(self, context: Any) -> None:
        """
        Optionally check whether package versions are up to date with PyPI.

        Parameters
        -----------
        context : .Any
            Has no effect at present, kept to retain structure of pydantic .BaseModel
        """

        self.valid = len(self.reasons) == 0

    class Config:  # noqa
        extra = "allow"

    @classmethod
    def from_task_doc(cls, task_doc: TaskDoc | TaskDocument, **kwargs) -> ValidationDoc:
        """
        Determines if a calculation is valid based on expected input parameters from a pymatgen inputset

        Args:
            task_doc: the task document to process
        Possible kwargs for `from_dict` method:
            input_sets: a dictionary of task_types -> pymatgen input set for validation
            potcar_summary_stats: Dictionary of potcar summary data. Mapping is calculation type -> potcar symbol -> summary data.
            kpts_tolerance: the tolerance to allow kpts to lag behind the input set settings
            allow_kpoint_shifts: Whether to consider a task valid if kpoints are shifted by the user
            allow_explicit_kpoint_mesh: Whether to consider a task valid if the user defines an explicit kpoint mesh
            fft_grid_tolerance: Relative tolerance for FFT grid parameters to still be a valid
            num_ionic_steps_to_avg_drift_over: Number of ionic steps to average over when validating drift forces
            max_allowed_scf_gradient: maximum uphill gradient allowed for SCF steps after the
                initial equillibriation period. Note this is in eV per atom.
            fast : whether to stop validation when any check fails
        """

        if isinstance(task_doc, TaskDocument):
            task_doc = TaskDoc(**task_doc.model_dump())

        return cls.from_dict(jsanitize(task_doc), **kwargs)

    @classmethod
    def from_dict(
        cls,
        task_doc: dict,
        input_sets: dict[str, ImportString] = SETTINGS.VASP_DEFAULT_INPUT_SETS,
        check_potcar: bool = True,
        kpts_tolerance: float = SETTINGS.VASP_KPTS_TOLERANCE,
        allow_kpoint_shifts: bool = SETTINGS.VASP_ALLOW_KPT_SHIFT,
        allow_explicit_kpoint_mesh: str | bool = SETTINGS.VASP_ALLOW_EXPLICIT_KPT_MESH,
        fft_grid_tolerance: float = SETTINGS.VASP_FFT_GRID_TOLERANCE,
        num_ionic_steps_to_avg_drift_over: int = SETTINGS.VASP_NUM_IONIC_STEPS_FOR_DRIFT,
        max_allowed_scf_gradient: float = SETTINGS.VASP_MAX_SCF_GRADIENT,
        fast: bool = SETTINGS.FAST_VALIDATION,
    ) -> ValidationDoc:
        """
        Determines if a calculation is valid based on expected input parameters from a pymatgen inputset

        Args:
            task_doc: the task document to process
        Kwargs:
            input_sets: a dictionary of task_types -> pymatgen input set for validation
            potcar_summary_stats: Dictionary of potcar summary data. Mapping is calculation type -> potcar symbol -> summary data.
            kpts_tolerance: the tolerance to allow kpts to lag behind the input set settings
            allow_kpoint_shifts: Whether to consider a task valid if kpoints are shifted by the user
            allow_explicit_kpoint_mesh: Whether to consider a task valid if the user defines an explicit kpoint mesh
            fft_grid_tolerance: Relative tolerance for FFT grid parameters to still be a valid
            num_ionic_steps_to_avg_drift_over: Number of ionic steps to average over when validating drift forces
            max_allowed_scf_gradient: maximum uphill gradient allowed for SCF steps after the
                initial equillibriation period. Note this is in eV per atom.
            fast : whether to stop validation when any check fails
        """

        bandgap = task_doc["output"]["bandgap"]
        calcs_reversed = task_doc["calcs_reversed"]

        # used for most input tag checks (as this is more reliable than examining the INCAR file directly in most cases)
        parameters = task_doc["input"]["parameters"]

        # used for INCAR tag checks where you need to look at the actual INCAR (semi-rare)
        incar = calcs_reversed[0]["input"]["incar"]

        orig_inputs = {} if (task_doc["orig_inputs"] is None) else task_doc["orig_inputs"]

        cls_kwargs: dict[str, Any] = {
            "task_id": task_doc["task_id"] if task_doc["task_id"] else None,
            "calc_type": _get_calc_type(calcs_reversed, orig_inputs),
            "task_type": _get_task_type(calcs_reversed, orig_inputs),
            "run_type": _get_run_type(calcs_reversed),
            "reasons": [],
            "warnings": [],
        }

        vasp_version = [int(x) for x in calcs_reversed[0]["vasp_version"].split(".")[:3]]
        CheckVaspVersion(
            reasons=cls_kwargs["reasons"],
            warnings=cls_kwargs["warnings"],
            vasp_version=vasp_version,
            parameters=parameters,
            incar=incar,
            defaults=_vasp_defaults,
            fast=fast,
        ).check()

        if len(cls_kwargs["reasons"]) > 0 and fast:
            return cls(**cls_kwargs)

        if allow_explicit_kpoint_mesh == "auto":
            allow_explicit_kpoint_mesh = True if "NSCF" in cls_kwargs["calc_type"].name else False

        if calcs_reversed[0].get("input", {}).get("structure", None):
            structure = calcs_reversed[0]["input"]["structure"]
        else:
            structure = task_doc["input"]["structure"] or task_doc["output"]["structure"]
        structure = Structure.from_dict(structure)

        try:
            valid_input_set = _get_input_set(
                cls_kwargs["run_type"],
                cls_kwargs["task_type"],
                cls_kwargs["calc_type"],
                structure,
                input_sets,
                bandgap,
            )
        except Exception as e:
            cls_kwargs["reasons"].append(
                "NO MATCHING MP INPUT SET --> no matching MP input set was found. If you believe this to be a mistake, please create a GitHub issue."
            )
            valid_input_set = None
            print(f"Error while finding MP input set: {e}.")

        if valid_input_set:
            # Tests ordered by expected computational burden - help optimize `fast` check
            # Intuitively, more important checks (INCAR, KPOINTS, and POTCAR settings) would come first
            # But to optimize speed in fast mode (relevant for validating a large batch of calculations)
            # the faster checks have to come first:
            #   1. VASP version
            #   2. Common errors (known bugs in VASP, erratic SCF convergence, etc.)
            #   3. KPOINTS or KSPACING (from INCAR)
            #   4. INCAR (many sequential checks of possible INCAR tags + updating defaults)

            # TODO: check for surface/slab calculations!!!!!!

            CheckCommonErrors(
                reasons=cls_kwargs["reasons"],
                warnings=cls_kwargs["warnings"],
                task_doc=task_doc,
                parameters=parameters,
                structure=structure,
                run_type=cls_kwargs["run_type"],
                fast=fast,
                defaults=_vasp_defaults,
                valid_max_allowed_scf_gradient=max_allowed_scf_gradient,
                num_ionic_steps_to_avg_drift_over=num_ionic_steps_to_avg_drift_over,
            ).check()

            CheckKpointsKspacing(
                reasons=cls_kwargs["reasons"],
                warnings=cls_kwargs["warnings"],
                valid_input_set=valid_input_set,
                kpoints=calcs_reversed[0]["input"]["kpoints"],
                structure=structure,
                defaults=_vasp_defaults,
                kpts_tolerance=kpts_tolerance,
                allow_explicit_kpoint_mesh=allow_explicit_kpoint_mesh,
                allow_kpoint_shifts=allow_kpoint_shifts,
                fast=fast,
            ).check()

            if check_potcar:
                CheckPotcar(
                    reasons=cls_kwargs["reasons"],
                    warnings=cls_kwargs["warnings"],
                    valid_input_set=valid_input_set,
                    structure=structure,
                    potcars=calcs_reversed[0]["input"]["potcar_spec"],
                    fast=fast,
                ).check()

            CheckIncar(
                reasons=cls_kwargs["reasons"],
                warnings=cls_kwargs["warnings"],
                valid_input_set=valid_input_set,
                task_doc=task_doc,
                parameters=parameters,
                structure=structure,
                vasp_version=vasp_version,
                task_type=cls_kwargs["task_type"],
                defaults=_vasp_defaults,
                fft_grid_tolerance=fft_grid_tolerance,
                fast=fast,
            ).check()

        return cls(**cls_kwargs)

    @classmethod
    def from_directory(cls, dir_name: Path | str, **kwargs) -> ValidationDoc:
        """
        Determines if a calculation is valid based on expected input parameters from a pymatgen inputset

        Args:
            dir_name: the directory containing the calculation files to process
        Possible kwargs for `from_dict` method:
            input_sets: a dictionary of task_types -> pymatgen input set for validation
            check_potcar: Whether to check POTCARs against known libraries.
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

            return cls.from_task_doc(task_doc=task_doc, **kwargs)

        except Exception as e:
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
