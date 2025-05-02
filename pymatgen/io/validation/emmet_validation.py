"""Validate VASP calculations using emmet."""

from __future__ import annotations

from datetime import datetime
from pydantic import Field

from emmet.core.tasks import TaskDoc
from emmet.core.base import EmmetBaseModel
from emmet.core.mpid import MPID
from emmet.core.utils import utcnow

from pymatgen.io.vasp import Incar

from pymatgen.io.validation.common import LightOutcar, LightVasprun, PotcarSummaryStats, VaspFiles, VaspInputSafe
from pymatgen.io.validation.validation import VaspValidator

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing_extensions import Self


class ValidationDoc(EmmetBaseModel):
    """
    Validation document for a VASP calculation
    """

    task_id: MPID | None = Field(None, description="The task_id for this validation document")

    valid: bool = Field(False, description="Whether this task is valid.")

    last_updated: datetime = Field(
        description="The most recent time when this document was updated.",
        default_factory=utcnow,
    )

    reasons: list[str] = Field([], description="List of deprecation tags detailing why this task is not valid")

    warnings: list[str] = Field([], description="List of potential warnings about this calculation")

    @classmethod
    def from_task_doc(cls, task_doc: TaskDoc, **kwargs) -> Self:

        final_calc = task_doc.calcs_reversed[0]

        potcar_stats = None
        if final_calc.input.potcar_spec:

            potcar_stats = [
                PotcarSummaryStats(
                    titel=ps.titel,
                    keywords=ps.summary_stats["keywords"] if ps.summary_stats else None,
                    stats=ps.summary_stats["stats"] if ps.summary_stats else None,
                    lexch="pe" if final_calc.input.potcar_type[0] == "PAW_PBE" else "ca",
                )
                for ps in final_calc.input.potcar_spec
            ]

        vasp_files = VaspFiles(
            user_input=VaspInputSafe(
                incar=Incar(final_calc.input.incar),
                structure=final_calc.input.structure,
                potcar=potcar_stats,
            )
        )
        vasp_files._outcar = LightOutcar(**{k: final_calc.output.outcar.get(k) for k in ("drift", "magnetization")})
        vasp_files._vasprun = LightVasprun(
            vasp_version=final_calc.vasp_version,
            ionic_steps=[ionic_step.model_dump() for ionic_step in final_calc.output.ionic_steps],
            final_energy=task_doc.output.energy,
            final_structure=task_doc.output.structure,
            kpoints=final_calc.input.kpoints,
            parameters=final_calc.input.parameters,
            bandgap=final_calc.output.bandgap,
        )

        validator = VaspValidator.from_vasp_input(vasp_files=vasp_files)
        return cls(
            valid=validator.is_valid,
            reasons=validator.reasons,
            warnings=validator.warnings,
            **kwargs,
        )
