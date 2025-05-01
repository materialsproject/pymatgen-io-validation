"""Common class constructor for validation checks."""

from __future__ import annotations

from functools import cached_property
from importlib import import_module
import os
from pathlib import Path
from pydantic import BaseModel, ConfigDict, Field, computed_field, model_serializer, PrivateAttr
from typing import TYPE_CHECKING, Any

from pymatgen.core import Structure
from pymatgen.io.vasp import Incar, Kpoints, Poscar, Potcar, Outcar, Vasprun
from pymatgen.io.vasp.sets import VaspInputSet

from pymatgen.io.validation.vasp_defaults import VaspParam, VASP_DEFAULTS_DICT

if TYPE_CHECKING:
    from typing_extensions import Self


class ValidationError(Exception):
    """Define custom exception during validation."""


class PotcarSummaryStats(BaseModel):
    """Schematize `PotcarSingle._summary_stats`."""

    class _PotcarSummaryStatsKeywords(BaseModel):
        """Schematize `PotcarSingle._summary_stats["keywords"]` field."""

        header: set[str] = Field(description="The keywords in the POTCAR header.")
        data: set[str] = Field(description="The keywords in the POTCAR body.")

    class _PotcarSummaryStatsStats(BaseModel):
        """Schematize `PotcarSingle._summary_stats["stats"]` field."""

        class _PotcarSummaryStatsNames(BaseModel):
            """Define statistics used in `PotcarSingle._summary_stats`."""

            MEAN: float = Field(description="Data mean.")
            ABSMEAN: float = Field(description="Data magnitude mean.")
            VAR: float = Field(description="Mean of squares of data.")
            MIN: float = Field(description="Data minimum.")
            MAX: float = Field(description="Data maximum.")

        header: _PotcarSummaryStatsNames = Field(description="The keywords in the POTCAR header.")
        data: _PotcarSummaryStatsNames = Field(description="The keywords in the POTCAR body.")

    keywords: _PotcarSummaryStatsKeywords
    stats: _PotcarSummaryStatsStats
    titel: str
    lexch: str

    @classmethod
    def from_file(cls, potcar: os.PathLike | Potcar) -> Self:
        """Create a list of PotcarSummaryStats from a POTCAR."""
        if not isinstance(potcar, Potcar):
            potcar = Potcar.from_file(potcar)
        return [cls(**p._summary_stats, titel=p.TITEL, lexch=p.LEXCH) for p in potcar]


class VaspInputSafe(BaseModel):
    """Stricter VaspInputSet with no POTCAR info."""

    incar: Incar = Field(description="The INCAR used in the calculation.")
    poscar: Poscar = Field(description="The structure associated with the calculation.")
    kpoints: Kpoints | None = Field(None, description="The optional KPOINTS or IBZKPT file used in the calculation.")
    potcar: list[PotcarSummaryStats] | None = Field(None, description="The optional POTCAR used in the calculation.")

    @model_serializer
    def deserialize_objects(self) -> dict[str, dict[str, Any]]:
        """Ensure all pymatgen objects are deserialized."""
        model_dumped: dict[str, dict[str, Any]] = {"potcar": [p.model_dump() for p in self.user_input.potcar]}
        for k in (
            "incar",
            "poscar",
            "kpoints",
        ):
            if pmg_obj := getattr(self, k):
                model_dumped[k] = pmg_obj.as_dict()
        return model_dumped

    @classmethod
    def from_vasp_input_set(cls, vis: VaspInputSet) -> Self:
        return cls(
            **{
                k: getattr(vis, k)
                for k in (
                    "incar",
                    "kpoints",
                    "poscar",
                )
            },
            potcar=PotcarSummaryStats.from_file(vis.potcar),
        )


class VaspFiles(BaseModel):
    """Define required and optional files for validation."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    user_input: VaspInputSafe = Field(description="The VASP input set used in the calculation.")
    _outcar_path: os.PathLike | None = PrivateAttr(None)
    _vasprun_path: os.PathLike | None = PrivateAttr(None)

    @cached_property
    def outcar(self) -> Outcar | None:
        """The optional OUTCAR."""
        if self._outcar_path:
            return Outcar(self._outcar_path)
        return None

    @cached_property
    def vasprun(self) -> Vasprun | None:
        """The optional vasprun.xml."""
        if self._vasprun_path:
            return Vasprun(self._vasprun_path)
        return None

    @property
    def actual_kpoints(self) -> Kpoints | None:
        if self.user_input.kpoints:
            return self.user_input.kpoints
        elif self.vasprun:
            return self.vasprun.kpoints
        return None

    @property
    def structure(self) -> Structure:
        """Return the Structure object from the POSCAR."""
        return self.user_input.poscar.structure

    @property
    def vasp_version(self) -> tuple[int, int, int] | None:
        """Return the VASP version as a tuple of int, if available."""
        if self.vasprun:
            vvn = [int(x) for x in self.vasprun.vasp_version.split(".")]
            return (vvn[0], vvn[1], vvn[2])
        return None

    @classmethod
    def from_paths(
        cls,
        incar: os.PathLike[str],
        poscar: os.PathLike[str],
        kpoints: os.PathLike[str] | None = None,
        potcar: os.PathLike[str] | None = None,
        outcar: os.PathLike[str] | None = None,
        vasprun: os.PathLike[str] | None = None,
    ):
        """Construct a set of VASP I/O from file paths."""
        config: dict[str, dict[str] | os.PathLike] = {"user_input": {}}
        _vars = locals()

        to_obj = {
            "incar": Incar,
            "kpoints": Kpoints,
            "poscar": Poscar,
            "potcar": PotcarSummaryStats,
        }
        for file_name, file_cls in to_obj.items():
            if (path := _vars.get(file_name)) and Path(path).exists():
                config["user_input"][file_name] = file_cls.from_file(path)

        vf = cls(**config)
        for file_name in ("outcar", "vasprun"):
            if (path := _vars.get(file_name)) and Path(path).exists():
                setattr(vf, f"_{file_name}_path", path)

        return vf

    @computed_field  # type: ignore[misc]
    @cached_property
    def run_type(self) -> str:
        """Get the run type of a calculation."""

        ibrion = self.user_input.incar.get("IBRION", VASP_DEFAULTS_DICT["IBRION"].value)
        if self.user_input.incar.get("NSW", VASP_DEFAULTS_DICT["NSW"].value) > 0 and ibrion == -1:
            ibrion = 0

        run_type = {
            -1: "static",
            0: "md",
            **{k: "relax" for k in range(1, 4)},
            **{k: "phonon" for k in range(5, 9)},
            **{k: "ts" for k in (40, 44)},
        }.get(ibrion)

        if self.user_input.incar.get("ICHARG", VASP_DEFAULTS_DICT["ICHARG"].value) >= 10:
            run_type = "nonscf"
        if self.user_input.incar.get("LCHIMAG", VASP_DEFAULTS_DICT["LCHIMAG"].value):
            run_type == "nmr"

        if run_type is None:
            raise ValidationError(
                "Could not determine a valid run type. We currently only validate "
                "Geometry optimizations (relaxations), single-points (statics), "
                "and non-self-consistent fixed charged density calculations. ",
            )

        return run_type

    @computed_field  # type: ignore[misc]
    @cached_property
    def functional(self) -> str:
        """Determine the functional used in the calculation.

        Note that this is not a complete determination.
        Only the functionals used by MP are detected here.
        """

        func = None
        func_from_potcar = None
        if self.user_input.potcar:
            func_from_potcar = {"pe": "pbe", "ca": "lda"}.get(self.user_input.potcar[0].lexch.lower())

        if gga := self.user_input.incar.get("GGA"):
            if gga.lower() == "pe":
                func = "pbe"
            elif gga.lower() == "ps":
                func = "pbesol"
            else:
                func = gga.lower()

        if metagga := self.user_input.incar.get("METAGGA"):
            if gga:
                raise ValidationError(
                    "Both the GGA and METAGGA tags were set, which can lead to large errors. "
                    "For context, see:\n"
                    "https://github.com/materialsproject/atomate2/issues/453#issuecomment-1699605867"
                )
            if metagga.lower() == "scan":
                func = "scan"
            elif metagga.lower().startswith("r2sca"):
                func = "r2scan"
            else:
                func = metagga.lower()

        if self.user_input.incar.get("LHFCALC", False):
            if (func == "pbe" or func_from_potcar == "pbe") and (self.user_input.incar.get("HFSCREEN", 0.0) > 0.0):
                func = "hse06"
            else:
                func = None

        func = func or func_from_potcar
        if func is None:
            raise ValidationError(
                "Currently, we only validate calculations using the following functionals:\n"
                "GGA : PBE, PBEsol\n"
                "meta-GGA : SCAN, r2SCAN\n"
                "Hybrids: HSE06"
            )
        return func

    @property
    def bandgap(self) -> float | None:
        """Determine the bandgap from vasprun.xml."""
        if self.vasprun:
            return self.vasprun.get_band_structure(efermi="smart").get_band_gap()["energy"]
        return None

    @computed_field  # type: ignore[misc]
    @cached_property
    def valid_input_set(self) -> VaspInputSafe:
        """
        Determine the MP-compliant input set for a calculation.

        We need only determine a rough input set here.
        The precise details of the input set do not matter.
        """

        incar_updates: dict[str, Any] = {}
        set_name: str | None = None
        if self.functional == "pbe":
            if self.run_type == "nonscf":
                set_name = "MPNonSCFSet"
            elif self.run_type == "nmr":
                set_name = "MPNMRSet"
            else:
                set_name = f"MP{self.run_type.capitalize()}Set"
        elif self.functional in ("pbesol", "scan", "r2scan", "hse06"):
            if self.functional == "pbesol":
                incar_updates["GGA"] = "PS"
            elif self.functional == "scan":
                incar_updates["METAGGA"] = "SCAN"
            elif self.functional == "hse06":
                incar_updates.update(
                    LHFCALC=True,
                    HFSCREEN=0.2,
                    GGA="PE",
                )
            set_name = f"MPScan{self.run_type.capitalize()}Set"

        if set_name is None:
            raise ValidationError(
                "Could not determine a valid input set from the specified "
                f"functional = {self.functional} and calculation type {self.run_type}."
            )

        # Note that only the *previous* bandgap informs the k-point density
        vis = getattr(import_module("pymatgen.io.vasp.sets"), set_name)(
            structure=self.user_input.poscar.structure,
            bandgap=None,
            user_incar_settings=incar_updates,
        )

        return VaspInputSafe.from_vasp_input_set(vis)


class BaseValidator(BaseModel):
    """
    Template for validation classes.

    This class will check any function with the name prefix `_check_`.
    `_check_*` functions should take VaspFiles, and two lists of strings
    (`reasons` and `warnings`) as args:

    def _check_example(self, vasp_files : VaspFiles, reasons : list[str], warnings : list[str]) -> None:
        if self.name == "whole mango":
            reasons.append("We only accept sliced or diced mango at this time.")
        elif self.name == "diced mango":
            warnings.append("We prefer sliced mango, but will accept diced mango.")
    """

    name: str = Field("Base validator class", description="Name of the validator class.")
    vasp_defaults: dict[str, VaspParam] = Field(VASP_DEFAULTS_DICT, description="Default VASP settings.")
    fast: bool = Field(False, description="Whether to perform a quick check (True) or to perform all checks (False).")

    def auto_fail(self, vasp_files: VaspFiles, reasons: list[str], warnings: list[str]) -> bool:
        """Quick stop in case none of the checks can be performed."""
        return False

    def check(self, vasp_files: VaspFiles, reasons: list[str], warnings: list[str]) -> None:
        """
        Execute all methods on the class with a name prefix `_check_`.

        Parameters
        -----------
        reasons : VaspFiles
            A set of required and optional VASP input and output objects.
        reasons : list of str
            A list of errors to update if a check fails. These are higher
            severity and would deprecate a calculation.
        warnings : list of str
            A list of warnings to update if a check fails. These are lower
            severity and would flag a calculation for possible review.
        """

        if self.auto_fail(vasp_files, reasons, warnings):
            return

        checklist = {attr for attr in dir(self) if attr.startswith("_check_")}
        for attr in checklist:
            if self.fast and len(reasons) > 0:
                # fast check: stop checking whenever a single check fails
                break

            getattr(self, attr)(vasp_files, reasons, warnings)
