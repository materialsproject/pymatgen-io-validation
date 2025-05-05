"""Common class constructor for validation checks."""

from __future__ import annotations

from functools import cached_property
from importlib import import_module
import os
from pathlib import Path
from pydantic import BaseModel, Field, model_serializer, PrivateAttr
from typing import TYPE_CHECKING, Any, Optional

from pymatgen.core import Structure
from pymatgen.io.vasp import Incar, Kpoints, Poscar, Potcar, Outcar, Vasprun
from pymatgen.io.vasp.sets import VaspInputSet

from pymatgen.io.validation.vasp_defaults import VaspParam, VASP_DEFAULTS_DICT
from pymatgen.io.validation.settings import IOValidationSettings

if TYPE_CHECKING:
    from typing_extensions import Self

SETTINGS = IOValidationSettings()


class ValidationError(Exception):
    """Define custom exception during validation."""


class PotcarSummaryStats(BaseModel):
    """Schematize `PotcarSingle._summary_stats`."""

    class _PotcarSummaryStatsKeywords(BaseModel):
        """Schematize `PotcarSingle._summary_stats["keywords"]` field."""

        header: set[str] = Field(description="The keywords in the POTCAR header.")
        data: set[str] = Field(description="The keywords in the POTCAR body.")

        @model_serializer
        def set_to_list(self) -> dict[str, list[str]]:
            """Ensure JSON compliance of set fields."""
            return {k: list(getattr(self, k)) for k in ("header", "data")}

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

    keywords: Optional[_PotcarSummaryStatsKeywords] = None
    stats: Optional[_PotcarSummaryStatsStats] = None
    titel: str
    lexch: str

    @classmethod
    def from_file(cls, potcar_path: os.PathLike | Potcar) -> list[Self]:
        """Create a list of PotcarSummaryStats from a POTCAR."""
        if isinstance(potcar_path, Potcar):
            potcar: Potcar = potcar_path
        else:
            potcar = Potcar.from_file(str(potcar_path))
        return [cls(**p._summary_stats, titel=p.TITEL, lexch=p.LEXCH) for p in potcar]


class LightOutcar(BaseModel):
    """Schematic of pymatgen's Outcar."""

    drift: Optional[list[list[float]]] = Field(None, description="The drift forces.")
    magnetization: Optional[list[dict[str, float]]] = Field(
        None, description="The on-site magnetic moments, possibly with orbital resolution."
    )


class LightVasprun(BaseModel):
    """Lightweight version of pymatgen Vasprun."""

    vasp_version: str = Field(description="The dot-separated version of VASP used.")
    ionic_steps: list[dict[str, Any]] = Field(description="The ionic steps in the calculation.")
    final_energy: float = Field(description="The final total energy in eV.")
    final_structure: Structure = Field(description="The final structure.")
    kpoints: Kpoints = Field(description="The actual k-points used in the calculation.")
    parameters: dict[str, Any] = Field(description="The default-padded input parameters interpreted by VASP.")
    bandgap: float = Field(description="The bandgap - note that this field is derived from the Vasprun object.")
    potcar_symbols: Optional[list[str]] = Field(
        None,
        description="Optional: if a POTCAR is unavailable, this is used to determine the functional used in the calculation.",
    )

    @classmethod
    def from_vasprun(cls, vasprun: Vasprun) -> Self:
        """
        Create a LightVasprun from a pymatgen Vasprun.

        Parameters
        -----------
        vasprun : pymatgen Vasprun

        Returns
        -----------
        LightVasprun
        """
        return cls(
            **{k: getattr(vasprun, k) for k in cls.model_fields if k != "bandgap"},
            bandgap=vasprun.get_band_structure(efermi="smart").get_band_gap()["energy"],
        )


class VaspInputSafe(BaseModel):
    """Stricter VaspInputSet with no POTCAR info."""

    incar: Incar = Field(description="The INCAR used in the calculation.")
    structure: Structure = Field(description="The structure associated with the calculation.")
    kpoints: Optional[Kpoints] = Field(None, description="The optional KPOINTS or IBZKPT file used in the calculation.")
    potcar: Optional[list[PotcarSummaryStats]] = Field(None, description="The optional POTCAR used in the calculation.")
    potcar_functional: Optional[str] = Field(None, description="The pymatgen-labelled POTCAR library release.")
    _pmg_vis: Optional[VaspInputSet] = PrivateAttr(None)

    @model_serializer
    def deserialize_objects(self) -> dict[str, Any]:
        """Ensure all pymatgen objects are deserialized."""
        model_dumped: dict[str, Any] = {}
        if self.potcar:
            model_dumped["potcar"] = [p.model_dump() for p in self.potcar]
        for k in (
            "incar",
            "structure",
            "kpoints",
        ):
            if pmg_obj := getattr(self, k):
                model_dumped[k] = pmg_obj.as_dict()
        return model_dumped

    @classmethod
    def from_vasp_input_set(cls, vis: VaspInputSet) -> Self:
        """
        Create a VaspInputSafe from a pymatgen VaspInputSet.

        Parameters
        -----------
        vasprun : pymatgen VaspInputSet

        Returns
        -----------
        VaspInputSafe
        """
        new_vis = cls(
            **{
                k: getattr(vis, k)
                for k in (
                    "incar",
                    "kpoints",
                    "structure",
                )
            },
            potcar=PotcarSummaryStats.from_file(vis.potcar),
            potcar_functional=vis.potcar_functional,
        )
        new_vis._pmg_vis = vis
        return new_vis

    def _calculate_ng(self, **kwargs) -> tuple[list[int], list[int]] | None:
        """Interface to pymatgen vasp input set as needed."""
        if self._pmg_vis:
            return self._pmg_vis.calculate_ng(**kwargs)
        return None


class VaspFiles(BaseModel):
    """Define required and optional files for validation."""

    user_input: VaspInputSafe = Field(description="The VASP input set used in the calculation.")
    outcar: Optional[LightOutcar] = None
    vasprun: Optional[LightVasprun] = None

    @property
    def actual_kpoints(self) -> Kpoints | None:
        """The actual KPOINTS / IBZKPT used in the calculation, if applicable."""
        if self.user_input.kpoints:
            return self.user_input.kpoints
        elif self.vasprun:
            return self.vasprun.kpoints
        return None

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
        config: dict[str, Any] = {"user_input": {}}
        _vars = locals()

        to_obj = {
            "incar": Incar,
            "kpoints": Kpoints,
            "poscar": Poscar,
            "potcar": PotcarSummaryStats,
            "outcar": Outcar,
            "vasprun": Vasprun,
        }
        potcar_enmax = None
        for file_name, file_cls in to_obj.items():
            if (path := _vars.get(file_name)) and Path(path).exists():
                if file_name == "poscar":
                    config["user_input"]["structure"] = Poscar.from_file(path).structure
                elif hasattr(file_cls, "from_file"):
                    config["user_input"][file_name] = file_cls.from_file(path)
                else:
                    config[file_name] = file_cls(path)

                if file_name == "potcar":
                    potcar_enmax = max(ps.ENMAX for ps in Potcar.from_file(path))

        if config.get("outcar"):
            config["outcar"] = LightOutcar(
                drift=config["outcar"].drift,
                magnetization=config["outcar"].magnetization,
            )

        if config.get("vasprun"):
            config["vasprun"] = LightVasprun.from_vasprun(config["vasprun"])
        elif not config["user_input"]["incar"].get("ENCUT") and potcar_enmax:
            config["user_input"]["incar"]["ENCUT"] = potcar_enmax

        return cls(**config)

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
        elif self.vasprun and self.vasprun.potcar_symbols:
            pot_func = self.vasprun.potcar_symbols[0].split()[0].split("_")[-1]
            func_from_potcar = "pbe" if pot_func == "PBE" else "lda"

        if gga := self.user_input.incar.get("GGA"):
            if gga.lower() == "pe":
                func = "pbe"
            elif gga.lower() == "ps":
                func = "pbesol"
            else:
                func = gga.lower()

        if (metagga := self.user_input.incar.get("METAGGA")) and metagga.lower() != "none":
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
            return self.vasprun.bandgap
        return None

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
            elif self.run_type == "md":
                set_name = None
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
            structure=self.user_input.structure,
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
