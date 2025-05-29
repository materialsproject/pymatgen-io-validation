"""Check POTCAR against known POTCARs in pymatgen, without setting up psp_resources."""

from __future__ import annotations
from copy import deepcopy
from functools import cached_property
from pathlib import Path
from pydantic import Field
from importlib.resources import files as import_resource_files
from monty.serialization import loadfn
from typing import TYPE_CHECKING

from pymatgen.io.vasp import PotcarSingle

from pymatgen.io.validation.common import BaseValidator, ValidationError

if TYPE_CHECKING:
    from typing import Any
    from pymatgen.io.validation.common import VaspFiles


class CheckPotcar(BaseValidator):
    """
    Check POTCAR against library of known valid POTCARs.
    """

    name: str = "Check POTCAR"
    potcar_summary_stats_path: str | Path | None = Field(
        str(import_resource_files("pymatgen.io.vasp") / "potcar-summary-stats.json.bz2"),
        description="Path to potcar summary data. Mapping is calculation type -> potcar symbol -> summary data.",
    )
    data_match_tol: float = Field(1.0e-6, description="Tolerance for matching POTCARs to summary statistics data.")
    ignore_header_keys: set[str] | None = Field(
        {"copyr", "sha256"}, description="POTCAR summary statistics keywords.header fields to ignore during validation"
    )

    @cached_property
    def potcar_summary_stats(self) -> dict:
        """Load POTCAR summary statistics file."""
        if self.potcar_summary_stats_path:
            return loadfn(self.potcar_summary_stats_path, cls=None)
        return {}

    def auto_fail(self, vasp_files: VaspFiles, reasons: list[str], warnings: list[str]) -> bool:
        """Skip if no POTCAR was provided, or if summary stats file was unset."""

        if self.potcar_summary_stats_path is None:
            # If no reference summary stats specified, or we're only doing a quick check,
            # and there are already failure reasons, return
            return True
        elif vasp_files.user_input.potcar is None or any(
            ps.keywords is None or ps.stats is None for ps in vasp_files.user_input.potcar
        ):
            reasons.append(
                "PSEUDOPOTENTIALS --> Missing POTCAR files. "
                "Alternatively, our potcar checker may have an issue--please create a GitHub issue if you "
                "know your POTCAR exists and can be read by Pymatgen."
            )
            return True
        return False

    def _check_potcar_spec(self, vasp_files: VaspFiles, reasons: list[str], warnings: list[str]):
        """
        Checks to make sure the POTCAR is equivalent to the correct POTCAR from the pymatgen input set."""

        if vasp_files.valid_input_set.potcar:
            # If the user has pymatgen set up, use the pregenerated POTCAR summary stats.
            valid_potcar_summary_stats: dict[str, list[dict[str, Any]]] = {
                p.titel.replace(" ", ""): [p.model_dump()] for p in vasp_files.valid_input_set.potcar
            }
        elif vasp_files.valid_input_set._pmg_vis:
            # Fallback, use the stats from pymatgen - only load and cache summary stats here.
            psp_subset = self.potcar_summary_stats.get(vasp_files.valid_input_set.potcar_functional, {})

            valid_potcar_summary_stats = {}
            for element in vasp_files.user_input.structure.composition.remove_charges().as_dict():
                potcar_symbol = vasp_files.valid_input_set._pmg_vis._config_dict["POTCAR"][element]
                for titel_no_spc in psp_subset:
                    for psp in psp_subset[titel_no_spc]:
                        if psp["symbol"] == potcar_symbol:
                            if titel_no_spc not in valid_potcar_summary_stats:
                                valid_potcar_summary_stats[titel_no_spc] = []
                            valid_potcar_summary_stats[titel_no_spc].append(psp)
        else:
            raise ValidationError("Could not determine reference POTCARs.")

        try:
            incorrect_potcars: list[str] = []
            for potcar in vasp_files.user_input.potcar:  # type: ignore[union-attr]
                reference_summary_stats = valid_potcar_summary_stats.get(potcar.titel.replace(" ", ""), [])
                potcar_symbol = potcar.titel.split(" ")[1]

                if len(reference_summary_stats) == 0:
                    incorrect_potcars.append(potcar_symbol)
                    continue

                for _ref_psp in reference_summary_stats:
                    user_summary_stats = potcar.model_dump()
                    ref_psp = deepcopy(_ref_psp)
                    for _set in (user_summary_stats, ref_psp):
                        _set["keywords"]["header"] = set(_set["keywords"]["header"]).difference(self.ignore_header_keys)  # type: ignore[arg-type]
                    if found_match := PotcarSingle.compare_potcar_stats(
                        ref_psp, user_summary_stats, tolerance=self.data_match_tol
                    ):
                        break

                if not found_match:
                    incorrect_potcars.append(potcar_symbol)
                    if self.fast:
                        # quick return, only matters that one POTCAR didn't match
                        break

            if len(incorrect_potcars) > 0:
                # format error string
                incorrect_potcars = [potcar.split("_")[0] for potcar in incorrect_potcars]
                if len(incorrect_potcars) == 1:
                    incorrect_potcar_str = incorrect_potcars[0]
                else:
                    incorrect_potcar_str = (
                        ", ".join(incorrect_potcars[:-1]) + f", and {incorrect_potcars[-1]}"
                    )  # type: ignore

                reasons.append(
                    f"PSEUDOPOTENTIALS --> Incorrect POTCAR files were used for {incorrect_potcar_str}. "
                    "Alternatively, our potcar checker may have an issue--please create a GitHub issue if you "
                    "believe the POTCARs used are correct."
                )

        except KeyError:
            reasons.append(
                "Issue validating POTCARS --> Likely due to an old version of Emmet "
                "(wherein potcar summary_stats is not saved in TaskDoc), though "
                "other errors have been seen. Hence, it is marked as invalid."
            )
