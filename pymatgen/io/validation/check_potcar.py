"""Check POTCAR against known POTCARs in pymatgen, without setting up psp_resources."""

from __future__ import annotations
from dataclasses import dataclass, field
from importlib.resources import files as import_resource_files
from monty.serialization import loadfn
import numpy as np

from pymatgen.io.validation.common import BaseValidator

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pymatgen.core import Structure
    from pymatgen.io.vasp.sets import VaspInputSet

_potcar_summary_stats = loadfn(import_resource_files("pymatgen.io.vasp") / "potcar-summary-stats.json.bz2")


@dataclass
class CheckPotcar(BaseValidator):
    """
    Check POTCAR against library of known valid POTCARs.

    reasons : list[str]
        A list of error strings to update if a check fails. These are higher
        severity and would deprecate a calculation.
    warnings : list[str]
        A list of warning strings to update if a check fails. These are lower
        severity and would flag a calculation for possible review.
    valid_input_set: VaspInputSet
        Valid input set to compare user INCAR parameters to.
    structure: Pymatgen Structure
        Structure used in the calculation.
    potcar: dict
        Spec (symbol, hash, and summary stats) for the POTCAR used in the calculation.
    name : str = "Check POTCARs"
        Name of the validator class
    fast : bool = False
        Whether to perform quick check.
        True: stop validation if any check fails.
        False: perform all checks.
    potcar_summary_stats : dict
        Dictionary of potcar summary data. Mapping is calculation type -> potcar symbol -> summary data.
    data_match_tol : float = 1.e-6
        Tolerance for matching POTCARs to summary statistics data.
    fast : bool = False
        True: stop validation when any single check fails
    """

    reasons: list[str]
    warnings: list[str]
    valid_input_set: VaspInputSet = None
    structure: Structure = None
    potcars: dict = None
    name: str = "Check POTCARs"
    potcar_summary_stats: dict = field(default_factory=lambda: _potcar_summary_stats)
    data_match_tol: float = 1.0e-6
    fast: bool = False

    def _check_potcar_spec(self):
        """
        Checks to make sure the POTCAR is equivalent to the correct POTCAR from the pymatgen input set."""

        if not self.potcar_summary_stats:
            # If no reference summary stats specified, or we're only doing a quick check,
            # and there are already failure reasons, return
            return

        if self.potcars is None or any(potcar.get("summary_stats") is None for potcar in self.potcars):
            self.reasons.append(
                "PSEUDOPOTENTIALS --> Missing POTCAR files. "
                "Alternatively, our potcar checker may have an issue--please create a GitHub issue if you "
                "know your POTCAR exists and can be read by Pymatgen."
            )
            return

        psp_subset = self.potcar_summary_stats.get(self.valid_input_set._config_dict["POTCAR_FUNCTIONAL"], {})

        valid_potcar_summary_stats = {}  # type: ignore
        for element in self.structure.composition.remove_charges().as_dict():
            potcar_symbol = self.valid_input_set._config_dict["POTCAR"][element]
            for titel_no_spc in psp_subset:
                for psp in psp_subset[titel_no_spc]:
                    if psp["symbol"] == potcar_symbol:
                        if titel_no_spc not in valid_potcar_summary_stats:
                            valid_potcar_summary_stats[titel_no_spc] = []
                        valid_potcar_summary_stats[titel_no_spc].append(psp)

        try:
            incorrect_potcars = []
            for potcar in self.potcars:
                reference_summary_stats = valid_potcar_summary_stats.get(potcar["titel"].replace(" ", ""), [])

                if len(reference_summary_stats) == 0:
                    incorrect_potcars.append(potcar["titel"].split(" ")[1])
                    continue

                for ref_psp in reference_summary_stats:
                    if found_match := self.compare_potcar_stats(ref_psp, potcar["summary_stats"]):
                        break

                if not found_match:
                    incorrect_potcars.append(potcar["titel"].split(" ")[1])
                    if self.fast:
                        # quick return, only matters that one POTCAR didn't match
                        break

            if len(incorrect_potcars) > 0:
                # format error string
                incorrect_potcars = [potcar.split("_")[0] for potcar in incorrect_potcars]
                if len(incorrect_potcars) == 2:
                    incorrect_potcars = ", ".join(incorrect_potcars[:-1]) + f" and {incorrect_potcars[-1]}"  # type: ignore
                elif len(incorrect_potcars) >= 3:
                    incorrect_potcars = ", ".join(incorrect_potcars[:-1]) + "," + f" and {incorrect_potcars[-1]}"  # type: ignore

                self.reasons.append(
                    f"PSEUDOPOTENTIALS --> Incorrect POTCAR files were used for {incorrect_potcars}. "
                    "Alternatively, our potcar checker may have an issue--please create a GitHub issue if you "
                    "believe the POTCARs used are correct."
                )

        except KeyError as e:
            print(f"POTCAR check exception: {e}")
            # Assume it is an old calculation without potcar_spec data and treat it as failing the POTCAR check
            self.reasons.append(
                "Issue validating POTCARS --> Likely due to an old version of Emmet "
                "(wherein potcar summary_stats is not saved in TaskDoc), though "
                "other errors have been seen. Hence, it is marked as invalid."
            )

    def compare_potcar_stats(self, potcar_stats_1: dict, potcar_stats_2: dict) -> bool:
        """Utility function to compare PotcarSingle._summary_stats."""

        if not all(
            potcar_stats_1.get(key)
            for key in (
                "keywords",
                "stats",
            )
        ) or (
            not all(
                potcar_stats_2.get(key)
                for key in (
                    "keywords",
                    "stats",
                )
            )
        ):
            return False

        key_match = all(
            set(potcar_stats_1["keywords"].get(key)) == set(potcar_stats_2["keywords"].get(key))  # type: ignore
            for key in ["header", "data"]
        )

        data_match = False
        if key_match:
            data_diff = [
                abs(potcar_stats_1["stats"].get(key, {}).get(stat) - potcar_stats_2["stats"].get(key, {}).get(stat))  # type: ignore
                for stat in ["MEAN", "ABSMEAN", "VAR", "MIN", "MAX"]
                for key in ["header", "data"]
            ]
            data_match = all(np.array(data_diff) < self.data_match_tol)

        return key_match and data_match
