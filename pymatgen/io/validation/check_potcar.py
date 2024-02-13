"""Check POTCAR against known POTCARs in pymatgen, without setting up psp_resources."""
from __future__ import annotations
from dataclasses import dataclass, field
from importlib.resources import files as import_resource_files
from monty.serialization import loadfn
import numpy as np

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pymatgen.core import Structure
    from pymatgen.io.vasp import Potcar
    from pymatgen.io.vasp.sets import VaspInputSet

_potcar_summary_stats = loadfn(import_resource_files("pymatgen.io.vasp") / "potcar-summary-stats.json.bz2")


@dataclass
class CheckPotcar:
    """
    Check POTCAR against library of known valid POTCARs.

    potcar_summary_stats : dict
        Dictionary of potcar summary data. Mapping is calculation type -> potcar symbol -> summary data.
    data_match_tol : float
        Tolerance for matching POTCARs to summary statistics data.
    """

    potcar_summary_stats: dict = field(default_factory=lambda: _potcar_summary_stats)
    data_match_tol: float = 1.0e-6

    def check(self, reasons: list[str], valid_input_set: VaspInputSet, structure: Structure, potcars: Potcar):
        """
        Checks to make sure the POTCAR is equivalent to the correct POTCAR from the pymatgen input set.

        Parameters
        -----------
        reasons : list[str]
            A list of error strings to update if a check fails. These are higher
            severity and would deprecate a calculation.
        valid_input_set: VaspInputSet
            Valid input set to compare user INCAR parameters to.
        structure: Pymatgen Structure
            Structure used in the calculation.
        potcar: Pymatgen Potcar
            POTCAR used in the calculation.
        """

        if not self.potcar_summary_stats:
            return

        if potcars is None:
            reasons.append(
                "PSEUDOPOTENTIALS --> Missing POTCAR files. "
                "Alternatively, our potcar checker may have an issue--please create a GitHub issue if you "
                "know your POTCAR exists and can be read by Pymatgen."
            )
            return

        psp_subset = self.potcar_summary_stats.get(valid_input_set._config_dict["POTCAR_FUNCTIONAL"], {})

        valid_potcar_summary_stats = {}  # type: ignore
        for element in structure.composition.remove_charges().as_dict():
            potcar_symbol = valid_input_set._config_dict["POTCAR"][element]
            for titel_no_spc in psp_subset:
                for psp in psp_subset[titel_no_spc]:
                    if psp["symbol"] == potcar_symbol:
                        if titel_no_spc not in valid_potcar_summary_stats:
                            valid_potcar_summary_stats[titel_no_spc] = []
                        valid_potcar_summary_stats[titel_no_spc].append(psp)

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
                    data_match = all(np.array(data_diff) < self.data_match_tol)
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
                "Issue validating POTCARS --> Likely due to an old version of Emmet "
                "(wherein potcar summary_stats is not saved in TaskDoc), though "
                "other errors have been seen. Hence, it is marked as invalid."
            )
