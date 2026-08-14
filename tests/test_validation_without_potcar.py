"""Test validation without using a library of fake POTCARs."""

from tempfile import TemporaryDirectory

import pytest
from monty.serialization import loadfn
from pymatgen.core import SETTINGS as PMG_SETTINGS
from pymatgen.core import Lattice, Structure
from pymatgen.io.vasp.inputs import PotcarSingle, _load_potcar_summary_stats

from pymatgen.io.validation.check_potcar import CheckPotcar
from pymatgen.io.validation.common import PotcarSummaryStats, VaspFiles
from pymatgen.io.validation.validation import VaspValidator


def test_validation_without_potcars(test_dir):
    with TemporaryDirectory() as tmp_dir:

        pytest.MonkeyPatch().setitem(PMG_SETTINGS, "PMG_VASP_PSP_DIR", tmp_dir)

        # ensure that potcar library is unset to empty temporary directory
        with pytest.raises(FileNotFoundError):
            PotcarSingle.from_symbol_and_functional(symbol="Si", functional="PBE")

        # Add summary stats to input files
        ref_titel = "PAW_PBE Si 05Jan2001"

        ref_pspec = _load_potcar_summary_stats()["PBE"][ref_titel.replace(" ", "")][0]
        vf = loadfn(test_dir / "vasp" / "Si_uniform.json.gz")
        vf["user_input"]["potcar"] = [
            PotcarSummaryStats(titel=ref_titel, lexch="PE", **ref_pspec)
        ]
        vf["user_input"]["potcar_functional"] = "PBE"
        vasp_files = VaspFiles(**vf)

        validated = VaspValidator(vasp_files=vasp_files)
        assert validated.valid


def test_validation_without_potcars_symbol_recurring_in_titel():
    """Regression test for POTCAR symbols whose letters recur in the TITEL.

    Without a local POTCAR library, reference stats are looked up from the
    pregenerated summary stats keyed by (space-stripped) TITEL. Selenium's TITEL,
    ``PAW_PBE Se 06Sep2000``, contains ``Se`` twice (in ``PBESe`` and ``Sep``).
    The previous fallback rebuilt the reference TITEL via
    ``titel_no_spc.split("Se")``, yielding the mangled ``PAW_PBE Se 06`` -- which
    never matches the real TITEL, so a valid POTCAR was flagged as incorrect.
    """
    with TemporaryDirectory() as tmp_dir:
        pytest.MonkeyPatch().setitem(PMG_SETTINGS, "PMG_VASP_PSP_DIR", tmp_dir)

        # Ensure no local POTCAR library is available, forcing the fallback path.
        with pytest.raises(FileNotFoundError):
            PotcarSingle.from_symbol_and_functional(symbol="Se", functional="PBE")

        ref_titel = "PAW_PBE Se 06Sep2000"
        ref_pspec = _load_potcar_summary_stats()["PBE"][ref_titel.replace(" ", "")][0]

        vasp_files = VaspFiles(
            user_input={
                "incar": {"GGA": "PE", "IBRION": 2, "NSW": 99},
                "structure": Structure(Lattice.cubic(3.0), ["Se"], [[0.0, 0.0, 0.0]]),
                "potcar": [PotcarSummaryStats(titel=ref_titel, lexch="PE", **ref_pspec)],
                "potcar_functional": "PBE",
            }
        )

        reasons: list[str] = []
        CheckPotcar().check(vasp_files, reasons, [])
        assert not reasons
