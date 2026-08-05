"""Test validation without using a library of fake POTCARs."""

from tempfile import TemporaryDirectory

import pytest
from monty.serialization import loadfn
from pymatgen.core import SETTINGS as PMG_SETTINGS
from pymatgen.io.vasp.inputs import PotcarSingle, _load_potcar_summary_stats

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
