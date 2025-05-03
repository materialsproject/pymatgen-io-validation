from pathlib import Path
import pytest

from monty.serialization import loadfn
from pymatgen.io.validation.common import VaspFiles

_test_dir = Path(__file__).parent.joinpath("test_files").resolve()


@pytest.fixture(scope="session")
def test_dir():
    return _test_dir


vasp_calc_data: dict[str, VaspFiles] = {
    k: VaspFiles(**loadfn(_test_dir / "vasp" / f"{k}.json.gz"))
    for k in ("Si_uniform", "Si_static", "Si_old_double_relax")
}


def incar_check_list():
    """Pre-defined list of pass/fail tests."""
    return loadfn(_test_dir / "vasp" / "scf_incar_check_list.yaml")
