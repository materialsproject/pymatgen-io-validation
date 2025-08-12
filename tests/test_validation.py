import pytest
import copy

from monty.serialization import loadfn
from pymatgen.core.structure import Structure
from pymatgen.io.vasp import Kpoints

from pymatgen.io.validation.validation import VaspValidator
from pymatgen.io.validation.common import ValidationError, VaspFiles, PotcarSummaryStats

from conftest import vasp_calc_data, incar_check_list, set_fake_potcar_dir


### TODO: add tests for many other MP input sets (e.g. MPNSCFSet, MPNMRSet, MPScanRelaxSet, Hybrid sets, etc.)
### TODO: add check for an MP input set that uses an IBRION other than [-1, 1, 2]
### TODO: add in check for MP set where LEFG = True
### TODO: add in check for MP set where LOPTICS = True
### TODO: fix logic for calc_type / run_type identification in Emmet!!! Or handle how we interpret them...

set_fake_potcar_dir()


def run_check(
    vasp_files: VaspFiles,
    error_message_to_search_for: str,
    should_the_check_pass: bool,
    vasprun_parameters_to_change: dict = {},  # for changing the parameters read from vasprun.xml
    incar_settings_to_change: dict = {},  # for directly changing the INCAR file,
    validation_doc_kwargs: dict = {},  # any kwargs to pass to the VaspValidator class
):
    _new_vf = vasp_files.model_dump()
    _new_vf["vasprun"]["parameters"].update(**vasprun_parameters_to_change)

    _new_vf["user_input"]["incar"].update(**incar_settings_to_change)

    validator = VaspValidator.from_vasp_input(vasp_files=VaspFiles(**_new_vf), **validation_doc_kwargs)
    has_specified_error = any([error_message_to_search_for in reason for reason in validator.reasons])

    assert (not has_specified_error) if should_the_check_pass else has_specified_error


def test_validation_from_files(test_dir):

    dir_name = test_dir / "vasp" / "Si_uniform"
    validator_from_paths = VaspValidator.from_directory(dir_name)
    validator_from_vasp_files = VaspValidator.from_vasp_input(vasp_files=vasp_calc_data["Si_uniform"])

    # Note: because the POTCAR info cannot be distributed, `validator_from_paths`
    # is missing POTCAR checks.
    assert set([r for r in validator_from_paths.reasons if "POTCAR" not in r]) == set(validator_from_vasp_files.reasons)
    assert set([r for r in validator_from_paths.warnings if "POTCAR" not in r]) == set(
        validator_from_vasp_files.warnings
    )
    assert all(
        getattr(validator_from_paths.vasp_files.user_input, k) == getattr(validator_from_paths.vasp_files.user_input, k)
        for k in ("incar", "structure", "kpoints")
    )

    # Ensure that user modifcation to inputs after submitting valid
    # input leads to subsequent validation failures.
    # Re-instantiate VaspValidator to ensure pointers don't get messed up
    validated = VaspValidator(**validator_from_paths.model_dump())
    og_md5 = validated.vasp_files.md5
    assert validated.valid
    assert validated._validated_md5 == og_md5

    validated.vasp_files.user_input.incar["ENCUT"] = 1.0
    new_md5 = validated.vasp_files.md5
    assert new_md5 != og_md5
    assert not validated.valid
    assert validated._validated_md5 == new_md5


@pytest.mark.parametrize(
    "object_name",
    [
        "Si_old_double_relax",
    ],
)
def test_potcar_validation(test_dir, object_name):
    vf_og = vasp_calc_data[object_name]

    correct_potcar_summary_stats = [
        PotcarSummaryStats(**ps) for ps in loadfn(test_dir / "vasp" / "fake_Si_potcar_spec.json.gz")
    ]

    # Check POTCAR (this test should PASS, as we ARE using a MP-compatible pseudopotential)
    vf = copy.deepcopy(vf_og)
    assert vf.user_input.potcar == correct_potcar_summary_stats
    run_check(vf, "PSEUDOPOTENTIALS", True)

    # Check POTCAR (this test should FAIL, as we are NOT using an MP-compatible pseudopotential)
    vf = copy.deepcopy(vf_og)
    incorrect_potcar_summary_stats = copy.deepcopy(correct_potcar_summary_stats)
    incorrect_potcar_summary_stats[0].stats.data.MEAN = 999999999
    vf.user_input.potcar = incorrect_potcar_summary_stats
    run_check(vf, "PSEUDOPOTENTIALS", False)


@pytest.mark.parametrize("object_name", ["Si_static", "Si_old_double_relax"])
def test_scf_incar_checks(test_dir, object_name):
    vf_og = vasp_calc_data[object_name]
    vf_og.vasprun.final_structure._charge = 0.0  # patch for old test files

    # Pay *very* close attention to whether a tag is modified in the incar or in the vasprun.xml's parameters!
    # Some parameters are validated from one or the other of these items, depending on whether VASP
    # changes the value between the INCAR and the vasprun.xml (which it often does)

    for incar_check in incar_check_list():
        run_check(
            vf_og,
            incar_check["err_msg"],
            incar_check["should_pass"],
            vasprun_parameters_to_change=incar_check.get("vasprun", {}),
            incar_settings_to_change=incar_check.get("incar", {}),
        )
    ### Most all of the tests below are too specific to use the kwargs in the
    # run_check() method. Hence, the calcs are manually modified. Apologies.

    # NELECT check
    vf = copy.deepcopy(vf_og)
    # must set NELECT in `incar` for NELECT checks!
    vf.user_input.incar["NELECT"] = 9
    vf.vasprun.final_structure._charge = 1.0
    run_check(vf, "NELECT", False)

    # POTIM check #2 (checks energy change between steps)
    vf = copy.deepcopy(vf_og)
    vf.user_input.incar["IBRION"] = 2
    temp_ionic_step_1 = copy.deepcopy(vf.vasprun.ionic_steps[0])
    temp_ionic_step_2 = copy.deepcopy(temp_ionic_step_1)
    temp_ionic_step_1["e_fr_energy"] = 0
    temp_ionic_step_2["e_fr_energy"] = 10000
    vf.vasprun.ionic_steps = [
        temp_ionic_step_1,
        temp_ionic_step_2,
    ]
    run_check(vf, "POTIM", False)

    # EDIFFG energy convergence check (this check SHOULD fail)
    vf = copy.deepcopy(vf_og)
    temp_ionic_step_1 = copy.deepcopy(vf.vasprun.ionic_steps[0])
    temp_ionic_step_2 = copy.deepcopy(temp_ionic_step_1)
    temp_ionic_step_1["e_0_energy"] = -1
    temp_ionic_step_2["e_0_energy"] = -2
    vf.vasprun.ionic_steps = [
        temp_ionic_step_1,
        temp_ionic_step_2,
    ]
    run_check(vf, "ENERGY CHANGE BETWEEN LAST TWO IONIC STEPS", False)

    # EDIFFG / force convergence check (the MP input set for R2SCAN has force convergence criteria)
    # (the below test should NOT fail, because final forces are 0)
    vf = copy.deepcopy(vf_og)
    vf.user_input.incar.update(METAGGA="R2SCA", ICHARG=1)
    vf.vasprun.ionic_steps[-1]["forces"] = [[0, 0, 0], [0, 0, 0]]
    run_check(vf, "MAX FINAL FORCE MAGNITUDE", True)

    # EDIFFG / force convergence check (the MP input set for R2SCAN has force convergence criteria)
    # (the below test SHOULD fail, because final forces are high)
    vf = copy.deepcopy(vf_og)
    vf.user_input.incar.update(METAGGA="R2SCA", ICHARG=1, IBRION=1, NSW=1)
    vf.vasprun.ionic_steps[-1]["forces"] = [[10, 10, 10], [10, 10, 10]]
    run_check(vf, "MAX FINAL FORCE MAGNITUDE", False)

    # ISMEAR wrong for nonmetal check
    vf = copy.deepcopy(vf_og)
    vf.user_input.incar["ISMEAR"] = 1
    vf.vasprun.bandgap = 1
    run_check(vf, "ISMEAR", False)

    # ISMEAR wrong for metal relaxation check
    vf = copy.deepcopy(vf_og)
    vf.user_input.incar.update(ISMEAR=-5, NSW=1, IBRION=1, ICHARG=9)
    vf.vasprun.bandgap = 0
    run_check(vf, "ISMEAR", False)

    # SIGMA too high for nonmetal with ISMEAR = 0 check
    vf = copy.deepcopy(vf_og)
    vf.user_input.incar.update(ISMEAR=0, SIGMA=0.2)
    vf.vasprun.bandgap = 1
    run_check(vf, "SIGMA", False)

    # SIGMA too high for nonmetal with ISMEAR = -5 check (should not error)
    vf = copy.deepcopy(vf_og)
    vf.user_input.incar.update(ISMEAR=-5, SIGMA=1e3)
    vf.vasprun.bandgap = 1
    run_check(vf, "SIGMA", True)

    # SIGMA too high for metal check
    vf = copy.deepcopy(vf_og)
    vf.user_input.incar.update(ISMEAR=1, SIGMA=0.5)
    vf.vasprun.bandgap = 0
    run_check(vf, "SIGMA", False)

    # SIGMA too large check (i.e. eentropy term is > 1 meV/atom)
    vf = copy.deepcopy(vf_og)
    vf.vasprun.ionic_steps[0]["electronic_steps"][-1]["eentropy"] = 1
    run_check(vf, "The entropy term (T*S)", False)

    # LMAXMIX check for SCF calc
    vf = copy.deepcopy(vf_og)
    vf.user_input.incar.update(
        LMAXMIX=0,
        ICHARG=1,
    )
    validated = VaspValidator.from_vasp_input(vasp_files=vf)
    # should not invalidate SCF calcs based on LMAXMIX
    assert not any(["LMAXMIX" in reason for reason in validated.reasons])
    # rather should add a warning
    assert any(["LMAXMIX" in warning for warning in validated.warnings])

    # EFERMI check (does not matter for VASP versions before 6.4)
    # must check EFERMI in the *incar*, as it is saved as a numerical value after VASP
    # guesses it in the vasprun.xml `parameters`
    vf = copy.deepcopy(vf_og)
    vf.vasprun.vasp_version = "5.4.4"
    vf.user_input.incar["EFERMI"] = 5
    run_check(vf, "EFERMI", True)

    # EFERMI check (matters for VASP versions 6.4 and beyond)
    # must check EFERMI in the *incar*, as it is saved as a numerical value after VASP
    # guesses it in the vasprun.xml `parameters`
    vf = copy.deepcopy(vf_og)
    vf.vasprun.vasp_version = "6.4.0"
    vf.user_input.incar["EFERMI"] = 5
    run_check(vf, "EFERMI", False)

    # LORBIT check (should have magnetization values for ISPIN=2)
    # Should be valid for this case, as no magmoms are expected for ISPIN = 1
    vf = copy.deepcopy(vf_og)
    vf.user_input.incar["ISPIN"] = 1
    vf.outcar.magnetization = []
    run_check(vf, "LORBIT", True)

    # LORBIT check (should have magnetization values for ISPIN=2)
    # Should be valid in this case, as magmoms are present for ISPIN = 2
    vf = copy.deepcopy(vf_og)
    vf.user_input.incar["ISPIN"] = 2
    vf.outcar.magnetization = (
        {"s": -0.0, "p": 0.0, "d": 0.0, "tot": 0.0},
        {"s": -0.0, "p": 0.0, "d": 0.0, "tot": -0.0},
    )
    run_check(vf, "LORBIT", True)

    # LORBIT check (should have magnetization values for ISPIN=2)
    # Should be invalid in this case, as no magmoms are present for ISPIN = 2
    vf = copy.deepcopy(vf_og)
    vf.user_input.incar["ISPIN"] = 2
    vf.outcar.magnetization = []
    run_check(vf, "LORBIT", False)

    # LMAXTAU check for METAGGA calcs (A value of 4 should fail for the `La` chemsys (has f electrons))
    vf = copy.deepcopy(vf_og)
    vf.user_input.structure = Structure(
        lattice=[[2.9, 0, 0], [0, 2.9, 0], [0, 0, 2.9]],
        species=["La", "La"],
        coords=[[0, 0, 0], [0.5, 0.5, 0.5]],
    )
    vf.user_input.incar.update(
        LMAXTAU=4,
        METAGGA="R2SCA",
        ICHARG=1,
    )
    run_check(vf, "LMAXTAU", False)


@pytest.mark.parametrize(
    "object_name",
    [
        "Si_uniform",
    ],
)
def test_nscf_checks(object_name):
    vf_og = vasp_calc_data[object_name]
    vf_og.vasprun.final_structure._charge = 0.0  # patch for old test files

    # ICHARG check
    run_check(vf_og, "ICHARG", True, incar_settings_to_change={"ICHARG": 11})

    # LMAXMIX check for NSCF calc
    vf = copy.deepcopy(vf_og)
    vf.user_input.incar["LMAXMIX"] = 0
    validated = VaspValidator.from_vasp_input(vasp_files=vf)
    # should invalidate NSCF calcs based on LMAXMIX
    assert any(["LMAXMIX" in reason for reason in validated.reasons])
    # and should *not* create a warning for NSCF calcs
    assert not any(["LMAXMIX" in warning for warning in validated.warnings])

    # Explicit kpoints for NSCF calc check (this should not raise any flags for NSCF calcs)
    vf = copy.deepcopy(vf_og)
    vf.user_input.kpoints = Kpoints.from_dict(
        {
            "kpoints": [[0, 0, 0], [0, 0, 0.5]],
            "nkpoints": 2,
            "kpts_weights": [0.5, 0.5],
            "labels": ["Gamma", "X"],
            "style": "line_mode",
            "generation_style": "line_mode",
        }
    )
    run_check(vf, "INPUT SETTINGS --> KPOINTS: explicitly", True)


@pytest.mark.parametrize(
    "object_name",
    [
        "Si_uniform",
    ],
)
def test_common_error_checks(object_name):
    vf_og = vasp_calc_data[object_name]
    vf_og.vasprun.final_structure._charge = 0.0  # patch for old test files

    # METAGGA and GGA tag check (should never be set together)
    with pytest.raises(ValidationError):
        vfd = vf_og.model_dump()
        vfd["user_input"]["incar"].update(
            GGA="PE",
            METAGGA="R2SCAN",
        )
        vf_new = VaspFiles(**vfd)
        vf_new.valid_input_set

    # Drift forces too high check - a warning
    vf = copy.deepcopy(vf_og)
    vf.outcar.drift = [[1, 1, 1]]
    validated = VaspValidator.from_vasp_input(vasp_files=vf)
    assert any("CONVERGENCE --> Excessive drift" in w for w in validated.warnings)

    # Final energy too high check
    vf = copy.deepcopy(vf_og)
    vf.vasprun.final_energy = 1e8
    run_check(vf, "LARGE POSITIVE FINAL ENERGY", False)

    # Excessive final magmom check (no elements Gd or Eu present)
    vf = copy.deepcopy(vf_og)
    vf.user_input.incar["ISPIN"] = 2
    vf.outcar.magnetization = [
        {"s": 9.0, "p": 0.0, "d": 0.0, "tot": 9.0},
        {"s": 9.0, "p": 0.0, "d": 0.0, "tot": 9.0},
    ]
    run_check(vf, "MAGNETISM", False)

    # Excessive final magmom check (elements Gd or Eu present)
    # Should pass here, as it has a final magmom < 10
    vf = copy.deepcopy(vf_og)
    vf.user_input.incar["ISPIN"] = 2
    vf.user_input.structure = Structure(
        lattice=[[2.9, 0, 0], [0, 2.9, 0], [0, 0, 2.9]],
        species=["Gd", "Eu"],
        coords=[[0, 0, 0], [0.5, 0.5, 0.5]],
    )
    vf.outcar.magnetization = (
        {"s": 9.0, "p": 0.0, "d": 0.0, "tot": 9.0},
        {"s": 9.0, "p": 0.0, "d": 0.0, "tot": 9.0},
    )
    run_check(vf, "MAGNETISM", True)

    # Excessive final magmom check (elements Gd or Eu present)
    # Should not pass here, as it has a final magmom > 10
    vf = copy.deepcopy(vf_og)
    vf.user_input.incar["ISPIN"] = 2
    vf.user_input.structure = Structure(
        lattice=[[2.9, 0, 0], [0, 2.9, 0], [0, 0, 2.9]],
        species=["Gd", "Eu"],
        coords=[[0, 0, 0], [0.5, 0.5, 0.5]],
    )
    vf.outcar.magnetization = (
        {"s": 11.0, "p": 0.0, "d": 0.0, "tot": 11.0},
        {"s": 11.0, "p": 0.0, "d": 0.0, "tot": 11.0},
    )
    run_check(vf, "MAGNETISM", False)

    # Element Po / Am present
    for unsupported_ele in ("Po", "Am"):
        vf = copy.deepcopy(vf_og)
        vf.user_input.structure.replace_species({ele: unsupported_ele for ele in vf.user_input.structure.elements})
        with pytest.raises(KeyError):
            run_check(vf, "COMPOSITION", False)


def _update_kpoints_for_test(vf: VaspFiles, kpoints_updates: dict | Kpoints) -> None:
    orig_kpoints = vf.user_input.kpoints.as_dict() if vf.user_input.kpoints else {}
    if isinstance(kpoints_updates, Kpoints):
        kpoints_updates = kpoints_updates.as_dict()
    orig_kpoints.update(kpoints_updates)
    vf.user_input.kpoints = Kpoints.from_dict(orig_kpoints)


@pytest.mark.parametrize("object_name", ["Si_old_double_relax"])
def test_kpoints_checks(object_name):
    vf_og = vasp_calc_data[object_name]
    vf_og.vasprun.final_structure._charge = 0.0  # patch for old test files

    # Valid mesh type check (should flag HCP structures)
    vf = copy.deepcopy(vf_og)
    vf.user_input.structure = Structure(
        lattice=[
            [0.5, -0.866025403784439, 0],
            [0.5, 0.866025403784439, 0],
            [0, 0, 1.6329931618554521],
        ],
        coords=[[0, 0, 0], [0.333333333333333, -0.333333333333333, 0.5]],
        species=["H", "H"],
    )  # HCP structure
    _update_kpoints_for_test(vf, {"generation_style": "monkhorst"})
    run_check(vf, "INPUT SETTINGS --> KPOINTS or KGAMMA:", False)

    # Valid mesh type check (should flag FCC structures)
    vf = copy.deepcopy(vf_og)
    vf.user_input.structure = Structure(
        lattice=[[0.0, 0.5, 0.5], [0.5, 0.0, 0.5], [0.5, 0.5, 0.0]],
        coords=[[0, 0, 0]],
        species=["H"],
    )  # FCC structure
    _update_kpoints_for_test(vf, {"generation_style": "monkhorst"})
    run_check(vf, "INPUT SETTINGS --> KPOINTS or KGAMMA:", False)

    # Valid mesh type check (should *not* flag BCC structures)
    vf = copy.deepcopy(vf_og)
    vf.user_input.structure = Structure(
        lattice=[[2.9, 0, 0], [0, 2.9, 0], [0, 0, 2.9]],
        species=["H", "H"],
        coords=[[0, 0, 0], [0.5, 0.5, 0.5]],
    )  # BCC structure
    _update_kpoints_for_test(vf, {"generation_style": "monkhorst"})
    run_check(vf, "INPUT SETTINGS --> KPOINTS or KGAMMA:", True)

    # Too few kpoints check
    vf = copy.deepcopy(vf_og)
    _update_kpoints_for_test(vf, {"kpoints": [[3, 3, 3]]})
    run_check(vf, "INPUT SETTINGS --> KPOINTS or KSPACING:", False)

    # Explicit kpoints for SCF calc check
    vf = copy.deepcopy(vf_og)
    _update_kpoints_for_test(
        vf,
        {
            "kpoints": [[0, 0, 0], [0, 0, 0.5]],
            "nkpoints": 2,
            "kpts_weights": [0.5, 0.5],
            "style": "reciprocal",
            "generation_style": "Reciprocal",
        },
    )
    run_check(vf, "INPUT SETTINGS --> KPOINTS: explicitly", False)

    # Shifting kpoints for SCF calc check
    vf = copy.deepcopy(vf_og)
    _update_kpoints_for_test(vf, {"usershift": [0.5, 0, 0]})
    run_check(vf, "INPUT SETTINGS --> KPOINTS: shifting", False)


@pytest.mark.parametrize("object_name", ["Si_old_double_relax"])
def test_vasp_version_check(object_name):
    vf_og = vasp_calc_data[object_name]
    vf_og.vasprun.final_structure._charge = 0.0  # patch for old test files

    vasp_version_list = [
        {"vasp_version": "4.0.0", "should_pass": False},
        {"vasp_version": "5.0.0", "should_pass": False},
        {"vasp_version": "5.4.0", "should_pass": False},
        {"vasp_version": "5.4.4", "should_pass": True},
        {"vasp_version": "6.0.0", "should_pass": True},
        {"vasp_version": "6.1.3", "should_pass": True},
        {"vasp_version": "6.4.2", "should_pass": True},
    ]

    for check_info in vasp_version_list:
        vf = copy.deepcopy(vf_og)
        vf.vasprun.vasp_version = check_info["vasp_version"]
        run_check(vf, "VASP VERSION", check_info["should_pass"])

    # Check for obscure VASP 5 bug with spin-polarized METAGGA calcs (should fail)
    vf = copy.deepcopy(vf_og)
    vf.vasprun.vasp_version = "5.0.0"
    vf.user_input.incar.update(
        METAGGA="R2SCAN",
        ISPIN=2,
    )
    run_check(vf, "POTENTIAL BUG --> We believe", False)


def test_fast_mode():
    vf = vasp_calc_data["Si_uniform"]
    validated = VaspValidator.from_vasp_input(vasp_files=vf, check_potcar=False)

    # Without POTCAR check, this doc is valid
    assert validated.valid

    # Now introduce sequence of changes to test how fast validation works
    # Check order:
    # 1. VASP version
    # 2. Common errors (known bugs, missing output, etc.)
    # 3. K-point density
    # 4. POTCAR check
    # 5. INCAR check

    og_kpoints = vf.user_input.kpoints
    # Introduce series of errors, then ablate them
    # use unacceptable version and set METAGGA and GGA simultaneously ->
    # should only get version error in reasons
    vf.vasprun.vasp_version = "4.0.0"
    vf.vasprun.parameters["NBANDS"] = -5
    # bad_incar_updates = {
    #     "METAGGA": "R2SCAN",
    #     "GGA": "PE",
    # }
    # vf.user_input.incar.update(bad_incar_updates)
    # print(vf.user_input.kpoints.as_dict)
    _update_kpoints_for_test(vf, {"kpoints": [[1, 1, 2]]})

    validated = VaspValidator.from_vasp_input(vasp_files=vf, check_potcar=True, fast=True)
    assert len(validated.reasons) == 1
    assert "VASP VERSION" in validated.reasons[0]

    # Now correct version, should just get METAGGA / GGA bug
    vf.vasprun.vasp_version = "6.3.2"
    # validated = VaspValidator.from_vasp_input(vf, check_potcar=True, fast=True)
    # assert len(validated.reasons) == 1
    # assert "KNOWN BUG" in validated.reasons[0]

    # Now remove GGA tag, get k-point density error
    # vf.user_input.incar.pop("GGA")
    validated = VaspValidator.from_vasp_input(vasp_files=vf, check_potcar=True, fast=True)
    assert len(validated.reasons) == 1
    assert "INPUT SETTINGS --> KPOINTS or KSPACING:" in validated.reasons[0]

    # Now restore k-points and don't check POTCAR --> get error
    _update_kpoints_for_test(vf, og_kpoints)
    validated = VaspValidator.from_vasp_input(vasp_files=vf, check_potcar=False, fast=True)
    assert len(validated.reasons) == 1
    assert "NBANDS" in validated.reasons[0]

    # Fix NBANDS, get no errors
    vf.vasprun.parameters["NBANDS"] = 10
    validated = VaspValidator.from_vasp_input(vasp_files=vf, check_potcar=True, fast=True)
    assert len(validated.reasons) == 0

    # Remove POTCAR, should fail validation
    vf.user_input.potcar = None
    validated = VaspValidator.from_vasp_input(vasp_files=vf, check_potcar=True, fast=True)
    assert "PSEUDOPOTENTIALS" in validated.reasons[0]


def test_site_properties(test_dir):

    vf = VaspFiles(**loadfn(test_dir / "vasp" / "mp-1245223_site_props_check.json.gz"))
    vd = VaspValidator.from_vasp_input(vasp_files=vf)

    assert not vd.valid
    assert any("selective dynamics" in reason.lower() for reason in vd.reasons)

    # map non-zero velocities to input structure and re-check
    vf.user_input.structure.add_site_property(
        "velocities", [[1.0, 2.0, 3.0] for _ in range(len(vf.user_input.structure))]
    )
    vd = VaspValidator.from_vasp_input(vasp_files=vf)
    assert any("non-zero velocities" in warning.lower() for warning in vd.warnings)
