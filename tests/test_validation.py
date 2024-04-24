import pytest
import copy
from conftest import get_test_object, test_data_task_docs
from pymatgen.io.validation import ValidationDoc
from emmet.core.tasks import TaskDoc
from monty.serialization import loadfn
from pymatgen.core.structure import Structure
from pymatgen.io.vasp import Kpoints

### TODO: add tests for many other MP input sets (e.g. MPNSCFSet, MPNMRSet, MPScanRelaxSet, Hybrid sets, etc.)
### TODO: add check for an MP input set that uses an IBRION other than [-1, 1, 2]
### TODO: add in check for MP set where LEFG = True
### TODO: add in check for MP set where LOPTICS = True
### TODO: fix logic for calc_type / run_type identification in Emmet!!! Or handle how we interpret them...


def run_check(
    task_doc,
    error_message_to_search_for: str,
    should_the_check_pass: bool,
    vasprun_parameters_to_change: dict = {},  # for changing the parameters read from vasprun.xml
    incar_settings_to_change: dict = {},  # for directly changing the INCAR file,
    validation_doc_kwargs : dict = {}, # any kwargs to pass to the ValidationDoc class
):
    for key, value in vasprun_parameters_to_change.items():
        task_doc.input.parameters[key] = value

    for key, value in incar_settings_to_change.items():
        task_doc.calcs_reversed[0].input.incar[key] = value

    validation_doc = ValidationDoc.from_task_doc(task_doc,**validation_doc_kwargs)
    has_specified_error = any([error_message_to_search_for in reason for reason in validation_doc.reasons])

    assert (not has_specified_error) if should_the_check_pass else has_specified_error


@pytest.mark.parametrize(
    "object_name",
    [
        pytest.param("SiOptimizeDouble", id="SiOptimizeDouble"),
    ],
)
def test_validation_doc_from_directory(test_dir, object_name):
    test_object = get_test_object(object_name)
    dir_name = test_dir / "vasp" / test_object.folder
    test_validation_doc = ValidationDoc.from_directory(dir_name=dir_name)

    task_doc = test_data_task_docs[object_name]
    valid_validation_doc = ValidationDoc.from_task_doc(task_doc)

    # The attributes below will always be different because the objects are created at
    # different times. Hence, ignore before checking.
    delattr(test_validation_doc.builder_meta, "build_date")
    delattr(test_validation_doc, "last_updated")
    delattr(valid_validation_doc.builder_meta, "build_date")
    delattr(valid_validation_doc, "last_updated")

    assert test_validation_doc == valid_validation_doc


@pytest.mark.parametrize(
    "object_name",
    [
        pytest.param("SiOptimizeDouble", id="SiOptimizeDouble"),
    ],
)
def test_potcar_validation(test_dir, object_name):

    task_doc = test_data_task_docs[object_name]

    correct_potcar_summary_stats = loadfn(test_dir / "vasp" / "Si_potcar_spec.json.gz")

    # Check POTCAR (this test should PASS, as we ARE using a MP-compatible pseudopotential)
    temp_task_doc = copy.deepcopy(task_doc)
    temp_task_doc.calcs_reversed[0].input.potcar_spec = correct_potcar_summary_stats
    run_check(temp_task_doc, "PSEUDOPOTENTIALS", True)

    # Check POTCAR (this test should FAIL, as we are NOT using an MP-compatible pseudopotential)
    temp_task_doc = copy.deepcopy(task_doc)
    incorrect_potcar_summary_stats = copy.deepcopy(correct_potcar_summary_stats)
    incorrect_potcar_summary_stats[0].summary_stats["stats"]["data"]["MEAN"] = 999999999
    temp_task_doc.calcs_reversed[0].input.potcar_spec = incorrect_potcar_summary_stats
    run_check(temp_task_doc, "PSEUDOPOTENTIALS", False)


@pytest.mark.parametrize(
    "object_name",
    [
        pytest.param("SiOptimizeDouble", id="SiOptimizeDouble"),
        pytest.param("SiStatic", id="SiStatic"),
    ],
)
def test_scf_incar_checks(test_dir, object_name):
    task_doc = test_data_task_docs[object_name]
    task_doc.calcs_reversed[0].output.structure._charge = 0.0  # patch for old test files

    # Pay *very* close attention to whether a tag is modified in the incar or in the vasprun.xml's parameters!
    # Some parameters are validated from one or the other of these items, depending on whether VASP
    # changes the value between the INCAR and the vasprun.xml (which it often does)

    list_of_checks = loadfn(test_dir / "vasp" / "scf_incar_check_list.yaml")

    for check_info in list_of_checks:
        temp_task_doc = copy.deepcopy(task_doc)
        run_check(
            temp_task_doc,
            check_info["err_msg"],
            check_info["should_pass"],
            vasprun_parameters_to_change=check_info["vasprun"],
            incar_settings_to_change=check_info["incar"],
        )

    ### Most all of the tests below are too specific to use the kwargs in the
    # run_check() method. Hence, the calcs are manually modified. Apologies.

    # ENMAX / ENCUT checks
    # Also assert that the ENCUT warning does not assert that ENCUT >= inf
    # This checks that ENCUT is appropriately updated to be finite, and
    # not just ENMAX
    temp_task_doc = copy.deepcopy(task_doc)
    temp_task_doc.input.parameters["ENMAX"] = 1
    run_check(temp_task_doc, "ENCUT", False)
    run_check(temp_task_doc, "should be >= inf.", True)

    # NELECT check
    temp_task_doc = copy.deepcopy(task_doc)
    # must set NELECT in `incar` for NELECT checks!
    temp_task_doc.calcs_reversed[0].input.incar["NELECT"] = 9
    temp_task_doc.calcs_reversed[0].output.structure._charge = 1.0
    run_check(temp_task_doc, "NELECT", False)

    # FFT grid check (NGX, NGY, NGZ, NGXF, NGYF, NGZF)
    # Must change `incar` *and* `parameters` for NG_ checks!
    ng_keys = []
    for direction in ["X", "Y", "Z"]:
        for mod in ["", "F"]:
            ng_keys.append(f"NG{direction}{mod}")

    for key in ng_keys:
        temp_task_doc = copy.deepcopy(task_doc)
        temp_task_doc.calcs_reversed[0].input.incar[key] = 1
        temp_task_doc.input.parameters[key] = 1
        run_check(temp_task_doc, key, False)

    # POTIM check #1 (checks parameter itself)
    ### TODO: add in second check for POTIM that checks for large energy changes between ionic steps
    temp_task_doc = copy.deepcopy(task_doc)
    temp_task_doc.input.parameters["POTIM"] = 10
    run_check(temp_task_doc, "POTIM", False)

    # POTIM check #2 (checks energy change between steps)
    temp_task_doc = copy.deepcopy(task_doc)
    temp_task_doc.input.parameters["IBRION"] = 2
    temp_ionic_step_1 = copy.deepcopy(temp_task_doc.calcs_reversed[0].output.ionic_steps[0])
    temp_ionic_step_2 = copy.deepcopy(temp_ionic_step_1)
    temp_ionic_step_1.e_fr_energy = 0
    temp_ionic_step_2.e_fr_energy = 10000
    temp_task_doc.calcs_reversed[0].output.ionic_steps = [temp_ionic_step_1, temp_ionic_step_2]
    run_check(temp_task_doc, "POTIM", False)

    # EDIFFG energy convergence check (this check should not raise any invalid reasons)
    temp_task_doc = copy.deepcopy(task_doc)
    run_check(temp_task_doc, "ENERGY CHANGE BETWEEN LAST TWO IONIC STEPS", True)

    # EDIFFG energy convergence check (this check SHOULD fail)
    temp_task_doc = copy.deepcopy(task_doc)
    temp_ionic_step_1 = copy.deepcopy(temp_task_doc.calcs_reversed[0].output.ionic_steps[0])
    temp_ionic_step_2 = copy.deepcopy(temp_ionic_step_1)
    temp_ionic_step_1.e_0_energy = -1
    temp_ionic_step_2.e_0_energy = -2
    temp_task_doc.calcs_reversed[0].output.ionic_steps = [temp_ionic_step_1, temp_ionic_step_2]
    run_check(temp_task_doc, "ENERGY CHANGE BETWEEN LAST TWO IONIC STEPS", False)

    # EDIFFG / force convergence check (the MP input set for R2SCAN has force convergence criteria)
    # (the below test should NOT fail, because final forces are 0)
    temp_task_doc = copy.deepcopy(task_doc)
    temp_task_doc.calcs_reversed[0].input.incar["METAGGA"] = "R2SCAN"
    temp_task_doc.output.forces = [[0, 0, 0], [0, 0, 0]]
    run_check(temp_task_doc, "MAX FINAL FORCE MAGNITUDE", True)

    # EDIFFG / force convergence check (the MP input set for R2SCAN has force convergence criteria)
    # (the below test SHOULD fail, because final forces are high)
    temp_task_doc = copy.deepcopy(task_doc)
    temp_task_doc.calcs_reversed[0].input.incar["METAGGA"] = "R2SCAN"
    temp_task_doc.output.forces = [[10, 10, 10], [10, 10, 10]]
    run_check(temp_task_doc, "MAX FINAL FORCE MAGNITUDE", False)

    # ISMEAR wrong for nonmetal check
    temp_task_doc = copy.deepcopy(task_doc)
    temp_task_doc.input.parameters["ISMEAR"] = 1
    temp_task_doc.output.bandgap = 1
    run_check(temp_task_doc, "ISMEAR", False)

    # ISMEAR wrong for metal relaxation check
    temp_task_doc = copy.deepcopy(task_doc)
    # make ionic_steps be length 2, meaning this gets classified as a relaxation calculation
    temp_task_doc.calcs_reversed[0].output.ionic_steps = 2 * temp_task_doc.calcs_reversed[0].output.ionic_steps
    temp_task_doc.input.parameters["ISMEAR"] = -5
    temp_task_doc.output.bandgap = 0
    run_check(temp_task_doc, "ISMEAR", False)

    # SIGMA too high for nonmetal with ISMEAR = 0 check
    temp_task_doc = copy.deepcopy(task_doc)
    temp_task_doc.input.parameters["ISMEAR"] = 0
    temp_task_doc.input.parameters["SIGMA"] = 0.2
    temp_task_doc.output.bandgap = 1
    run_check(temp_task_doc, "SIGMA", False)

    # SIGMA too high for nonmetal with ISMEAR = -5 check (should not error)
    temp_task_doc = copy.deepcopy(task_doc)
    temp_task_doc.input.parameters["ISMEAR"] = -5
    temp_task_doc.input.parameters["SIGMA"] = 1000  # should not matter
    temp_task_doc.output.bandgap = 1
    run_check(temp_task_doc, "SIGMA", True)

    # SIGMA too high for metal check
    temp_task_doc = copy.deepcopy(task_doc)
    temp_task_doc.input.parameters["ISMEAR"] = 1
    temp_task_doc.input.parameters["SIGMA"] = 0.5
    temp_task_doc.output.bandgap = 0
    run_check(temp_task_doc, "SIGMA", False)

    # SIGMA too large check (i.e. eentropy term is > 1 meV/atom)
    temp_task_doc = copy.deepcopy(task_doc)
    temp_task_doc.calcs_reversed[0].output.ionic_steps[0].electronic_steps[-1].eentropy = 1
    run_check(temp_task_doc, "The entropy term (T*S)", False)

    # LMAXMIX check for SCF calc
    temp_task_doc = copy.deepcopy(task_doc)
    temp_task_doc.input.parameters["LMAXMIX"] = 0
    temp_validation_doc = ValidationDoc.from_task_doc(temp_task_doc)
    # should not invalidate SCF calcs based on LMAXMIX
    assert not any(["LMAXMIX" in reason for reason in temp_validation_doc.reasons])
    # rather should add a warning
    assert any(["LMAXMIX" in warning for warning in temp_validation_doc.warnings])

    # EFERMI check (does not matter for VASP versions before 6.4)
    # must check EFERMI in the *incar*, as it is saved as a numerical value after VASP
    # guesses it in the vasprun.xml `parameters`
    temp_task_doc = copy.deepcopy(task_doc)
    temp_task_doc.calcs_reversed[0].vasp_version = "5.4.4"
    temp_task_doc.calcs_reversed[0].input.incar["EFERMI"] = 5
    run_check(temp_task_doc, "EFERMI", True)

    # EFERMI check (matters for VASP versions 6.4 and beyond)
    # must check EFERMI in the *incar*, as it is saved as a numerical value after VASP
    # guesses it in the vasprun.xml `parameters`
    temp_task_doc = copy.deepcopy(task_doc)
    temp_task_doc.calcs_reversed[0].vasp_version = "6.4.0"
    temp_task_doc.calcs_reversed[0].input.incar["EFERMI"] = 5
    run_check(temp_task_doc, "EFERMI", False)

    # LORBIT check (should have magnetization values for ISPIN=2)
    # Should be valid for this case, as no magmoms are expected for ISPIN = 1
    temp_task_doc = copy.deepcopy(task_doc)
    temp_task_doc.input.parameters["ISPIN"] = 1
    temp_task_doc.calcs_reversed[0].output.outcar["magnetization"] = []
    run_check(temp_task_doc, "LORBIT", True)

    # LORBIT check (should have magnetization values for ISPIN=2)
    # Should be valid in this case, as magmoms are present for ISPIN = 2
    temp_task_doc = copy.deepcopy(task_doc)
    temp_task_doc.input.parameters["ISPIN"] = 2
    temp_task_doc.calcs_reversed[0].output.outcar["magnetization"] = (
        {"s": -0.0, "p": 0.0, "d": 0.0, "tot": 0.0},
        {"s": -0.0, "p": 0.0, "d": 0.0, "tot": -0.0},
    )
    run_check(temp_task_doc, "LORBIT", True)

    # LORBIT check (should have magnetization values for ISPIN=2)
    # Should be invalid in this case, as no magmoms are present for ISPIN = 2
    temp_task_doc = copy.deepcopy(task_doc)
    temp_task_doc.input.parameters["ISPIN"] = 2
    temp_task_doc.calcs_reversed[0].output.outcar["magnetization"] = []
    run_check(temp_task_doc, "LORBIT", False)

    # LMAXTAU check for METAGGA calcs (A value of 4 should fail for the `La` chemsys (has f electrons))
    temp_task_doc = copy.deepcopy(task_doc)
    temp_task_doc.chemsys = "La"
    temp_task_doc.calcs_reversed[0].input.structure = Structure(
        lattice=[[2.9, 0, 0], [0, 2.9, 0], [0, 0, 2.9]], species=["La", "La"], coords=[[0, 0, 0], [0.5, 0.5, 0.5]]
    )
    temp_task_doc.calcs_reversed[0].input.incar["LMAXTAU"] = 4
    temp_task_doc.calcs_reversed[0].input.incar["METAGGA"] = "R2SCAN"
    run_check(temp_task_doc, "LMAXTAU", False)

    # LMAXTAU check for METAGGA calcs (A value of 2 should fail for the `Si` chemsys)
    temp_task_doc = copy.deepcopy(task_doc)
    temp_task_doc.calcs_reversed[0].input.incar["LMAXTAU"] = 2
    temp_task_doc.calcs_reversed[0].input.incar["METAGGA"] = "R2SCAN"
    run_check(temp_task_doc, "LMAXTAU", False)

    # LMAXTAU should always pass for non-METAGGA calcs
    temp_task_doc = copy.deepcopy(task_doc)
    temp_task_doc.calcs_reversed[0].input.incar["LMAXTAU"] = 0
    temp_task_doc.calcs_reversed[0].input.incar["METAGGA"] = "None"
    run_check(temp_task_doc, "LMAXTAU", True)

    # ENAUG check for r2SCAN calcs
    temp_task_doc = copy.deepcopy(task_doc)
    temp_task_doc.input.parameters["ENAUG"] = 1
    temp_task_doc.calcs_reversed[0].input.incar["METAGGA"] = "R2SCAN"
    run_check(temp_task_doc, "ENAUG", False)


@pytest.mark.parametrize(
    "object_name",
    [
        pytest.param("SiNonSCFUniform", id="SiNonSCFUniform"),
    ],
)
def test_nscf_incar_checks(object_name):
    task_doc = test_data_task_docs[object_name]
    task_doc.calcs_reversed[0].output.structure._charge = 0.0  # patch for old test files

    # ICHARG check
    temp_task_doc = copy.deepcopy(task_doc)
    temp_task_doc.input.parameters["ICHARG"] = 11
    run_check(temp_task_doc, "ICHARG", True)

    # LMAXMIX check for NSCF calc
    temp_task_doc = copy.deepcopy(task_doc)
    temp_task_doc.input.parameters["LMAXMIX"] = 0
    temp_validation_doc = ValidationDoc.from_task_doc(temp_task_doc)
    # should invalidate NSCF calcs based on LMAXMIX
    assert any(["LMAXMIX" in reason for reason in temp_validation_doc.reasons])
    # and should *not* create a warning for NSCF calcs
    assert not any(["LMAXMIX" in warning for warning in temp_validation_doc.warnings])


@pytest.mark.parametrize(
    "object_name",
    [
        pytest.param("SiNonSCFUniform", id="SiNonSCFUniform"),
    ],
)
def test_nscf_kpoints_checks(object_name):
    task_doc = test_data_task_docs[object_name]
    task_doc.calcs_reversed[0].output.structure._charge = 0.0  # patch for old test files

    # Explicit kpoints for NSCF calc check (this should not raise any flags for NSCF calcs)
    temp_task_doc = copy.deepcopy(task_doc)
    _update_kpoints_for_test(
        temp_task_doc,
        {
            "kpoints": [[0, 0, 0], [0, 0, 0.5]],
            "nkpoints": 2,
            "kpts_weights": [0.5, 0.5],
            "labels": ["Gamma", "X"],
            "style": "line_mode",
            "generation_style": "line_mode",
        },
    )
    run_check(temp_task_doc, "INPUT SETTINGS --> KPOINTS: explicitly", True)


@pytest.mark.parametrize(
    "object_name",
    [
        pytest.param("SiOptimizeDouble", id="SiOptimizeDouble"),
        # pytest.param("SiStatic", id="SiStatic"),
    ],
)
def test_common_error_checks(object_name):
    task_doc = test_data_task_docs[object_name]
    task_doc.calcs_reversed[0].output.structure._charge = 0.0  # patch for old test files

    # METAGGA and GGA tag check (should never be set together)
    temp_task_doc = copy.deepcopy(task_doc)
    temp_task_doc.calcs_reversed[0].input.incar["METAGGA"] = "R2SCAN"
    temp_task_doc.calcs_reversed[0].input.incar["GGA"] = "PE"
    run_check(temp_task_doc, "KNOWN BUG", False)

    # METAGGA and GGA tag check (should not flag any reasons when METAGGA set to None)
    temp_task_doc = copy.deepcopy(task_doc)
    temp_task_doc.calcs_reversed[0].input.incar["METAGGA"] = "None"
    temp_task_doc.calcs_reversed[0].input.incar["GGA"] = "PE"
    run_check(temp_task_doc, "KNOWN BUG", True)

    # No electronic convergence check (i.e. more electronic steps than NELM)
    temp_task_doc = copy.deepcopy(task_doc)
    temp_task_doc.input.parameters["NELM"] = 1
    run_check(temp_task_doc, "CONVERGENCE --> Did not achieve electronic", False)

    # Drift forces too high check
    temp_task_doc = copy.deepcopy(task_doc)
    temp_task_doc.calcs_reversed[0].output.outcar["drift"] = [[1, 1, 1]]
    run_check(temp_task_doc, "CONVERGENCE --> Excessive drift", False)

    # Final energy too high check
    temp_task_doc = copy.deepcopy(task_doc)
    temp_task_doc.output.energy_per_atom = 100
    run_check(temp_task_doc, "LARGE POSITIVE FINAL ENERGY", False)

    # Excessive final magmom check (no elements Gd or Eu present)
    temp_task_doc = copy.deepcopy(task_doc)
    temp_task_doc.input.parameters["ISPIN"] = 2
    temp_task_doc.calcs_reversed[0].output.outcar["magnetization"] = (
        {"s": 9.0, "p": 0.0, "d": 0.0, "tot": 9.0},
        {"s": 9.0, "p": 0.0, "d": 0.0, "tot": 9.0},
    )
    run_check(temp_task_doc, "MAGNETISM", False)

    # Excessive final magmom check (elements Gd or Eu present)
    # Should pass here, as it has a final magmom < 10
    temp_task_doc = copy.deepcopy(task_doc)
    temp_task_doc.input.parameters["ISPIN"] = 2
    temp_task_doc.calcs_reversed[0].input.structure = Structure(
        lattice=[[2.9, 0, 0], [0, 2.9, 0], [0, 0, 2.9]], species=["Gd", "Eu"], coords=[[0, 0, 0], [0.5, 0.5, 0.5]]
    )
    temp_task_doc.calcs_reversed[0].output.outcar["magnetization"] = (
        {"s": 9.0, "p": 0.0, "d": 0.0, "tot": 9.0},
        {"s": 9.0, "p": 0.0, "d": 0.0, "tot": 9.0},
    )
    run_check(temp_task_doc, "MAGNETISM", True)

    # Excessive final magmom check (elements Gd or Eu present)
    # Should not pass here, as it has a final magmom > 10
    temp_task_doc = copy.deepcopy(task_doc)
    temp_task_doc.input.parameters["ISPIN"] = 2
    temp_task_doc.calcs_reversed[0].input.structure = Structure(
        lattice=[[2.9, 0, 0], [0, 2.9, 0], [0, 0, 2.9]], species=["Gd", "Eu"], coords=[[0, 0, 0], [0.5, 0.5, 0.5]]
    )
    temp_task_doc.calcs_reversed[0].output.outcar["magnetization"] = (
        {"s": 11.0, "p": 0.0, "d": 0.0, "tot": 11.0},
        {"s": 11.0, "p": 0.0, "d": 0.0, "tot": 11.0},
    )
    run_check(temp_task_doc, "MAGNETISM", False)

    # Element Po present
    temp_task_doc = copy.deepcopy(task_doc)
    temp_task_doc.chemsys = "Po"
    run_check(temp_task_doc, "COMPOSITION", False)

    # Elements Am present check
    temp_task_doc = copy.deepcopy(task_doc)
    temp_task_doc.chemsys = "Am"
    run_check(temp_task_doc, "COMPOSITION", False)


def _update_kpoints_for_test(task_doc: TaskDoc, kpoints_updates: dict):
    if isinstance(task_doc.calcs_reversed[0].input.kpoints, Kpoints):
        kpoints = task_doc.calcs_reversed[0].input.kpoints.as_dict()
    elif isinstance(task_doc.calcs_reversed[0].input.kpoints, dict):
        kpoints = task_doc.calcs_reversed[0].input.kpoints.copy()
    kpoints.update(kpoints_updates)
    task_doc.calcs_reversed[0].input.kpoints = Kpoints.from_dict(kpoints)


@pytest.mark.parametrize(
    "object_name",
    [
        pytest.param("SiOptimizeDouble", id="SiOptimizeDouble"),
    ],
)
def test_kpoints_checks(object_name):
    task_doc = test_data_task_docs[object_name]
    task_doc.calcs_reversed[0].output.structure._charge = 0.0  # patch for old test files

    # Valid mesh type check (should flag HCP structures)
    temp_task_doc = copy.deepcopy(task_doc)
    temp_task_doc.calcs_reversed[0].input.structure = Structure(
        lattice=[[0.5, -0.866025403784439, 0], [0.5, 0.866025403784439, 0], [0, 0, 1.6329931618554521]],
        coords=[[0, 0, 0], [0.333333333333333, -0.333333333333333, 0.5]],
        species=["H", "H"],
    )  # HCP structure
    _update_kpoints_for_test(temp_task_doc, {"generation_style": "monkhorst"})
    run_check(temp_task_doc, "INPUT SETTINGS --> KPOINTS or KGAMMA:", False)

    # Valid mesh type check (should flag FCC structures)
    temp_task_doc = copy.deepcopy(task_doc)
    temp_task_doc.calcs_reversed[0].input.structure = Structure(
        lattice=[[0.0, 0.5, 0.5], [0.5, 0.0, 0.5], [0.5, 0.5, 0.0]], coords=[[0, 0, 0]], species=["H"]
    )  # FCC structure
    _update_kpoints_for_test(temp_task_doc, {"generation_style": "monkhorst"})
    run_check(temp_task_doc, "INPUT SETTINGS --> KPOINTS or KGAMMA:", False)

    # Valid mesh type check (should *not* flag BCC structures)
    temp_task_doc = copy.deepcopy(task_doc)
    temp_task_doc.calcs_reversed[0].input.structure = Structure(
        lattice=[[2.9, 0, 0], [0, 2.9, 0], [0, 0, 2.9]], species=["H", "H"], coords=[[0, 0, 0], [0.5, 0.5, 0.5]]
    )  # BCC structure
    _update_kpoints_for_test(temp_task_doc, {"generation_style": "monkhorst"})
    run_check(temp_task_doc, "INPUT SETTINGS --> KPOINTS or KGAMMA:", True)

    # Too few kpoints check
    temp_task_doc = copy.deepcopy(task_doc)
    _update_kpoints_for_test(temp_task_doc, {"kpoints": [[3, 3, 3]]})
    run_check(temp_task_doc, "INPUT SETTINGS --> KPOINTS or KSPACING:", False)

    # Explicit kpoints for SCF calc check
    temp_task_doc = copy.deepcopy(task_doc)
    _update_kpoints_for_test(
        temp_task_doc,
        {
            "kpoints": [[0, 0, 0], [0, 0, 0.5]],
            "nkpoints": 2,
            "kpts_weights": [0.5, 0.5],
            "style": "reciprocal",
            "generation_style": "Reciprocal",
        },
    )
    run_check(temp_task_doc, "INPUT SETTINGS --> KPOINTS: explicitly", False)

    # Shifting kpoints for SCF calc check
    temp_task_doc = copy.deepcopy(task_doc)
    _update_kpoints_for_test(temp_task_doc, {"usershift": [0.5, 0, 0]})
    run_check(temp_task_doc, "INPUT SETTINGS --> KPOINTS: shifting", False)


@pytest.mark.parametrize(
    "object_name",
    [
        pytest.param("SiOptimizeDouble", id="SiOptimizeDouble"),
    ],
)
def test_vasp_version_check(object_name):
    task_doc = test_data_task_docs[object_name]
    task_doc.calcs_reversed[0].output.structure._charge = 0.0  # patch for old test files

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
        temp_task_doc = copy.deepcopy(task_doc)
        temp_task_doc.calcs_reversed[0].vasp_version = check_info["vasp_version"]
        run_check(temp_task_doc, "VASP VERSION", check_info["should_pass"])

    # Check for obscure VASP 5 bug with spin-polarized METAGGA calcs (should fail)
    temp_task_doc = copy.deepcopy(task_doc)
    temp_task_doc.calcs_reversed[0].vasp_version = "5.0.0"
    temp_task_doc.calcs_reversed[0].input.incar["METAGGA"] = "R2SCAN"
    temp_task_doc.input.parameters["ISPIN"] = 2
    run_check(temp_task_doc, "POTENTIAL BUG --> We believe", False)


def test_task_document(test_dir):

    from emmet.core.vasp.task_valid import TaskDocument

    calcs = {}
    calcs["compliant"] = loadfn(
        str(test_dir / "vasp" / "TaskDocuments" / "MP_compatible_GaAs_r2SCAN_static_TaskDocument.json.gz"), cls=None
    )
    calcs["non-compliant"] = loadfn(
        str(test_dir / "vasp" / "TaskDocuments" / "MP_incompatible_GaAs_r2SCAN_static_TaskDocument.json.gz"), cls=None
    )

    valid_docs = {}
    for calc in calcs:
        valid_docs[calc] = ValidationDoc.from_task_doc(TaskDocument(**calcs[calc]))
        # quickly check that `from_dict` and `from_task_doc` give same document
        assert set(ValidationDoc.from_dict(calcs[calc]).reasons) == set(valid_docs[calc].reasons)

    assert valid_docs["compliant"].valid
    assert not valid_docs["non-compliant"].valid

    expected_reasons = ["KPOINTS", "ENCUT", "ENAUG"]
    for expected_reason in expected_reasons:
        assert any(expected_reason in reason for reason in valid_docs["non-compliant"].reasons)        

def test_fast_mode():
    task_doc = test_data_task_docs["SiStatic"]
    valid_doc = ValidationDoc.from_task_doc(task_doc,check_potcar=False)

    # Without POTCAR check, this doc is valid
    assert valid_doc.valid

    # Now introduce sequence of changes to test how fast validation works
    # Check order:
    # 1. VASP version
    # 2. Common errors (known bugs, missing output, etc.)
    # 3. K-point density
    # 4. POTCAR check
    # 5. INCAR check

    og_kpoints = task_doc.calcs_reversed[0].input.kpoints
    # Introduce series of errors, then ablate them
    # use unacceptable version and set METAGGA and GGA simultaneously ->
    # should only get version error in reasons
    task_doc.calcs_reversed[0].vasp_version = "4.0.0"
    task_doc.input.parameters["NBANDS"] = -5
    bad_incar_updates = {"METAGGA": "R2SCAN", "GGA": "PE",}
    task_doc.calcs_reversed[0].input.incar.update(bad_incar_updates)
    
    _update_kpoints_for_test(task_doc, {"kpoints": [[1,1,2]]})
    
    valid_doc = ValidationDoc.from_task_doc(task_doc, check_potcar = True, fast = True)
    assert len(valid_doc.reasons) == 1
    assert "VASP VERSION" in valid_doc.reasons[0]

    # Now correct version, should just get METAGGA / GGA bug
    task_doc.calcs_reversed[0].vasp_version = "6.3.2"
    valid_doc = ValidationDoc.from_task_doc(task_doc, check_potcar = True, fast = True)
    assert len(valid_doc.reasons) == 1
    assert "KNOWN BUG" in valid_doc.reasons[0]

    # Now remove GGA tag, get k-point density error
    task_doc.calcs_reversed[0].input.incar.pop("GGA")
    valid_doc = ValidationDoc.from_task_doc(task_doc, check_potcar = True, fast = True)
    assert len(valid_doc.reasons) == 1
    assert "INPUT SETTINGS --> KPOINTS or KSPACING:" in valid_doc.reasons[0]

    # Now restore k-points and check POTCAR --> get error
    _update_kpoints_for_test(task_doc, og_kpoints)
    valid_doc = ValidationDoc.from_task_doc(task_doc, check_potcar = True, fast = True)
    assert len(valid_doc.reasons) == 1
    assert "PSEUDOPOTENTIALS" in valid_doc.reasons[0]

    # Without POTCAR check, should get INCAR check error for NGX
    valid_doc = ValidationDoc.from_task_doc(task_doc, check_potcar = False, fast = True)
    assert len(valid_doc.reasons) == 1
    assert "NBANDS" in valid_doc.reasons[0]
