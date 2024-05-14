from __future__ import annotations

from emmet.core.utils import jsanitize
from emmet.core.vasp.task_valid import TaskDocument
from jobflow import Flow
from monty.serialization import dumpfn
import numpy as np
from pymatgen.core import Structure, Lattice


def get_GaAs_structure(a0: float = 5.6) -> Structure:
    lattice_vectors = a0 * np.array([[0.0 if i == j else 0.5 for j in range(3)] for i in range(3)])
    return Structure(
        lattice=Lattice(lattice_vectors),
        species=["Ga", "As"],
        coords=[[0.125, 0.125, 0.125], [0.875, 0.875, 0.875]],
        coords_are_cartesian=False,
    )


def assign_meta(flow, metadata: dict, name: str | None = None):
    if hasattr(flow, "jobs"):
        for ijob in range(len(flow.jobs)):
            assign_meta(flow.jobs[ijob], metadata, name=name)
        if name:
            flow.name = name
    else:
        flow.metadata = metadata.copy()
        if name:
            flow.name = name


def get_MP_compliant_r2SCAN_flow(
    structure: Structure,
    user_incar_settings: dict | None = None,
    metadata: dict | None = None,
    name: str | None = None,
) -> Flow:
    from atomate2.vasp.jobs.mp import MPMetaGGAStaticMaker
    from atomate2.vasp.powerups import update_user_incar_settings

    maker = MPMetaGGAStaticMaker()

    user_incar_settings = user_incar_settings or {}
    if len(user_incar_settings) > 0:
        maker = update_user_incar_settings(maker, incar_updates=user_incar_settings)

    flow = maker.make(structure)

    metadata = metadata or {}
    assign_meta(flow, metadata, name=name)

    return flow


def run_job_fully_locally(flow, job_store=None):
    from jobflow import run_locally, JobStore
    from maggma.stores import MemoryStore

    if job_store is None:
        job_store = JobStore(MemoryStore(), additional_stores={"data": MemoryStore()})

    response = run_locally(flow, store=job_store, create_folders=True)
    uuid = list(response)[0]
    return response[uuid][1].output


def MP_compliant_calc():
    structure = get_GaAs_structure()
    flow = get_MP_compliant_r2SCAN_flow(
        structure=structure,
        user_incar_settings={  # some forward-looking settings
            "LREAL": False,
            "LMAXMIX": 6,
            "LCHARG": False,  # following tags just set for convenience
            "LWAVE": False,
            "LAECHG": False,
            "NCORE": 16,
            "KPAR": 2,
        },
    )
    return run_job_fully_locally(flow)


def MP_non_compliant_calc():
    structure = get_GaAs_structure()
    flow = get_MP_compliant_r2SCAN_flow(
        structure=structure,
        user_incar_settings={  # some backward-looking settings
            "ENCUT": 450.0,
            "ENAUG": 900.0,
            "KSPACING": 0.5,
            "LCHARG": False,  # following tags just set for convenience
            "LWAVE": False,
            "LAECHG": False,
            "NCORE": 16,
            "KPAR": 2,
        },
    )
    return run_job_fully_locally(flow)


def MP_flows() -> None:
    compliant_task_doc = MP_compliant_calc()
    dumpfn(jsanitize(compliant_task_doc), "./MP_compatible_GaAs_r2SCAN_static.json.gz")

    non_compliant_task_doc = MP_non_compliant_calc()
    dumpfn(
        jsanitize(non_compliant_task_doc),
        "./MP_incompatible_GaAs_r2SCAN_static.json.gz",
    )


def generate_task_documents(cdir, task_id: str | None = None, filename: str | None = None) -> TaskDocument:
    from atomate.vasp.drones import VaspDrone
    from emmet.core.mpid import MPID

    drone = VaspDrone(store_volumetric_data=[])
    task_doc_dict = drone.assimilate(cdir)

    task_id = task_id or "mp-100000000"
    task_doc_dict["task_id"] = MPID(task_id)
    task_doc = TaskDocument(**task_doc_dict)

    if filename:
        dumpfn(jsanitize(task_doc), filename)

    return task_doc


if __name__ == "__main__":
    MP_flows()
