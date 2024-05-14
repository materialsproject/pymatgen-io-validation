"""Module for checking if a structure is not a bulk crystal"""

from pymatgen.analysis.local_env import VoronoiNN
import numpy as np


def check_for_excess_empty_space(structure):
    """Relatively robust method for checking if a structure is a surface slab/1d structure/anything that is not a bulk crystal"""
    # Check 1: find large gaps along one of the defined lattice vectors
    lattice_vec_lengths = structure.lattice.lengths
    fcoords = np.array(structure.frac_coords)

    lattice_vec_1_frac_coords = list(np.sort(fcoords[:, 0]))
    lattice_vec_1_frac_coords.append(lattice_vec_1_frac_coords[0] + 1)
    lattice_vec_1_diffs = np.diff(lattice_vec_1_frac_coords)
    max_lattice_vec_1_empty_dist = max(lattice_vec_1_diffs) * lattice_vec_lengths[0]

    lattice_vec_2_frac_coords = list(np.sort(fcoords[:, 1]))
    lattice_vec_2_frac_coords.append(lattice_vec_2_frac_coords[0] + 1)
    lattice_vec_2_diffs = np.diff(lattice_vec_2_frac_coords)
    max_lattice_vec_2_empty_dist = max(lattice_vec_2_diffs) * lattice_vec_lengths[1]

    lattice_vec_3_frac_coords = list(np.sort(fcoords[:, 2]))
    lattice_vec_3_frac_coords.append(lattice_vec_3_frac_coords[0] + 1)
    lattice_vec_3_diffs = np.diff(lattice_vec_3_frac_coords)
    max_lattice_vec_3_empty_dist = max(lattice_vec_3_diffs) * lattice_vec_lengths[2]

    max_empty_distance = max(
        max_lattice_vec_1_empty_dist,
        max_lattice_vec_2_empty_dist,
        max_lattice_vec_3_empty_dist,
    )

    # Check 2: get max voronoi polyhedra volume in structure
    def get_max_voronoi_polyhedra_volume(structure):
        max_voronoi_polyhedra_vol = 0
        vnn = VoronoiNN().get_all_voronoi_polyhedra(structure)
        for polyhedra in vnn:
            for key in polyhedra.keys():
                cur_vol = polyhedra[key]["volume"]
                if cur_vol > max_voronoi_polyhedra_vol:
                    max_voronoi_polyhedra_vol = cur_vol
        return max_voronoi_polyhedra_vol

    max_voronoi_polyhedra_vol = 0
    try:
        max_voronoi_polyhedra_vol = get_max_voronoi_polyhedra_volume(structure)
    except Exception as e:
        if "No Voronoi neighbors found for site - try increasing cutoff".lower() in str(e).lower():
            try:
                structure.make_supercell(
                    2
                )  # to circumvent weird issue with voronoi class, though this decreases performance significantly.
                max_voronoi_polyhedra_vol = get_max_voronoi_polyhedra_volume(structure)
            except Exception:
                pass

        if "infinite vertex in the Voronoi construction".lower() in str(e).lower():
            print(f"{str(e)} As a result, this structure is marked as having excess empty space.")
            max_voronoi_polyhedra_vol = np.inf

    if (max_voronoi_polyhedra_vol > 25) or (max_voronoi_polyhedra_vol > 5 and max_empty_distance > 7.5):
        return True
    else:
        return False
