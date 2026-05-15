"""DEPRECATED: Module for checking if a structure's energy is within a certain distance of the MPDB hull"""

import warnings

warnings.warn(
    "`compare_to_MP_ehull` has been removed in favor of "
    "`mp_api.client.MPRester().get_stability` "
    "which provides identical functionality. "
    "This stub will be removed in the next version "
    "of pymatgen-io-validation.",
    stacklevel=2,
    category=DeprecationWarning,
)

compare_to_MP_ehull = None
