"""Define VASP defaults and input categories to check."""

from __future__ import annotations
from typing import TYPE_CHECKING
from enum import Enum

from pymatgen.io.validation.common import VALID_OPERATIONS, InvalidOperation

if TYPE_CHECKING:
    from typing import Any, Literal, Sequence


class InputCategory(Enum):
    """Predefined VASP input categories."""

    chemical_shift = "chemical shift"
    density_mixing = "density mixing"
    dft = "dft"
    dft_plus_u = "dft plus u"
    dipole = "dipole"
    electronic = "electronic"
    electronic_projector = "electronic projector"
    electronic_self_consistency = "electronic self consistency"
    fft = "fft"
    hybrid = "hybrid"
    ionic = "ionic"
    k_mesh = "k mesh"
    misc = "misc"
    misc_special = "misc special"
    ncl = "ncl"
    post_init = "post init"
    precision = "precision"
    smearing = "smearing"
    startup = "startup"
    symmetry = "symmetry"
    tddft = "tddft"
    write = "write"


class VaspParam:
    """Define the default value of a VASP parameter.

    Args:
        name (str) : the name of the INCAR keyword
        value (Any) : the default value of this parameter if
            statically assigned by VASP. If this parameter is dynamically assigned
            by VASP, set the default to None.
        tag (InputCategory, str) : the general category of input the tag belongs to.
            Used only to properly update INCAR fields in the same way VASP does.
        operation : str, Sequence of str, or None
            Mathematical operation used to determine if an input value is valid.
            See VALID_OPERATIONS for a list of possible operators.
            If a single str, this specifies one operation.
            Can be a list of valid operations.
        alias : str or None
            If a str, an alternate name for a parameter to use when reporting
            invalid values. A good example is ENCUT, which is set by the
            user, but is overwritten to ENMAX in the vasprun.xml parameters.
            In this case, `name = "ENMAX"` but `alias = "ENCUT"` to be informative.
            If None, it is set to `name`.
        tolerance : float, default = 1.e-4
            The tolerance used when evaluating approximate float equality.
        commment : str or None
            Additional information to pass to the user if a check fails.
    """

    __slots__: tuple[str, ...] = (
        "name",
        "value",
        "operation",
        "alias",
        "tag",
        "tolerance",
        "comment",
        "warning",
        "severity",
    )

    def __init__(
        self,
        name: str,
        value: Any,
        tag: InputCategory | str,
        operation: str | Sequence[str] | None = None,
        alias: str | None = None,
        tolerance: float = 1.0e-4,
        comment: str | None = None,
        warning: str | None = None,
        severity: Literal["reason", "warning"] = "reason",
    ) -> None:

        self.name = name
        self.value = value
        if (isinstance(operation, str) and operation not in VALID_OPERATIONS) or (
            isinstance(operation, list | tuple) and any(op not in VALID_OPERATIONS for op in operation)
        ):
            if isinstance(operation, list | tuple):
                operation = f"[{', '.join(operation)}]"
            raise InvalidOperation(operation)

        self.operation = operation
        self.alias = alias or name
        if isinstance(tag, str):
            if tag in InputCategory.__members__:
                tag = InputCategory[tag]
            else:
                tag = InputCategory(tag)
        self.tag = tag.name
        self.tolerance = tolerance
        self.comment = comment
        self.warning = warning

        if severity not in {"reason", "warning"}:
            raise ValueError(f"`severity` must either be 'reason' or 'warning', not {severity}")
        self.severity = severity

    def __getitem__(self, name: str) -> Any:
        """Make attributes subscriptable."""
        return getattr(self, name)

    def __setitem__(self, name: str, value: Any) -> None:
        """Allow dict-style attribute assignment."""
        setattr(self, name, value)

    def update(self, dct: dict[str, Any]) -> None:
        """Mimic dict update."""
        for k, v in dct.items():
            self[k] = v

    def as_dict(self) -> dict[str, Any]:
        """Convert to a dict."""
        return {k: getattr(self, k) for k in self.__slots__}


VASP_DEFAULTS_LIST = [
    VaspParam("ADDGRID", False, "fft", operation="=="),
    VaspParam("AEXX", 0.0, "hybrid", tolerance=0.0001),
    VaspParam("AGGAC", 1.0, "hybrid", tolerance=0.0001),
    VaspParam("AGGAX", 1.0, "hybrid", tolerance=0.0001),
    VaspParam("ALDAC", 1.0, "hybrid", tolerance=0.0001),
    VaspParam("ALDAX", 1.0, "hybrid", tolerance=0.0001),
    VaspParam("ALGO", "normal", "electronic self consistency"),
    VaspParam("AMGGAC", 1.0, "hybrid", tolerance=0.0001),
    VaspParam("AMGGAX", 1.0, "hybrid", tolerance=0.0001),
    VaspParam(
        "DEPER",
        0.3,
        "misc",
        operation="==",
        comment="According to the VASP manual, DEPER should not be set by the user.",
    ),
    VaspParam(
        "EBREAK", None, "post init", comment="According to the VASP manual, EBREAK should not be set by the user."
    ),
    VaspParam("EDIFF", 0.0001, "electronic", operation="<="),
    VaspParam("EFERMI", "LEGACY", "misc special"),
    VaspParam("EFIELD", 0.0, "dipole", operation="=="),
    VaspParam("ENAUG", float("inf"), "electronic"),
    VaspParam("ENINI", 0, "electronic"),
    VaspParam("ENMAX", float("inf"), "fft", operation=">=", alias="ENCUT"),
    VaspParam("EPSILON", 1.0, "dipole", operation="=="),
    VaspParam("GGA_COMPAT", True, "misc", operation="=="),
    VaspParam("IALGO", 38, "misc special", operation="in"),
    VaspParam("IBRION", 0, "ionic", operation="in"),
    VaspParam("ICHARG", 2, "startup"),
    VaspParam("ICORELEVEL", 0, "misc", operation="=="),
    VaspParam("IDIPOL", 0, "dipole", operation="=="),
    VaspParam("IMAGES", 0, "misc", operation="=="),
    VaspParam("INIWAV", 1, "startup", operation="=="),
    VaspParam("ISIF", 2, "ionic", operation=">=", comment="ISIF values < 2 do not output the complete stress tensor."),
    VaspParam("ISMEAR", 1, "smearing", operation="in"),
    VaspParam("ISPIN", 1, "misc special"),
    VaspParam("ISTART", 0, "startup", operation="in"),
    VaspParam("ISYM", 2, "symmetry", operation="in"),
    VaspParam("IVDW", 0, "misc", operation="=="),
    VaspParam("IWAVPR", None, "misc special"),
    VaspParam("KGAMMA", True, "k mesh"),
    VaspParam("KSPACING", 0.5, "k mesh"),
    VaspParam("LASPH", True, "misc", operation="=="),
    VaspParam("LBERRY", False, "misc", operation="=="),
    VaspParam("LCALCEPS", False, "misc", operation="=="),
    VaspParam("LCALCPOL", False, "misc", operation="=="),
    VaspParam("LCHIMAG", False, "chemical shift", operation="=="),
    VaspParam("LCORR", True, "misc special"),
    VaspParam("LDAU", False, "dft plus u"),
    VaspParam("LDAUJ", [], "dft plus u"),
    VaspParam("LDAUL", [], "dft plus u"),
    VaspParam("LDAUTYPE", 2, "dft plus u"),
    VaspParam("LDAUU", [], "dft plus u"),
    VaspParam("LDIPOL", False, "dipole", operation="=="),
    VaspParam("LEFG", False, "write", operation="=="),
    VaspParam("LEPSILON", False, "misc", operation="=="),
    VaspParam("LHFCALC", False, "hybrid"),
    VaspParam("LHYPERFINE", False, "misc", operation="=="),
    VaspParam("LKPOINTS_OPT", False, "misc", operation="=="),
    VaspParam("LKPROJ", False, "misc", operation="=="),
    VaspParam("LMAXMIX", 2, "density mixing"),
    VaspParam("LMAXPAW", -100, "electronic projector", operation="=="),
    VaspParam("LMAXTAU", 6, "density mixing"),
    VaspParam("LMONO", False, "dipole", operation="=="),
    VaspParam("LMP2LT", False, "misc", operation="=="),
    VaspParam("LNMR_SYM_RED", False, "chemical shift", operation="=="),
    VaspParam("LNONCOLLINEAR", False, "ncl", operation="=="),
    VaspParam("LOCPROJ", "NONE", "misc", operation="=="),
    VaspParam("LOPTICS", False, "tddft", operation="=="),
    VaspParam("LORBIT", None, "misc special"),
    VaspParam("LREAL", "false", "precision", operation="in"),
    VaspParam("LRPA", False, "misc", operation="=="),
    VaspParam("LSMP2LT", False, "misc", operation="=="),
    VaspParam("LSORBIT", False, "ncl", operation="=="),
    VaspParam("LSPECTRAL", False, "misc", operation="=="),
    VaspParam("LSUBROT", False, "misc", operation="=="),
    VaspParam("METAGGA", None, "dft"),
    VaspParam("ML_LMLFF", False, "misc", operation="=="),
    VaspParam("NELECT", None, "electronic"),
    VaspParam("NELM", 60, "electronic self consistency"),
    VaspParam("NELMDL", -5, "electronic self consistency"),
    VaspParam("NLSPLINE", False, "electronic projector", operation="=="),
    VaspParam("NSW", 0, "startup"),
    VaspParam(
        "NWRITE",
        2,
        "write",
        operation=">=",
        comment="The specified value of NWRITE does not output all needed information.",
    ),
    VaspParam("POTIM", 0.5, "ionic"),
    VaspParam("PREC", "NORMAL", "precision"),
    VaspParam("PSTRESS", 0.0, "ionic", operation="approx", tolerance=0.0001),
    VaspParam("RWIGS", [-1.0], "misc special"),
    VaspParam("SCALEE", 1.0, "ionic", operation="approx"),
    VaspParam("SIGMA", 0.2, "smearing"),
    VaspParam("SYMPREC", 1e-05, "symmetry"),
    VaspParam("VCA", [1.0], "misc special"),
    VaspParam("WEIMIN", 0.001, "misc", operation="<="),
]

VASP_DEFAULTS_DICT = {v.name: v for v in VASP_DEFAULTS_LIST}
