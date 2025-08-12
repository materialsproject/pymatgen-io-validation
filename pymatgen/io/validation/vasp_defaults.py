"""Define VASP defaults and input categories to check."""

from __future__ import annotations
from typing import Any, Literal, Optional
import math
from pathlib import Path
from pydantic import BaseModel, Field, field_validator
from enum import Enum

VALID_OPERATIONS: set[str | None] = {
    "==",
    ">",
    ">=",
    "<",
    "<=",
    "in",
    "approx",
    "auto fail",
    None,
}


class InvalidOperation(Exception):
    """Define custom exception when checking valid operations."""

    def __init__(self, operation: str) -> None:
        """Define custom exception when checking valid operations.

        Args:
        operation (str) : a symbolic string for an operation that is not valid.
        """
        msg = f"Unknown operation type {operation}; valid values are: {VALID_OPERATIONS}"
        super().__init__(msg)


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


class VaspParam(BaseModel):
    """Define a schema for validating VASP parameters."""

    name: str = Field(description="The name of the INCAR keyword")
    value: Any = Field(
        description="The default value of this parameter if statically assigned by VASP. If this parameter is dynamically assigned by VASP, set the default to None."
    )
    tag: str = Field(
        description="the general category of input the tag belongs to. Used only to properly update INCAR fields in the same way VASP does."
    )
    operation: Optional[str | list[str] | tuple[str]] = Field(
        None, description="One or more of VALID_OPERATIONS to apply in validating this parameter."
    )
    alias: Optional[str] = Field(
        None,
        description="If a str, an alternate name for a parameter to use when reporting invalid values, e.g., ENMAX instead of ENCUT.",
    )
    tolerance: float = Field(1e-4, description="The tolerance used when evaluating approximate float equality.")
    comment: Optional[str] = Field(None, description="Additional information to pass to the user if a check fails.")
    warning: Optional[str] = Field(None, description="Additional warnings to pass to the user if a check fails.")
    severity: Literal["reason", "warning"] = Field("reason", description="The severity of failing this check.")

    @staticmethod
    def listify(val: Any) -> list[Any]:
        """Return scalars as list of single scalar.

        Parameters
        -----------
        val (Any) : scalar or vector-like

        Returns
        -----------
        list containing val if val was a scalar,
        otherwise the list version of val.
        """
        if hasattr(val, "__len__"):
            if isinstance(val, str):
                return [val]
            return list(val)
        return [val]

    @field_validator("operation", mode="after")
    @classmethod
    def set_operation(cls, v):
        """Check operations."""

        list_v = cls.listify(v)
        if not all(v in VALID_OPERATIONS for v in list_v):
            raise InvalidOperation(f"[{', '.join(v for v in list_v if v not in VALID_OPERATIONS)}]")
        return v

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

    @staticmethod
    def _comparator(lhs: Any, operation: str | None, rhs: Any, **kwargs) -> bool:
        """
        Compare different values using one of VALID_OPERATIONS.

        Parameters
        -----------
        lhs : Any
            Left-hand side of the operation.
        operation : str or None
            Operation acting on rhs from lhs. For example, if operation is ">",
            this returns (lhs > rhs).
            Check is skipped if operation is None
        rhs : Any
            Right-hand of the operation.
        kwargs
            If needed, kwargs to pass to operation.
        """
        if operation is None:
            c = True
        elif operation == "auto fail":
            c = False
        elif operation == "==":
            c = lhs == rhs
        elif operation == ">":
            c = lhs > rhs
        elif operation == ">=":
            c = lhs >= rhs
        elif operation == "<":
            c = lhs < rhs
        elif operation == "<=":
            c = lhs <= rhs
        elif operation == "in":
            c = lhs in rhs
        elif operation == "approx":
            c = math.isclose(lhs, rhs, **kwargs)
        else:
            raise InvalidOperation(operation)
        return c

    def check(
        self,
        current_values: Any,
        reference_values: Any,
    ) -> dict[str, list[str]]:
        """
        Determine validity of parameter according to one or more operations.

        Parameters
        -----------
        current_values : Any
            The test value(s). If multiple operations are specified, must be a Sequence
            of test values.
        reference_values : Any
            The value(s) to compare the test value(s) to. If multiple operations are
            specified, must be a Sequence of reference values.
        """

        checks: dict[str, list[str]] = {self.severity: []}

        if not isinstance(self.operation, list | tuple):
            operations: list[str | None] = [self.operation]
            current_values = [current_values]
            reference_values = [reference_values]
        else:
            operations = list(self.operation)

        for iop, operation in enumerate(operations):

            cval = current_values[iop]
            if isinstance(cval, str):
                cval = cval.upper()

            kwargs: dict[str, Any] = {}
            if operation == "approx" and isinstance(cval, float):
                kwargs.update({"rel_tol": self.tolerance, "abs_tol": 0.0})
            valid_value = self._comparator(cval, operation, reference_values[iop], **kwargs)

            if not valid_value:
                comment_str = (
                    f"INPUT SETTINGS --> {self.alias or self.name}: is {cval}, but should be "
                    f"{'' if operation == 'auto fail' else f'{operation} '}{reference_values[iop]}."
                )
                if self.comment:
                    comment_str += f"{' ' if len(self.comment) > 0 else ''}{self.comment}"
                checks[self.severity].append(comment_str)
        return checks


def _make_pythonic_defaults(config_path: str | Path | None = None) -> str:
    """Rerun this to regenerate VASP_DEFAULTS_LIST."""

    from monty.serialization import loadfn

    def format_val(val: Any) -> Any:
        if isinstance(val, str):
            return f'"{val}"'
        elif isinstance(val, float) and math.isinf(val):
            return 'float("inf")'
        return val

    config_path = config_path or Path(__file__).parent / "vasp_defaults.yaml"
    config = loadfn(config_path)

    return (
        "VASP_DEFAULTS_LIST = [\n"
        + ",\n".join(
            [f"    VaspParam({', '.join(f'{k} = {format_val(v)}' for k, v in param.items())})" for param in config]
        )
        + "\n]"
    )


VASP_DEFAULTS_LIST = [
    VaspParam(
        name="ADDGRID",
        value=False,
        operation="==",
        alias="ADDGRID",
        tag="fft",
        tolerance=0.0001,
        comment=None,
        warning=None,
        severity="reason",
    ),
    VaspParam(
        name="AEXX",
        value=0.0,
        operation=None,
        alias="AEXX",
        tag="hybrid",
        tolerance=0.0001,
        comment=None,
        warning=None,
        severity="reason",
    ),
    VaspParam(
        name="AGGAC",
        value=1.0,
        operation=None,
        alias="AGGAC",
        tag="hybrid",
        tolerance=0.0001,
        comment=None,
        warning=None,
        severity="reason",
    ),
    VaspParam(
        name="AGGAX",
        value=1.0,
        operation=None,
        alias="AGGAX",
        tag="hybrid",
        tolerance=0.0001,
        comment=None,
        warning=None,
        severity="reason",
    ),
    VaspParam(
        name="ALDAC",
        value=1.0,
        operation=None,
        alias="ALDAC",
        tag="hybrid",
        tolerance=0.0001,
        comment=None,
        warning=None,
        severity="reason",
    ),
    VaspParam(
        name="ALDAX",
        value=1.0,
        operation=None,
        alias="ALDAX",
        tag="hybrid",
        tolerance=0.0001,
        comment=None,
        warning=None,
        severity="reason",
    ),
    VaspParam(
        name="ALGO",
        value="normal",
        operation=None,
        alias="ALGO",
        tag="electronic_self_consistency",
        tolerance=0.0001,
        comment=None,
        warning=None,
        severity="reason",
    ),
    VaspParam(
        name="AMGGAC",
        value=1.0,
        operation=None,
        alias="AMGGAC",
        tag="hybrid",
        tolerance=0.0001,
        comment=None,
        warning=None,
        severity="reason",
    ),
    VaspParam(
        name="AMGGAX",
        value=1.0,
        operation=None,
        alias="AMGGAX",
        tag="hybrid",
        tolerance=0.0001,
        comment=None,
        warning=None,
        severity="reason",
    ),
    VaspParam(
        name="DEPER",
        value=0.3,
        operation="==",
        alias="DEPER",
        tag="misc",
        tolerance=0.0001,
        comment="According to the VASP manual, DEPER should not be set by the user.",
        warning=None,
        severity="reason",
    ),
    VaspParam(
        name="EBREAK",
        value=None,
        operation=None,
        alias="EBREAK",
        tag="post_init",
        tolerance=0.0001,
        comment="According to the VASP manual, EBREAK should not be set by the user.",
        warning=None,
        severity="reason",
    ),
    VaspParam(
        name="EDIFF",
        value=0.0001,
        operation="<=",
        alias="EDIFF",
        tag="electronic",
        tolerance=0.0001,
        comment=None,
        warning=None,
        severity="reason",
    ),
    VaspParam(
        name="EFERMI",
        value="LEGACY",
        operation=None,
        alias="EFERMI",
        tag="misc_special",
        tolerance=0.0001,
        comment=None,
        warning=None,
        severity="reason",
    ),
    VaspParam(
        name="EFIELD",
        value=0.0,
        operation="==",
        alias="EFIELD",
        tag="dipole",
        tolerance=0.0001,
        comment=None,
        warning=None,
        severity="reason",
    ),
    VaspParam(
        name="ENAUG",
        value=float("inf"),
        operation=None,
        alias="ENAUG",
        tag="electronic",
        tolerance=0.0001,
        comment=None,
        warning=None,
        severity="reason",
    ),
    VaspParam(
        name="ENINI",
        value=0,
        operation=None,
        alias="ENINI",
        tag="electronic",
        tolerance=0.0001,
        comment=None,
        warning=None,
        severity="reason",
    ),
    VaspParam(
        name="ENMAX",
        value=float("inf"),
        operation=">=",
        alias="ENCUT",
        tag="fft",
        tolerance=0.0001,
        comment=None,
        warning=None,
        severity="reason",
    ),
    VaspParam(
        name="EPSILON",
        value=1.0,
        operation="==",
        alias="EPSILON",
        tag="dipole",
        tolerance=0.0001,
        comment=None,
        warning=None,
        severity="reason",
    ),
    VaspParam(
        name="GGA_COMPAT",
        value=True,
        operation="==",
        alias="GGA_COMPAT",
        tag="misc",
        tolerance=0.0001,
        comment=None,
        warning=None,
        severity="reason",
    ),
    VaspParam(
        name="IALGO",
        value=38,
        operation="in",
        alias="IALGO",
        tag="misc_special",
        tolerance=0.0001,
        comment=None,
        warning=None,
        severity="reason",
    ),
    VaspParam(
        name="IBRION",
        value=-1,
        operation="in",
        alias="IBRION",
        tag="ionic",
        tolerance=0.0001,
        comment=None,
        warning=None,
        severity="reason",
    ),
    VaspParam(
        name="ICHARG",
        value=2,
        operation=None,
        alias="ICHARG",
        tag="startup",
        tolerance=0.0001,
        comment=None,
        warning=None,
        severity="reason",
    ),
    VaspParam(
        name="ICORELEVEL",
        value=0,
        operation="==",
        alias="ICORELEVEL",
        tag="misc",
        tolerance=0.0001,
        comment=None,
        warning=None,
        severity="reason",
    ),
    VaspParam(
        name="IDIPOL",
        value=0,
        operation="==",
        alias="IDIPOL",
        tag="dipole",
        tolerance=0.0001,
        comment=None,
        warning=None,
        severity="reason",
    ),
    VaspParam(
        name="IMAGES",
        value=0,
        operation="==",
        alias="IMAGES",
        tag="misc",
        tolerance=0.0001,
        comment=None,
        warning=None,
        severity="reason",
    ),
    VaspParam(
        name="INIWAV",
        value=1,
        operation="==",
        alias="INIWAV",
        tag="startup",
        tolerance=0.0001,
        comment=None,
        warning=None,
        severity="reason",
    ),
    VaspParam(
        name="ISIF",
        value=2,
        operation=">=",
        alias="ISIF",
        tag="ionic",
        tolerance=0.0001,
        comment="ISIF values < 2 do not output the complete stress tensor.",
        warning=None,
        severity="reason",
    ),
    VaspParam(
        name="ISMEAR",
        value=1,
        operation="in",
        alias="ISMEAR",
        tag="smearing",
        tolerance=0.0001,
        comment=None,
        warning=None,
        severity="reason",
    ),
    VaspParam(
        name="ISPIN",
        value=1,
        operation=None,
        alias="ISPIN",
        tag="misc_special",
        tolerance=0.0001,
        comment=None,
        warning=None,
        severity="reason",
    ),
    VaspParam(
        name="ISTART",
        value=0,
        operation="in",
        alias="ISTART",
        tag="startup",
        tolerance=0.0001,
        comment=None,
        warning=None,
        severity="reason",
    ),
    VaspParam(
        name="ISYM",
        value=2,
        operation="in",
        alias="ISYM",
        tag="symmetry",
        tolerance=0.0001,
        comment=None,
        warning=None,
        severity="reason",
    ),
    VaspParam(
        name="IVDW",
        value=0,
        operation="==",
        alias="IVDW",
        tag="misc",
        tolerance=0.0001,
        comment=None,
        warning=None,
        severity="reason",
    ),
    VaspParam(
        name="IWAVPR",
        value=None,
        operation=None,
        alias="IWAVPR",
        tag="misc_special",
        tolerance=0.0001,
        comment=None,
        warning=None,
        severity="reason",
    ),
    VaspParam(
        name="KGAMMA",
        value=True,
        operation=None,
        alias="KGAMMA",
        tag="k_mesh",
        tolerance=0.0001,
        comment=None,
        warning=None,
        severity="reason",
    ),
    VaspParam(
        name="KSPACING",
        value=0.5,
        operation=None,
        alias="KSPACING",
        tag="k_mesh",
        tolerance=0.0001,
        comment=None,
        warning=None,
        severity="reason",
    ),
    VaspParam(
        name="LASPH",
        value=True,
        operation="==",
        alias="LASPH",
        tag="misc",
        tolerance=0.0001,
        comment=None,
        warning=None,
        severity="reason",
    ),
    VaspParam(
        name="LBERRY",
        value=False,
        operation="==",
        alias="LBERRY",
        tag="misc",
        tolerance=0.0001,
        comment=None,
        warning=None,
        severity="reason",
    ),
    VaspParam(
        name="LCALCEPS",
        value=False,
        operation="==",
        alias="LCALCEPS",
        tag="misc",
        tolerance=0.0001,
        comment=None,
        warning=None,
        severity="reason",
    ),
    VaspParam(
        name="LCALCPOL",
        value=False,
        operation="==",
        alias="LCALCPOL",
        tag="misc",
        tolerance=0.0001,
        comment=None,
        warning=None,
        severity="reason",
    ),
    VaspParam(
        name="LCHIMAG",
        value=False,
        operation="==",
        alias="LCHIMAG",
        tag="chemical_shift",
        tolerance=0.0001,
        comment=None,
        warning=None,
        severity="reason",
    ),
    VaspParam(
        name="LCORR",
        value=True,
        operation=None,
        alias="LCORR",
        tag="misc_special",
        tolerance=0.0001,
        comment=None,
        warning=None,
        severity="reason",
    ),
    VaspParam(
        name="LDAU",
        value=False,
        operation=None,
        alias="LDAU",
        tag="dft_plus_u",
        tolerance=0.0001,
        comment=None,
        warning=None,
        severity="reason",
    ),
    VaspParam(
        name="LDAUJ",
        value=[],
        operation=None,
        alias="LDAUJ",
        tag="dft_plus_u",
        tolerance=0.0001,
        comment=None,
        warning=None,
        severity="reason",
    ),
    VaspParam(
        name="LDAUL",
        value=[],
        operation=None,
        alias="LDAUL",
        tag="dft_plus_u",
        tolerance=0.0001,
        comment=None,
        warning=None,
        severity="reason",
    ),
    VaspParam(
        name="LDAUTYPE",
        value=2,
        operation=None,
        alias="LDAUTYPE",
        tag="dft_plus_u",
        tolerance=0.0001,
        comment=None,
        warning=None,
        severity="reason",
    ),
    VaspParam(
        name="LDAUU",
        value=[],
        operation=None,
        alias="LDAUU",
        tag="dft_plus_u",
        tolerance=0.0001,
        comment=None,
        warning=None,
        severity="reason",
    ),
    VaspParam(
        name="LDIPOL",
        value=False,
        operation="==",
        alias="LDIPOL",
        tag="dipole",
        tolerance=0.0001,
        comment=None,
        warning=None,
        severity="reason",
    ),
    VaspParam(
        name="LEFG",
        value=False,
        operation="==",
        alias="LEFG",
        tag="write",
        tolerance=0.0001,
        comment=None,
        warning=None,
        severity="reason",
    ),
    VaspParam(
        name="LEPSILON",
        value=False,
        operation="==",
        alias="LEPSILON",
        tag="misc",
        tolerance=0.0001,
        comment=None,
        warning=None,
        severity="reason",
    ),
    VaspParam(
        name="LHFCALC",
        value=False,
        operation=None,
        alias="LHFCALC",
        tag="hybrid",
        tolerance=0.0001,
        comment=None,
        warning=None,
        severity="reason",
    ),
    VaspParam(
        name="LHYPERFINE",
        value=False,
        operation="==",
        alias="LHYPERFINE",
        tag="misc",
        tolerance=0.0001,
        comment=None,
        warning=None,
        severity="reason",
    ),
    VaspParam(
        name="LKPOINTS_OPT",
        value=False,
        operation="==",
        alias="LKPOINTS_OPT",
        tag="misc",
        tolerance=0.0001,
        comment=None,
        warning=None,
        severity="reason",
    ),
    VaspParam(
        name="LKPROJ",
        value=False,
        operation="==",
        alias="LKPROJ",
        tag="misc",
        tolerance=0.0001,
        comment=None,
        warning=None,
        severity="reason",
    ),
    VaspParam(
        name="LMAXMIX",
        value=2,
        operation=None,
        alias="LMAXMIX",
        tag="density_mixing",
        tolerance=0.0001,
        comment=None,
        warning=None,
        severity="reason",
    ),
    VaspParam(
        name="LMAXPAW",
        value=-100,
        operation="==",
        alias="LMAXPAW",
        tag="electronic_projector",
        tolerance=0.0001,
        comment=None,
        warning=None,
        severity="reason",
    ),
    VaspParam(
        name="LMAXTAU",
        value=6,
        operation=None,
        alias="LMAXTAU",
        tag="density_mixing",
        tolerance=0.0001,
        comment=None,
        warning=None,
        severity="reason",
    ),
    VaspParam(
        name="LMONO",
        value=False,
        operation="==",
        alias="LMONO",
        tag="dipole",
        tolerance=0.0001,
        comment=None,
        warning=None,
        severity="reason",
    ),
    VaspParam(
        name="LMP2LT",
        value=False,
        operation="==",
        alias="LMP2LT",
        tag="misc",
        tolerance=0.0001,
        comment=None,
        warning=None,
        severity="reason",
    ),
    VaspParam(
        name="LNMR_SYM_RED",
        value=False,
        operation="==",
        alias="LNMR_SYM_RED",
        tag="chemical_shift",
        tolerance=0.0001,
        comment=None,
        warning=None,
        severity="reason",
    ),
    VaspParam(
        name="LNONCOLLINEAR",
        value=False,
        operation="==",
        alias="LNONCOLLINEAR",
        tag="ncl",
        tolerance=0.0001,
        comment=None,
        warning=None,
        severity="reason",
    ),
    VaspParam(
        name="LOCPROJ",
        value=None,
        operation="==",
        alias="LOCPROJ",
        tag="misc",
        tolerance=0.0001,
        comment=None,
        warning=None,
        severity="reason",
    ),
    VaspParam(
        name="LOPTICS",
        value=False,
        operation="==",
        alias="LOPTICS",
        tag="tddft",
        tolerance=0.0001,
        comment=None,
        warning=None,
        severity="reason",
    ),
    VaspParam(
        name="LORBIT",
        value=0,
        operation=None,
        alias="LORBIT",
        tag="misc_special",
        tolerance=0.0001,
        comment=None,
        warning=None,
        severity="reason",
    ),
    VaspParam(
        name="LREAL",
        value="false",
        operation="in",
        alias="LREAL",
        tag="precision",
        tolerance=0.0001,
        comment=None,
        warning=None,
        severity="reason",
    ),
    VaspParam(
        name="LRPA",
        value=False,
        operation="==",
        alias="LRPA",
        tag="misc",
        tolerance=0.0001,
        comment=None,
        warning=None,
        severity="reason",
    ),
    VaspParam(
        name="LSMP2LT",
        value=False,
        operation="==",
        alias="LSMP2LT",
        tag="misc",
        tolerance=0.0001,
        comment=None,
        warning=None,
        severity="reason",
    ),
    VaspParam(
        name="LSORBIT",
        value=False,
        operation="==",
        alias="LSORBIT",
        tag="ncl",
        tolerance=0.0001,
        comment=None,
        warning=None,
        severity="reason",
    ),
    VaspParam(
        name="LSPECTRAL",
        value=False,
        operation="==",
        alias="LSPECTRAL",
        tag="misc",
        tolerance=0.0001,
        comment=None,
        warning=None,
        severity="reason",
    ),
    VaspParam(
        name="LSUBROT",
        value=False,
        operation="==",
        alias="LSUBROT",
        tag="misc",
        tolerance=0.0001,
        comment=None,
        warning=None,
        severity="reason",
    ),
    VaspParam(
        name="METAGGA",
        value=None,
        operation=None,
        alias="METAGGA",
        tag="dft",
        tolerance=0.0001,
        comment=None,
        warning=None,
        severity="reason",
    ),
    VaspParam(
        name="ML_LMLFF",
        value=False,
        operation="==",
        alias="ML_LMLFF",
        tag="misc",
        tolerance=0.0001,
        comment=None,
        warning=None,
        severity="reason",
    ),
    VaspParam(
        name="NELECT",
        value=None,
        operation=None,
        alias="NELECT",
        tag="electronic",
        tolerance=0.0001,
        comment=None,
        warning=None,
        severity="reason",
    ),
    VaspParam(
        name="NELM",
        value=60,
        operation=None,
        alias="NELM",
        tag="electronic_self_consistency",
        tolerance=0.0001,
        comment=None,
        warning=None,
        severity="reason",
    ),
    VaspParam(
        name="NELMDL",
        value=-5,
        operation=None,
        alias="NELMDL",
        tag="electronic_self_consistency",
        tolerance=0.0001,
        comment=None,
        warning=None,
        severity="reason",
    ),
    VaspParam(
        name="NLSPLINE",
        value=False,
        operation="==",
        alias="NLSPLINE",
        tag="electronic_projector",
        tolerance=0.0001,
        comment=None,
        warning=None,
        severity="reason",
    ),
    VaspParam(
        name="NSW",
        value=0,
        operation=None,
        alias="NSW",
        tag="startup",
        tolerance=0.0001,
        comment=None,
        warning=None,
        severity="reason",
    ),
    VaspParam(
        name="NWRITE",
        value=2,
        operation=">=",
        alias="NWRITE",
        tag="write",
        tolerance=0.0001,
        comment="The specified value of NWRITE does not output all needed information.",
        warning=None,
        severity="reason",
    ),
    VaspParam(
        name="POTIM",
        value=0.5,
        operation=None,
        alias="POTIM",
        tag="ionic",
        tolerance=0.0001,
        comment=None,
        warning=None,
        severity="reason",
    ),
    VaspParam(
        name="PREC",
        value="NORMAL",
        operation=None,
        alias="PREC",
        tag="precision",
        tolerance=0.0001,
        comment=None,
        warning=None,
        severity="reason",
    ),
    VaspParam(
        name="PSTRESS",
        value=0.0,
        operation="approx",
        alias="PSTRESS",
        tag="ionic",
        tolerance=0.0001,
        comment=None,
        warning=None,
        severity="reason",
    ),
    VaspParam(
        name="RWIGS",
        value=[-1.0],
        operation=None,
        alias="RWIGS",
        tag="misc_special",
        tolerance=0.0001,
        comment=None,
        warning=None,
        severity="reason",
    ),
    VaspParam(
        name="SCALEE",
        value=1.0,
        operation="approx",
        alias="SCALEE",
        tag="ionic",
        tolerance=0.0001,
        comment=None,
        warning=None,
        severity="reason",
    ),
    VaspParam(
        name="SIGMA",
        value=0.2,
        operation=None,
        alias="SIGMA",
        tag="smearing",
        tolerance=0.0001,
        comment=None,
        warning=None,
        severity="reason",
    ),
    VaspParam(
        name="SYMPREC",
        value=1e-05,
        operation=None,
        alias="SYMPREC",
        tag="symmetry",
        tolerance=0.0001,
        comment=None,
        warning=None,
        severity="reason",
    ),
    VaspParam(
        name="VCA",
        value=[1.0],
        operation=None,
        alias="VCA",
        tag="misc_special",
        tolerance=0.0001,
        comment=None,
        warning=None,
        severity="reason",
    ),
    VaspParam(
        name="WEIMIN",
        value=0.001,
        operation="<=",
        alias="WEIMIN",
        tag="misc",
        tolerance=0.0001,
        comment=None,
        warning=None,
        severity="reason",
    ),
]

VASP_DEFAULTS_DICT = {v.name: v for v in VASP_DEFAULTS_LIST}
