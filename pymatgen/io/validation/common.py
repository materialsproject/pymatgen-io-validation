"""Common class constructor for validation checks."""

from __future__ import annotations
from math import isclose
from typing import TYPE_CHECKING, Literal

from pydantic import BaseModel

if TYPE_CHECKING:
    from typing import Any

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


class BasicValidator:
    """
    Compare test and reference values according to one or more operations.

    Parameters
    -----------
    global_tolerance : float = 1.e-4
        Default tolerance for assessing approximate equality via math.isclose
    """

    # avoiding dunder methods because these raise too many NotImplemented's

    def __init__(self, global_tolerance: float = 1.0e-4) -> None:
        """Set math.isclose tolerance"""
        self.tolerance = global_tolerance

    @staticmethod
    def _comparator(lhs: Any, operation: str, rhs: Any, **kwargs) -> bool:
        """
        Compare different values using one of VALID_OPERATIONS.

        Parameters
        -----------
        lhs : Any
            Left-hand side of the operation.
        operation : str
            Operation acting on rhs from lhs. For example, if operation is ">",
            this returns (lhs > rhs).
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
            c = isclose(lhs, rhs, **kwargs)
        else:
            raise InvalidOperation(operation)
        return c

    def _check_parameter(
        self,
        error_list: list[str],
        input_tag: str,
        current_value: Any,
        reference_value: Any,
        operation: str,
        tolerance: float | None = None,
        append_comments: str | None = None,
    ) -> None:
        """
        Determine validity of parameter subject to a single specified operation.

        Parameters
        -----------
        error_list : list[str]
            A list of error/warning strings to update if a check fails.
        input_tag : str
            The name of the input tag which is being checked.
        current_value : Any
            The test value.
        reference_value : Any
            The value to compare the test value to.
        operation : str
            A valid operation in self.operations. For example, if operation = "<=",
            this checks `current_value <= reference_value` (note order of values).
        tolerance : float or None (default)
            If None and operation == "approx", default tolerance to self.tolerance.
            Otherwise, use the user-supplied tolerance.
        append_comments : str or None (default)
            Additional comments that may be helpful for the user to understand why
            a check failed.
        """

        append_comments = append_comments or ""

        if isinstance(current_value, str):
            current_value = current_value.upper()

        kwargs: dict[str, Any] = {}
        if operation == "approx" and isinstance(current_value, float):
            kwargs.update({"rel_tol": tolerance or self.tolerance, "abs_tol": 0.0})
        valid_value = self._comparator(current_value, operation, reference_value, **kwargs)

        if not valid_value:
            error_list.append(
                f"INPUT SETTINGS --> {input_tag}: is {current_value}, but should be "
                f"{'' if operation == 'auto fail' else operation + ' '}{reference_value}."
                f"{' ' if len(append_comments) > 0 else ''}{append_comments}"
            )

    def check_parameter(
        self,
        reasons: list[str],
        warnings: list[str],
        input_tag: str,
        current_values: Any,
        reference_values: Any,
        operations: str | list[str],
        tolerance: float = None,
        append_comments: str | None = None,
        severity: Literal["reason", "warning"] = "reason",
    ) -> None:
        """
        Determine validity of parameter according to one or more operations.

        Parameters
        -----------
        reasons : list[str]
            A list of error strings to update if a check fails. These are higher
            severity and would deprecate a calculation.
        warnings : list[str]
            A list of warning strings to update if a check fails. These are lower
            severity and would flag a calculation for possible review.
        input_tag : str
            The name of the input tag which is being checked.
        current_values : Any
            The test value(s). If multiple operations are specified, must be a Sequence
            of test values.
        reference_values : Any
            The value(s) to compare the test value(s) to. If multiple operations are
            specified, must be a Sequence of reference values.
        operations : str
            One or more valid operations in VALID_OPERATIONS.
            For example, if operations = "<=", this checks
                `current_values <= reference_values`
            (note the order of values).

            Or, if operations == ["<=", ">"], this checks
            ```
            (
                (current_values[0] <= reference_values[0])
                and (current_values[1] > reference_values[1])
            )
            ```
        tolerance : float or None (default)
            Tolerance to use in math.isclose if any of operations is "approx". Defaults
            to self.tolerance.
        append_comments : str or None (default)
            Additional comments that may be helpful for the user to understand why
            a check failed.
        severity : Literal["reason", "warning"]
            If a calculation fails, the severity of failure. Directs output to
            either reasons or warnings.
        """

        severity_to_list = {"reason": reasons, "warning": warnings}

        if not isinstance(operations, list):
            operations = [operations]
            current_values = [current_values]
            reference_values = [reference_values]

        for iop in range(len(operations)):
            self._check_parameter(
                error_list=severity_to_list[severity],
                input_tag=input_tag,
                current_value=current_values[iop],
                reference_value=reference_values[iop],
                operation=operations[iop],
                tolerance=tolerance,
                append_comments=append_comments,
            )


class BaseValidator(BaseModel):
    """
    Template for validation classes.

    This class will check any function with the name prefix `_check_`.

    `_check_*` functions must take no args by default:

    def _check_example(self) -> None:
        if self.name == "whole mango":
            self.reasons.append("We only accept sliced or diced mango at this time.")
        elif self.name == "diced mango":
            self.warnings.append("We prefer sliced mango, but will accept diced mango.")

    Attrs:
        reasons : list[str]
            A list of error strings to update if a check fails. These are higher
            severity and would deprecate a calculation.
        warnings : list[str]
            A list of warning strings to update if a check fails. These are lower
            severity and would flag a calculation for possible review.
        name : str = "Base validator class"
            Name of the validator class
        fast : bool = False
            Whether to perform quick check.
            True: stop validation if any check fails.
            False: perform all checks.
    """

    reasons: list[str]
    warnings: list[str]
    name: str = "Base validator class"
    fast: bool = False

    def check(self) -> None:
        """
        Execute any checks on the class with a name prefix `_check_`.

        See class docstr for an example.
        """

        checklist = {attr for attr in dir(self) if attr.startswith("_check_")}
        for attr in checklist:
            if self.fast and len(self.reasons) > 0:
                # fast check: stop checking whenever a single check fails
                break
            
            getattr(self, attr)()
