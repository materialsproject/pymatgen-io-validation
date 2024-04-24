""" Common class constructor for validation checks. """

from dataclasses import dataclass

@dataclass
class BaseValidator:
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
    reasons : list[str]
    warnings : list[str]
    name : str = "Base validator class"
    fast : bool = False

    def check(self) -> None:
        """
        Execute any checks on the class with a name prefix `_check_`.

        See class docstr for an example.
        """
                
        checklist = set(attr for attr in dir(self) if attr.startswith("_check_"))
        for attr in checklist:

            if self.fast and len(self.reasons) > 0:
                # fast check: stop checking whenever a single check fails
                break

            getattr(self,attr)()
