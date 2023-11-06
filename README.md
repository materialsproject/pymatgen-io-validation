pymatgen-io-validation
=====

This package is an extension to `pymatgen` for performing I/O validation. Specifically, this package checks for discrepancies with a provided input set, known bugs when certain input parameters are used in combination, and several other small checks. The initial motivation for creating this package was to validate calculations performed by groups outside of the Materials Project (MP), enabling their raw data to be included in the MP Database.


Usage
=====

For validating calculations from the raw files, run:
```
from emmet.core.vasp.validation.validation import ValidationDoc
validation_doc = ValidationDoc.from_directory(dir_name = path_to_vasp_calculation_directory)
```

In the above case, whether a calculation passes the validator can be accessed via `validation_doc.is_valid`. Moreover, reasons for an invalidated calculation can be accessed via `validation_doc.reasons` (this will be empty for valid calculations). Last but not least, warnings for potential issues (sometimes minor, sometimes major) can be accessed via `validation_doc.warnings`.
\
\
For validating calculations from `TaskDoc` objects from the [Emmet](https://github.com/materialsproject/emmet) package, run:
```
from emmet.core.vasp.validation.validation import ValidationDoc
validation_doc = ValidationDoc.from_task_doc(task_doc = my_task_doc)
```

Contributors
=====

* Matthew Kuner (lead), email: matthewkuner@gmail.com
* Janosh Riebesell
* Jason Munro
* Aaron Kaplan
