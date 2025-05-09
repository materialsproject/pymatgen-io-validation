pymatgen-io-validation
=====

This package is an extension to `pymatgen` for performing I/O validation. Specifically, this package checks for discrepancies between a specific calculation and a provided input set; it also checks for known bugs when certain input parameters are used in combination, alongside several other small checks. The motivation for creating this package was to ensure VASP calculations performed by groups outside of the Materials Project (MP) are compliant with MP data, thus enabling their raw data to be included in the MP Database.


Installation
=====

You can install this package by simply running

`pip install pymatgen-io-validation`


Usage
=====

For validating calculations from the raw files, run:
```
from pymatgen.io.validation import VaspValidator
validation_doc = VaspValidator.from_directory(path_to_vasp_calculation_directory)
```

In the above case, whether a calculation passes the validator can be accessed via `validation_doc.valid`. Moreover, reasons for an invalidated calculation can be accessed via `validation_doc.reasons` (this will be empty for valid calculations). Last but not least, warnings for potential issues (sometimes minor, sometimes major) can be accessed via `validation_doc.warnings`.

Contributors
=====

* [Matthew Kuner](https://github.com/matthewkuner) (lead), email: matthewkuner@gmail.com
* [Aaron Kaplan](https://github.com/esoteric-ephemera)
* [Janosh Riebesell](https://github.com/janosh)
* [Jason Munro](https://github.com/munrojm)


Rationale
====

| **Parameter** | **Reason** |
| ---- | ---- |
| ADDGRID | ADDGRID must be set to False. MP uses ADDGRID = False, as the VASP manual states "please do not use this tag [ADDGRID] as default in all your calculations!". ADDGRID can affect the outputted forces, hence all calculations are thus required to have ADDGRID = False for compatibility. |
| AEXX / AMGGAX / AMGGAC / AGGAX / ALDAC / ALDAX | These parameters should be the VASP defaults unless otherwise specified in a given MP input set, as changing them is effectively a change to the level of theory. |
| ALGO / IALGO | ALGO must be one of: "Normal", "Conjugate", "All", "Fast", "Exact". (This corresponds to an IALGO of 38, 58, 58, 68, 90, respectively). |
| DEPER / EBREAK / WEIMIN | DEPER, EBREAK, and WEIMIN should not be changed according to the VASP wiki, hence MP requires them to remain as their default values. |
| EDIFF | EDIFF must be equal to or greater than the value in the relevant MP input set. This will ensure compatibility between results with those in the MP Database. |
| EDIFFG | Should be the same or better than in the relevant MP input set. For MP input sets with an energy-based cutoff, the calculation must have an energy change between the last two steps be less in magnitude than the specified EDIFFG (so even if your calculation uses force-based convergence, the energy must converge within the MP input set’s specification). The same logic applies to MP input sets with force-based EDIFFG settings. |
| EFERMI | EFERMI must be one of: "LEGACY", "MIDGAP" |
| EFIELD | Current MP input sets used to construct the main MP database do not set EFIELD, and hence we require this to be unset or set to 0. |
| ENAUG | If ENAUG is present in the relevant MP input set, then a calculation’s ENAUG must be equal to or better than that value. |
| ENCUT | ENCUT must be equal to or greater than the value in the relevant MP input set, as otherwise results will likely be incompatible with MP. |
| ENINI | ENINI must not be adjusted for high-throughput calculations, and hence should be left equal to the value used for ENCUT in the relevant MP input set. Values greater than the ENCUT in the relevant MP input set will also be accepted, though we expect this to be uncommon and do not recommend it. |
| EPSILON | EPSILON must be set to the VASP default of 1. Changing the dielectric constant of the medium will cause results to be incompatible with MP. |
| GGA / METAGGA | The level of theory used must match the relevant MP input set. Moreover, GGA and METAGGA should never be set simultaneously, as this has been shown to result in seriously erroneous results. See https://github.com/materialsproject/atomate2/issues/453#issuecomment-1699605867 for more details. |
| GGA_COMPAT | GGA_COMPAT must be set to the VASP default of True. The VASP manual only recommends setting this to False for noncollinear magnetic calculations, which are not currently included in the MP database. |
| IBRION | IBRION values must be one of: -1, 1, 2. Other IBRION values correspond to non-standard forms of DFT calculations that are not included in the MP database. (Note that, while phonon data is included in the MP database, such values are not calculated using, say, IBRION = 5. Such logic applies to all other IBRION values not allowed). |
| ICHARG | ICHARG must be set to be compatible with the calculation type. For example, if the relevant MP input set is for a SCF calculation, ICHARG $\leq 9$ must be used. For NSCF calculations, the value for ICHARG must exactly match the value contained in the relevant MP input set. |
| ICORELEVEL | ICORELEVEL must be set to 0. MP does not explicitly calculate core energies. |
| IDIPOL | IDIPOL must be set to 0 (the VASP default). |
| IMAGES | IMAGES must be set to 0 to match MP calculations. |
| INIWAV | INIWAV must be set as the VASP default of 1 to be consistent with MP calculations. |
| ISPIN | All values of ISPIN are allowed, though it should be noted that virtually all MP calculations permit spin symmetry breaking, and have ferromagnetic, antiferrogmanetic, or nonmagnetic ordering. |
| ISMEAR | The appropriate ISMEAR depends on the bandgap of the material (which cannot be known a priori). As per the VASP manual: for metals (bandgap = 0), any ISMEAR value in [0, 1, 2] is acceptable. For nonmetals (bandgap > 0), any ISMEAR value in [-5, 0] is acceptable. Hence, for those who are performing normal relaxations/static calculations and want to ensure their calculations are MP-compatible, we recommend setting ISMEAR to 0. |
| ISIF | MP allows any ISIF $\geq 2$. This value is restricted as such simply because all ISIF values $\geq 2$ output the complete stress tensor. |
| ISYM | ISYM must be one of -1, 0, 1, 2, except for when the relevant MP input set uses a hybrid functional, in which case ISYM=3 is also allowed. |
| ISTART | ISTART must be one of: 0, 1, 2. |
| IVDW | IVDW must be set to 0. MP currently does not apply any vdW dispersion corrections. |
| IWAVPR | IWAVPR must be set to 0 (the default). VASP discourages users from setting this tag. |
| KGAMMA | KGAMMA must be set to True (the VASP default). This is only relevant when no KPOINTS file is used. |
| KSPACING / KPOINTS | The KSPACING parameter or KPOINTS file must correspond with at least 0.9 times **the number of KPOINTS in the non-symmetry-reduced Brillouin zone specified by the relevant MP input set** (i.e., not the number of points in the irreducible wedge of the first Brillouin zone). Hence, either method of specifying the KPOINTS can be chosen. This ensures that a calculation uses a comparable number of kpoints to MP. |
| Kpoint mesh type (for KPOINTS) | The type of Kpoint mesh must be valid for the symmetry of the crystal structure in the calculation. For example, for a hexagonal closed packed structure, one must use a $\Gamma$-centered mesh. All Kpoints generated using Pymatgen *should* be valid. |
| LASPH | LASPH must be set to True (this is **<u>*not*</u>** the VASP default). |
| LCALCEPS | LCALCEPS must be set to False (the VASP default). |
| LBERRY | LBERRY must be set to False (the VASP default). |
| LCALCPOL | LCALCPOL  must be set to False (the VASP default). |
| LCHIMAG | LCHIMAG must be set to False (the VASP default). |
| LCORR | LCORR must be set to True (the VASP default) for calculations with IALGO = 58. |
| LDAU / LDAUU / LDAUJ / LDAUL / LDAUTYPE | For DFT$`+U`$ calculations, all parameters corresponding to $+U$ or $+J$ corrections must exactly match those specified in the relevant MP input set. Alternatively, LDAU = False (DFT) is always acceptable. |
| LDIPOL | LDIPOL must be set to False (the VASP default). |
| LMONO | LMONO must be set to False (the VASP default). |
| LEFG | LEFG must be set to False (the VASP default), unless explicitly specified to be True by the relevant MP input set. |
| LEPSILON | LEPSILON must be set to False (the VASP default), unless explicitly specified to be True by the relevant MP input set. |
| LHFCALC | The value of LHFCALC should match that of the relevant MP input set, as it will otherwise result in a change in the level of theory applied in the calculation. |
| LHYPERFINE | LHYPERFINE must be set to False (the VASP default). |
| LKPROJ | LKPROJ must be set to False (the VASP default). |
| LKPOINTS_OPT | LKPOINTS_OPT must be set to False. |
| LMAXPAW | LMAXPAW must remain unspecified, as the VASP wiki states that "Energies should be evaluated with the default setting for LMAXPAW". |
| LMAXMIX | LMAXMIX must be set to 6. This is based on tests from Aaron Kaplan (@esoteric-ephemera) — see the "bench_vasp_pars.docx" document in https://github.com/materialsproject/pymatgen/issues/3322. |
| LMAXTAU | LMAXTAU must be set to 6 (the VASP default when using LASPH = True). |
| LMP2LT / LSMP2LT | Both must be set to False (VASP defaults) |
| LNONCOLLINEAR / LSORBIT | Both must be set to False (VASP defaults) |
| LOCPROJ | LOCPROJ must be set to None (the VASP default). |
| LOPTICS | LOPTICS must be set to False (the VASP default), unless explicitly specified by the relevant MP input set. |
| LORBIT | LORBIT must **<u>*not*</u>** be None if the user also sets ISPIN=2, otherwise all values of LORBIT are acceptable. This is due to magnetization values not being output when ISPIN=2 and LORBIT = None are set together. |
| LREAL | If the LREAL in the relevant MP input set is "Auto", then the user must be one of: "Auto", False. Otherwise, if the LREAL in the relevant MP input set is False, then the user must use False. |
| LRPA | LRPA must be set to False (the VASP default). MP does not currently support random phase approximation (RPA) calculations. |
| LSPECTRAL | LSPECTRAL must be set to False (the VASP default for most calculations). |
| LSUBROT | LSUBROT must be set to False (the VASP default). |
| MAGMOM | While any initial magnetic moments are allowed, the final total magnetic moment for any given atom must be less than 5 $\mu_B$ (Bohr magnetons) (except for elements Gd and Eu, which must be less than 10 $\mu_B$). This simply serves as a filter for erroneous data. |
| ML_LMFF | ML_LMFF must be set to False (the VASP default). |
| NGX / NGY / NGZ | The values for NGX/NGY/NGZ must be at least 0.9 times the default value for the respective parameter generated by VASP. If the user simply does not specify these parameters, the calculation should be compatible with MP data. |
| NGFX / NGFY / NGFZ | The values for NGFX/NGFY/NGFZ must be at least 0.9 times the default value for the respective parameter generated by VASP. If the user simply does not specify these parameters, the calculation should be compatible with MP data. |
| NLSPLINE | NLSPLINE should be set to False (the VASP default), unless explicitly specified by the relevant MP input set. |
| NBANDS | NBANDS must be greater than the value $\mathrm{ceil}(\mathrm{NELECT}/2) + 1 $(minimum allowable number of bands to avoid degeneracy) and less than 4 times (minimum allowable number of bands to avoid degeneracy). For high-throughput calculations, it is generally recommended to not set this parameter directly. See https://github.com/materialsproject/custodian/issues/224 for more information. |
| NELECT | NELECT must not be changed from the default value VASP would use for the particular structure and pseudopotentials calculated. The easiest way to ensure that NELECT is compliant with MP data is to simply not specify NELECT in the INCAR file. |
| NWRITE | NWRITE must be set to be $\geq 2$ (the VASP default is 2). |
| POTIM | POTIM $\leq 5$. We suggest not setting POTIM in the INCAR file, and rather allowing VASP to set it to the default value of 0.5. |
| PSTRESS | PSTRESS must be set to exactly 0.0 (the VASP default). |
| PREC | PREC must be one of: "High", "Accurate". |
| ROPT | ROPT should be set to be less than or equal to the default ROPT value (which is set based on the PREC tag). Hence, it is recommended to not set the ROPT tag in the INCAR file. |
| RWIGS | RWIGS should not be set in the INCAR file. |
| SCALEE | SCALEE should not be set in the INCAR file. |
| SYMPREC | SYMPREC must be less than or equal to 1e-3 (as this is the maximum value that the Custodian package will set SYMPREC as of March 2024). For general use, we recommend leaving SYMPREC as the default generated by your desired MP input set. |
| SIGMA | There are several rules for setting SIGMA:  <ol> <li> SIGMA  must be $\leq 0.05$ for non-metals (bandgap $>  0$). </li> <li> SIGMA must be $\leq 0.2$ for a metal (bandgap = 0). </li> <li> For metals, the SIGMA value must be small enough that the entropy term in the energy is $\leq$ 1 meV/atom (as suggested by the VASP manual). </li> </ol> |
| VCA | MP data does not include Virtual Crystal Approximation (VCA) calculations from VASP. As such, this parameter should not be set. |
| VASP version | The following versions of VASP are allowed: 5.4.4 or $>$ 6.0.0. For example, versions $<=$ 5.4.3 are not allowed, whereas version 6.3.1 is allowed. |
