# Import THEREDA Thermodynamic Database from json format

- [Import THEREDA Jupyter](import-THEREDA-json-data.ipynb)
- [Add extra GEMS3 records (IComp/DComp) on top of the THEREDA import](import-add-gems-records.ipynb)
- [Export THEREDA to GEMS3 Jupyter](export-THEREDA-GEMS3-data.ipynb)

Pipeline order: import → add extra GEMS3 records → export, all against the ArangoDB database `ORD_THEREDA_R17`.

## Changes (2026-10)

- **Database name**: all four notebooks/scripts now use `ORD_THEREDA_R17` (was the scratch database `ORD_THEREDA_enabled_scratch`).
- **`enabled` flag**: THEREDA `Declaration.enabled` is imported as `properties.enabled`; disabled records are skipped on export.
- **Consistency checks**
  - Import: `thereda_consistency.py` checks the source data at 298.15 K (molar mass, logK vs ΔrG, G/H/S/Cp); reports are written to `data-out/consistency/`. `FIX_MODE` is `warn-only` (data imported as is) or `prefer-logK`.
  - Export: `check_consistency`, `consistency_mode`, `derive_missing_g`, `skip_without_g` on `ExportToGems3`. A missing DFG298 is derived from the reaction; records that still have no G0 are not exported.
  - Reaction dHr/dSr/dCpr missing in THEREDA are derived from the logK(T) coefficients at export (constant-logK reactions are left empty).
- **Redlich-Kister**: the second parameter a1 changes sign on export (GEMS convention).
- **Compositions**: `data-in/Compos.backup.json` + generated `data-in/Compos.extra.json` are merged into `data-out/10_2026/Compos.backup.json` (154 records) and converted to `compos.*.pdb/.ndx` by a `json2db` that knows the `compos` keyword (rebuilt 2026-10-30 against GEMS3K `develop_optima` with the Optima solver).
- **GEMS split sets**: output is split into `cemdata`, `main.aqueous|solids|gases`, pure phases in `pure/`; SDref is one single file.
- **Gas phase**: no `g` phase record is exported on purpose. GEMS collects gas species (`g`, ideal) into the ideal gas phase automatically; `f` fluid species (PR78/PRSV/CORK) form their own fluid mixture the same way.

## Known issues in the THEREDA source data (left as published)

- **`O2<0>` CP298 inconsistent with its reaction.** `O2(g)` CP298 = 29.376 J/(mol·K); `O2<0>` CP298 = 249.9; the dissolution reaction's LOGKT (f = 9.984 on ln T) gives ΔCp = +191.1. This implies `O2<0>` CP298 = 220.5, not 249.9. The data file is **not modified** (the ThermoHub record stays faithful to the source); report to the THEREDA providers. The current consistency check does not compare ΔCp with the logK(T) function — see future work.
- `O2<0>` DFG298 (16593) vs the DRG-implied value (16576.15) differ by 16.8 J/mol.
- ~32 more warnings on 28 species in `data-out/consistency/thereda_consistency_report.txt`.

## Future work

- Check whether `O2(g)`/`H2(g)` `TCint = [[25],[25]]` (from the constant-Cp default's 273.15–298.15 K limits) is why GEMS gives logK = 0 above 25 °C for `O2@`/`H2@`; widen the interval if so.
- Add a ΔCp-vs-logK(T) check (CP298 of species vs reactants) to the import and export consistency checks.
- Current `O2(g)`/`H2(g)`/`CO2(g)` records have no `fluid_peng_robinson78` method — confirm the format-file edits are intended.
