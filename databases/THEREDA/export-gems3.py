# Headless equivalent of export-THEREDA-GEMS3-data.ipynb: THEREDA ThermoHub graph -> GEMS3 backup
# JSON. Run against a database already populated by import-THEREDA-json-data.ipynb. Prefer this
# script over the notebook when testing a thermomatch rebuild - a live Jupyter kernel can't
# hot-reload a rebuilt native extension, but `python export-gems3.py` always uses the current build.
import thermomatch as match

match.ThermoImpexGenerator.setResourcesDirectory("/home/dmiron/git/hub/thermoimpex-jupyter/Resources")

# Redox elements / formula overrides / symbol-shortening / SDref author / mode-code rules, loaded
# as process-wide state - must happen before ExportToGems3(...) below (see exportGems3.h / CLAUDE.md).
match.loadGems3ExportRulesFile("/home/dmiron/git/hub/thermoimpex-jupyter/databases/THEREDA/scripts-out/gems3-export-rules.json")

# Last arg is the idThermoDataSet string (not a list - a list silently matches the different
# reaction_keys overload and exports nothing).
gem_export = match.ExportToGems3("http://localhost:8529", "root", "", "ORD_THEREDA_2026-01",
                                  "thermodatasets/THEREDA2026;1:TDS_LMA;0")
gem_export.ExportAllJson("/home/dmiron/git/hub/thermoimpex-jupyter/databases/THEREDA/data-out")
print("done")
