# Headless equivalent of export-THEREDA-GEMS3-data.ipynb: THEREDA ThermoHub graph -> GEMS3 backup
# JSON. Run against a database already populated by import-THEREDA-json-data.ipynb. Prefer this
# script over the notebook when testing a thermomatch rebuild - a live Jupyter kernel can't
# hot-reload a rebuilt native extension, but `python export-gems3.py` always uses the current build.
import os
import thermomatch as match

# ExportAllJson(dir_path) derives its output files' "backup" tag from dir_path's own last path
# component when dir_path contains a "/" (see ExportToGems3::ExportAllJson in exportGems3.cpp) -
# an absolute/multi-segment path like ".../THEREDA/data-out" makes it tag files
# "IComp.data-out.json" etc instead of "IComp.backup.json", silently missing the
# "*.backup.json" names the json2db conversion step (and the notebook) expect. Chdir into this
# script's own directory and pass the bare "data-out" (no "/") to sidestep that, matching the
# notebook's own relative-path call exactly, while keeping this script runnable from any cwd.
os.chdir(os.path.dirname(os.path.abspath(__file__)))

match.ThermoImpexGenerator.setResourcesDirectory("../../Resources")

# Redox elements / formula overrides / symbol-shortening / SDref author / mode-code rules, loaded
# as process-wide state - must happen before ExportToGems3(...) below (see exportGems3.h / CLAUDE.md).
match.loadGems3ExportRulesFile("scripts-out/gems3-export-rules.json")

# Last arg is the idThermoDataSet string (not a list - a list silently matches the different
# reaction_keys overload and exports nothing).
gem_export = match.ExportToGems3("http://localhost:8529", "root", "", "ORD_THEREDA_2026-01",
                                  "thermodatasets/THEREDA2026;1:TDS_LMA;0")
gem_export.ExportAllJson("data-out")
print("done")
