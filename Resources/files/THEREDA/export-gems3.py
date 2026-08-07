import thermomatch as match

match.ThermoImpexGenerator.setResourcesDirectory("/home/dmiron/git/thermomatch/thermomatch/Resources")

gem_export = match.ExportToGems3("http://localhost:8529", "root", "", "ORD_THEREDA_2026-01",
                                  "thermodatasets/THEREDA2026;1:TDS_LMA;0")
gem_export.ExportAllJson("/home/dmiron/git/thermomatch/thermomatch/Resources/files/THEREDA/data-out")
print("done")
