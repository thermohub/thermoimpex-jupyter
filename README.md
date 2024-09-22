# <img src="Imp-exp-foreign.png" width="100" height="100"> </br> thermoimpex-jupyter 
Import / Export to / from [ThermoHub](https://thermohub.org/thermohub/thermohub/) using jsonimpex scripts 

A collection of Jupyter notebooks used to import(export) foreign format data to and from [ThermoHub](https://thermohub.org/thermohub/thermohub/) database. ThermoHub aims to ensure the consistency traceability and completeness of thermodynamic datasets within a unified thermodynamic database in an general JSON format, accessible from zenodo, gihub or [db.ThermoHub.org](https://db.thermohub.org), with datasets ready to use for geochemical modelling applications.

Provided the data file in original foreign format and the respective import scripts, using jsonimpex library and ThermoMatch, the workflow and specific data operations are given in a jupyter notebook. This allows to read the data from the foreign file and match text, keys, columns, to the respective fields in the ThermoHub format. If provided in a bibJSON format, the bibliographic references are connected to the imported data. 

Arbitrary foreign format types: 
* Format Structured data file: JSON, YAML, or XML, nested structures
* Format Table data file: comma, space, tab, ..., separated file
* Format Key values file: key value

Import Thermodynamic Databases

- [PSI/Nagra TDB2020](/databases/PSINA-TDB2020/readme.md)
- [ThermoChimie](/databases/THERMOCHIMIE/readme.md)
- [SUCPRT](/databases/SUPCRT/readme.md)
- [CEMDATA](/databases/CEMDATA/readme.md)
- [NASA](/databases/NASA/readme.md)
- [MINES](/databases/MINES/readme.md)
- [HERACLES](/databases/HERACLES/readme.md)
- [CODATA](/databases/CODATA/readme.md)
- **New databases are always added**

Propose a new database to be added in the [issue tracker](https://github.com/thermohub/thermoimpex-jupyter/issues)

## ThermoHub

Formats for data types in the [ThermoHub](https://thermohub.org/thermohub/thermohub/) database are described by their respective [JSON schemas](https://github.com/thermohub/thermoimpex-jupyter/tree/main/Resources/data/schemas). GitHub [ThermoHub](https://github.com/thermohub/thermohub) and Zenodo [ThermoHub](https://zenodo.org/records/7385311).


## Acknowledgements  
This project was supported by the [Open Research Data Program](https://ethrat.ch/en/eth-domain/open-research-data/) of the ETH Board.