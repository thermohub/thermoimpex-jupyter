# Import Table Data from CSV

This functionality allows to import data from csv files previously exported from a spreadsheet. Any number of files and data types (elements, substances, reactions) can be imported at once.

## Prepare your csv files

Let us assume that you have an excel file with four spreadsheets containing: data for elements, data for master species, reactions for product aqueous species, and reactions for solids. Each spreadsheet is saved to a corresponding csv file. Give the files some reasonable names, for example: `elements.csv`, `master_aqueous.csv`, `product_aqueous.csv`, `product_solids.csv`.

One can name the files using multiple parts separated by '.' . For example: `elements.mydata2020.csv`, `master_aqueous.mydata2020.csv`, etc. What is important is that files that we want to import at once should all have the same segment names after the first '.', in this case `mydata2020`.

## Prepare your import scripts

If import scripts that describe the csv files you want to import are not available in the database these need to be produced. The main idea is to map (connect) the name of the columns in the csv file with the path to the respective property in the record that will be imported.

(details on preparing import scripts, TBD)

## Import Table Data from CSV dialog

The import dialog has 4 sections.

### Settings:

Import/Export (mode); ticked to overwrite records that we import if they are already present; ticked to read the import scripts from the database and not from local files.

### Import Setup:

Here we define each file and data type we want to import and what import script should be used. If we only want to import one file we only have define one row. 

There are some combinations possible, for example, in one import run, the same file can be used to import and create reaction records but also substance records for the substances defined by these reactions.

### Import Setup row editor:

Row editor where we set the desired options for each row of the Import Setup.

* Collection: defines the database record type

VertexElements - for importing element records
VertexSubstances - for importing substance records
VertexReactions - for importing reaction records

* Block title or file: defines the name of the block or file (segment until first ".") where to read the data from. E.g. `master_aqueous` to read data from `master_aqueous.mydata2020.csv`

If the file has multiple segments, this is not the full name but just the segment until the first dot ".". It is also possible that we have one file with different data types, then `master_aqueous` would represent the marker for the start of the block containing the data for master aqueous species.

* Script Type: defines the type of import script

FormatKeyValueFile - for scripts to import from key - value, JSON like type of data files.
FormatTableFile - for scripts to import from table like, csv type of data files.

* Script key or file: defines the path or database key of the import script.

If `Read import from database` is ticked the `...` will allow us to select from existing script saved in the database.

Un-ticked `Read import from database` allows us to select an import script file stored locally.

* Condition: defines if upon reading records or records and graph links are created.

This condition only has an effect in case `VertexReaction` is selected as Collection. Two options are available:

`records`: imports reaction data and only creates reaction records
`records_and_links`: imports reaction data and creates reaction records and links to the corresponding reactants (master and product substances).

The links are created by parsing the reaction equation. If a reactant is not found in the database (searched by symbol) an error will appear and the process will be halted.

### Path to Import file/files:

Even if we import multiple files only one file has to be selected here. The rest of the files will be defined bellow. If the file names contain multiple segments separated by "." all segment names after the first dot have to be the same for all files we want to import in one run. For example if the name of the selected file is `elements.mydata2020.csv`, all the other files should be named `<what segment name we want>.mydata2020.csv`.

## Example

We want to import data containing elements, substances and reactions as master aqueous and product aqueous, from the following csv files exported from excel: `elements.mydata2020.csv`, `master_aqueous.mydata2020.csv`, `product_aqueous.mydata2020.csv`. The `product_aqueous.mydata2020.csv` file we will use to import substance and reaction records for the product substances, as well as creating the links between the reactions and the reactants.

* Import Setup. For this we have to create 4 rows to: import elements, import master aqueous substances, import product substances, and import reactions with graph links.

The rows should look like this:

| 1 (block title or file)  | 2 (Collection)  | 3 (Condition)  | 4 (Script type)  | 5 (Script key or file)  |
|---|---|---|---|---|
| elements  | VertexElement  | records  | FormatTableFile  | db id or path to script file  |
| master_aqueous  | VertexSubstance  | records  | FormatTableFile  | db id or path to script file  |
| product_aqueous  | VertexSubstance  | records  | FormatTableFile  | db id or path to script file  |
| product_aqueous  | VertexReaction  | records_and_links  | FormatTableFile  | db id or path to script file  |

To create a row click `Add` and click to select the new created row. In the `Edit selected row` section choose all the respective options as shown in the table above. When finished click `Submit`.

We start with the import of elements and set: `Block title or file` to `elements` (this is the first part, before '.', of how we named the files!), `Collection` to `VertexElement`, `Condition` to `records` (no links need to be created), `Script Type` to `FormatTableFile`, and `Script key or file` to `path to the import script file or database key` (if Read Import script from database is ticked we can read the script from the database).

Do this for all the remaining files and conditions.

As you can see `product_aqueous` or `product_aqueous.mydata2020.csv` file is used twice. Once for importing records for product substances and another time for importing the reactions.

* Now select the path of one of the files we want to import (any), Browse/Enter File to Import `...`. In this case we choose `elements.mydata2020.csv`. Any other file should also work. It is important that all other files should have `mydata2020.csv` in common.

This setup is valid for any other group of files containing data with the same structure. We can now select `elements.mydata2019.csv` to import a different set of data.

![Import CSV Dialog][dialog-after]

[dialog-after]: images/import-csv-after.png "Import CSV Dialog"
[dialog-before]: images/import-csv-before.png "Import CSV Dialog"