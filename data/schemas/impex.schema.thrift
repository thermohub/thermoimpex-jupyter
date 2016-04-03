/** Apache Thrift IDL definition for text import-export interfaces */
/** PMATCH++ prototype   April 2016 */
/** Created by KD 03-4-2016 */

namespace * impex

/** Format string in scanf/printf syntax "%s" | "in" | "out" | "endl" | "txel" | "txkw" */
typedef string Format   // "in", "out", "endl" mean stream input (type depends on Data )
                        // "txel" means text until end of line; "txkw" means text until next keyword 
/** Intermediate value read from file (import) or to be printed to file */
typedef string Value      

/** Thrift key of data object "8" or "3.8" or "2.3.8" or "" to ignore (import); any string not starting from a digit as comment (export)  */
typedef string DataObject  
                                       
/** Export/import of each data record to/from (formatted) text file - strict data order */
/** Definition of the data value */
struct DataValue
{
	/** Format, Value pair (formats: "%s" | "in" | "out" | "endl" | "txel" | "txkw" ) */
	1: map<Format,Value> value 
    /** Value separator (for arrays) " " | "," | "\t" | "integer" (fixed field width) */
	2: string vs
	/** line separator "\n" ... */
	3: string ls 
	/** Factor != 0, default 1; Each num.value is multiplied (import) or divided (export) by factor */
	4: optional double factor 
	/** Increment, default 0; added to each numerical value (import) or subtracted from (export) */
	5: optional double increment 
	/** Default "" or contains ECMA function script for operation on data value TBD */
	6: optional string fscript  
}

/** Definition of text line in file, for mapping values to internal data objects */
struct TextLine {
    /** One or more Value,DataObject pairs */
	1: map<DataValue,DataObject> line 
	/** Line separator, default "\n" ... */
	2: optional string ls 
}
// number of actually exported data elements depends on the respective data object definition

/** Text block in file corresponding to one database document (record) */
struct TextBlock {
    /** One or more text lines with data */
	1: list <TextLine> block 
	/** Head block separator - any characteristic string e.g. "{" or just "\n" */
	2: string bs_head 
	/** end block separator - any characteristic string e.g. "}" or "\n" */
	3: string bs_end
	/** number of data items per block */
	4: optional i32 Ndata 
}

/** Definition of text data file */
struct TextFile {
    /** One or more blocks for data records */
	1: required list<TextBlock> blocks 
	/** Label of data type (vertex type), e.g. "datasource", "element" ... */
	3: required string label 
	/** Export: the whole comment text; Import: the comment begin markup string (to skip until endl) */
	4: optional string comment 
	/** File name or "console" for export */                  
	5: optional string fname 
	/** string indicating end of data in file or "" as default (end of file) */
	6: optional string eod  
	/** encoding ("" for standard system encoding) */
    7: optional string encoding 
    /** number of data block in file >=1, 0 if unknown */ 
	8: optional i32 Nblocks 
    /** number of text lines in file (>=1), 0 if unknown */ 
	9: optional i32 Nlines   
	/** total number of characters in file, 0 if unknown */
	10: optional i32 Nchars   
}

// Export/import of data records to table format file (strict order of columns and rows)
/** Definition of header column value */
struct HeaderColumnValue
{
	/** Format,Value pair for value(s) */
	1: map<Format,Value> value
	/** value separator (for arrays) */
	2: string vs 
	/** line separator "\n" ... */
	3: string ls 
}

/** Definition of the header row */
struct HeaderRow
{
	/** header columns (in the same order as data in data rows) */
	1: list<HeaderColumnValue> header_columns 
}

/** Definition of table header - can have 1 or more rows */
struct TableHeader {
    /** This list (table) to be analyzed to identify data objects */
	1: list<HeaderRow> header_rows 
	/** Mapping of header index to internal thrift data object keyword */
	2: map<i32,DataObject> col_keys 
    /** header item separator "," for csv file, "\t" for tsv, "" for fixed */
	3: string his
	/** end of header row separator e.g. "\n" for csv file */   
	4: optional string hs 
	/** Number of colums in the table >=1, 0 if unknown */     
	5: optional i32 Ncols
	/** Number of rows in the table header, default 1, 0 if unknown */ 
	6: optional i32 Nhrows
}

/** Definition of data column value */
struct DataColumnValue
{
	/** Format,Value pair for value(s) */
	1: map<Format,Value> value  
	/** value separator (for arrays), "," for csv, "\t" for tsv files, "<integer>" for fixed, etc. */
	2: string vs 
	/** line separator "\n" ... */
	3: string ls 
	/** Factor, != 0, default 1; Each num.value is multiplied (import) or divided (export) by factor */
	4: optional double factor 
	/** Increment, default 0; added to each numerical value (import) or subtracted from (export) */
	5: optional double increment 
	/** Default "" or contains ECMA function script for operation on data value TBD */
	6: optional string fscript     
}

/** Definition of table row */
struct TableRow {
    /** Array of column data forming table row. For mapping, see TableHeader (to avoid duplication) */
	1: list<DataColumnValue> col_data  
//	1: map<i32,DataObject> match 
    /** data item separator "," for csv file, "\t" for tsv, "<integer>" for fixed field width */
	2: string dis
	/** end of row separator e.g. "\n" for csv file */
	3: string rs 
}

// Definition of table text file
struct TableFile {
    /** Table header  */
	1: required TableHeader header       
    /** Table data (one or more rows) */
	2: required list<TableRow> rows      
	/** Label of data type (vertex type), e.g. "datasource", "element" ... */
	3: required string label 
	/** Export: the whole comment text; Import: the comment begin markup string (to skip until endl) */
	4: optional string comment 
	/** File name or "console" for export */                  
	5: optional string fname 
	/** String indicating end of data in file or "" as default (end of file) */
	6: optional string eod  
	/** Encoding ("" for standard system encoding) */
    7: optional string encoding 
    /** Number of colums in the table >=1, 0 if unknown */ 
    8: optional i32 Ncols
    /** Number of rows in the table header 1 or more, 0 if unknown */ 
	9: optional i32 Nhrows   
	/** Number of rows in the table data >=1, 0 if unknown */
	10: optional i32 Ndrows   
}

// Export/import of each data record to/from text file organized as block of key-value pairs
// Key-Value pairs may follow in arbitrary order in the input (imported) file
/** Keyword in key-value pair */
typedef string Keyword

/** Definition of key-value pair (line) in file */
struct KeyValuePair {
    /** Format,Data pair for keyword */
	1: map<Format,Keyword> keyword 
    /** Format,Data pair for value(s) */
	2: map<Format,Value> value  
	/** keyword- value separator */
	3: string kwvs 
	/** Factor != 0, default 1; Each num.value is multiplied (import) or divided (export) by factor */
	4: optional double factor 
	/** Increment, default 0; added to each numerical value (import) or subtracted from (export) */
	5: optional double increment 
	/** Default "" or contains ECMA function script for operation on data value TBD */
	6: optional string fscript 
    /** Text line separator, default "\n" ...*/
	7: optional string ls 
}        

/** Text block in file corresponding to one database document (record) */
struct KeyValueBlock {
    /** one or more keyword-value pairs */
	1: set<KeyValuePair> block  
	/** mapping of Keywords to (thrift-schema-defined) data */
	2: map<Keyword,DataObject> match // mapping of Keywords to (thrift-schema-defined) data
             // At import: if Data = "" then input for keyword is ignored, otherwise read from Value
             // according to the internal type of the object referenced in Data
             // At export: the Data = "" if the object is empty (this pair is then not written to file)
             // 
    /** head block separator - any characteristic string e.g. "{" or just "\n"   */      
	3: string bs_head 
	/** end block separator - any characteristic string e.g. "}" or "\n" */
	4: string bs_end  
}

/** Definition of text file with key-value pair data */
struct KeyValueFile {
    /** one or more blocks for data records */
	1: required list<KeyValueBlock> blocks 
	/** Label of data type (vertex type), e.g. "datasource", "element" ... */
	3: required string label 
	/** Export: the whole comment text; Import: the comment begin markup string (to skip until endl) */
	4: optional string comment 
	/** File name or "console" for export */                  
	5: optional string fname 
	/** string indicating end of data in file or "" as default (end of file) */
	6: optional string eod  
	/** encoding ("" for standard system encoding) */
    7: optional string encoding 
    /** max. number of keywords >= 1 in data block, 0 if unknown */ 
	8: optional i32 Nkeys   
	/** number of data blocks (records) >=1, 0 if unknown */
	9: optional i32 Nblocks 
	/** total number of text lines in the file, 0 if unknown */   
	10: optional i32 Nlines    	
}

// End of file impex.thrift