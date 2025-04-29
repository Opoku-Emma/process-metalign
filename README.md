# Microbiome Data Analysis Using Metalign Output and SQL Database Management
The goal of this project is to provide a streamlined and easy analysis of microbiome sequencing data that has been generated using metalign.
Given that you are new to [metalign](https://github.com/nlapier2/Metalign) and how it works, you might want to check out their documentation page [here](https://github.com/nlapier2/Metalign).


## Getting Started & Usage
### src/
contains necessary scripts to run analysis\
-- src\
-- assign_lineage_codes.py\
-- my_decorators.py\
-- preprocess_data.py\
-- sql_tables.py\
-- make_rows.py

### data/
Contains demo data for analysis.
- leaf_all.nostrain.txt
- leaf_phenotype.txt

### main_analysis.py
This runs a demo on how the script works. It reads in the metalign file and metadata file. Then it runs a simple analysis and returns a Principal Coordinate Analysis plot colored by site.

## Example
